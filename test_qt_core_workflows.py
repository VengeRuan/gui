"""Small, repeatable regression checks for the Qt migration's core workflows."""

from __future__ import annotations

import os
import tempfile
import csv
import json
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
from scipy import signal
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

import qt_gui as qt_gui_module
from qt_data_model import ArraySource, LazyH5Source, filter_array
from h5_provenance import read_h5_provenance
from snr_gui import RESTING_MULTIBAND_DEFINITIONS, compute_resting_multiband_snr, compute_resting_snr
from qt_gui import (
    BadChannelWorker,
    AnalysisCache,
    IcaWorker,
    LeaveOneOutMedianCARWorker,
    ItpcWorker,
    LfpWorker,
    RestingMultibandWorker,
    QtAnalysisGUI,
    compute_task_continuous_psd_metrics,
    compute_task_psd_metrics_by_run_length,
    compute_task_window_spectral_metrics,
    compute_external_baseline_reference,
    compute_external_rest_cluster_test,
    compute_external_rest_tf_cluster_test,
    ExternalTimeFrequencyWorker,
    TrialVepWorker,
    TaskEpochWorker,
    evaluate_task_trial_window_quality,
)


def run_worker(worker):
    output, errors = [], []
    worker.completed.connect(lambda *args: output.append(args))
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors, errors
    assert output, "worker did not emit completion"
    return output[0]


def main() -> int:
    app = QApplication.instance() or QApplication([])
    rng = np.random.default_rng(1)
    source = ArraySource(rng.normal(size=(1000, 2)).astype(np.float32), 250, time_offset=1.25, channel_ids=[7, 500])

    analysis_cache = AnalysisCache(data_budget_bytes=32_000)
    cached_columns = np.asarray([0], dtype=np.int64)
    cached_first = analysis_cache.get_channel_data(source, cached_columns)
    cached_second = analysis_cache.get_channel_data(source, cached_columns)
    assert cached_first is cached_second
    analysis_cache.put_saturation_metric(source, 7, 1.0, {"bottom_ratio": .4})
    assert analysis_cache.get_saturation_metric(source, 7, 1.0)["bottom_ratio"] == .4
    assert analysis_cache.get_saturation_metric(source, 7, 2.0) is None
    result_key = ("result", 1)
    analysis_cache.put_result(result_key, {"rows": [{"channel": 7}]})
    copied_result = analysis_cache.get_result(result_key)
    copied_result["rows"][0]["channel"] = 99
    assert analysis_cache.get_result(result_key)["rows"][0]["channel"] == 7

    filtered, _ = filter_array(source.data, 250, "bandpass", .5, 80, notch=True, channels=[0])
    assert filtered.shape == source.data.shape and np.allclose(filtered[:, 1], source.data[:, 1])

    # Resting multi-band SNR keeps the legacy ratio definition while masking
    # mains harmonics from both numerator and denominator.
    fs = 1000.0
    t = np.arange(12000, dtype=float) / fs
    resting = (.08 * np.sin(2 * np.pi * 10 * t) + .05 * np.sin(2 * np.pi * 50 * t)
               + .02 * rng.normal(size=t.size)).astype(np.float32)
    multiband = compute_resting_multiband_snr(resting, fs)
    assert multiband["line_metrics"]["line_50_hz"]["peak_db"] > 0
    assert multiband["bands"]["low_gamma"]["effective_bandwidth_hz"] < 50
    assert np.isfinite(compute_resting_snr(resting, fs, (30, 80), (1, 200)))
    unsupported = compute_resting_multiband_snr(resting[:6000], 300.0)
    assert not unsupported["bands"]["high_gamma"]["available"]

    multiband_source = ArraySource(np.column_stack([resting, resting * .5]), fs, channel_ids=[7, 500])
    default_multiband_worker = RestingMultibandWorker(
        multiband_source,
        settings={"target_frequency_resolution": .5},
    )
    assert default_multiband_worker.max_workers == 20
    multi_worker, = run_worker(default_multiband_worker)
    assert len(multi_worker["rows"]) == 2
    assert multi_worker["workers"] == 2
    assert "quality_grade" in multi_worker["rows"][0]
    serial_multiband, = run_worker(RestingMultibandWorker(
        multiband_source, settings={"target_frequency_resolution": .5, "workers": 1},
    ))
    for parallel_row, serial_row in zip(multi_worker["rows"], serial_multiband["rows"]):
        assert parallel_row["channel"] == serial_row["channel"]
        for band_name, _, _ in RESTING_MULTIBAND_DEFINITIONS:
            assert np.isclose(
                parallel_row["bands"][band_name]["snr_db"],
                serial_row["bands"][band_name]["snr_db"],
                equal_nan=True,
            )

    # A filtering selection is made against input physical channels.  When a
    # remap moves ch7 to FPC2, the worker must filter FPC2 rather than raw
    # input column 0 / FPC1.
    original_remap_reader = qt_gui_module.read_channel_remap_for_ids
    qt_gui_module.read_channel_remap_for_ids = lambda path, channel_ids: np.asarray([2, 1], dtype=np.int64)
    try:
        remapped_source, processed_source, _, remap_filter_info = run_worker(
            qt_gui_module.PreprocessWorker(
                source, "test-remap.xlsx", "highpass", 1.0, 80.0,
                channels=np.asarray([0], dtype=np.int64), notch=False,
            )
        )
    finally:
        qt_gui_module.read_channel_remap_for_ids = original_remap_reader
    assert remap_filter_info["filtered_channel_ids"] == {2}
    assert remapped_source.metadata.provenance["operations"][-1]["name"] == "channel_remap"
    assert processed_source.metadata.provenance["operations"][-1]["name"] == "filter"

    # Page 4 must preserve physical channel IDs when it intentionally limits
    # SNR to the consistently preprocessed subset.
    lfp_rows, = run_worker(LfpWorker(source, (30, 80), (1, 100), columns=[1]))
    assert len(lfp_rows) == 1 and lfp_rows[0]["channel"] == 500
    assert lfp_rows[0]["channel_scope"] == "filtered_subset"

    bad_channel_source = ArraySource(
        np.column_stack([np.zeros(1000), source.data[:, 1]]),
        250, channel_ids=[7, 500],
    )
    parallel_worker = BadChannelWorker(
        bad_channel_source,
        {"fast_artifact_check": True, "parallel": True, "workers": "2"},
    )
    parallel_progress = []
    parallel_worker.progress.connect(lambda _value, message: parallel_progress.append(message))
    good, bad, candidates = run_worker(parallel_worker)
    assert 7 in bad and 500 in good and isinstance(candidates, dict)
    serial_good, serial_bad, serial_candidates = run_worker(BadChannelWorker(
        bad_channel_source,
        {"fast_artifact_check": True, "parallel": False, "workers": "8"},
    ))
    assert (serial_good, serial_bad, serial_candidates) == (good, bad, candidates)
    assert any("通道并行 2" in message for message in parallel_progress)
    # Retired flatness parameters must not classify a channel as bad.
    half_flat = np.concatenate([
        np.zeros(500, dtype=np.float32),
        rng.normal(size=500).astype(np.float32),
    ])
    isolated_source = ArraySource(half_flat[:, None], 100, channel_ids=[11])
    isolated_good, isolated_bad, _ = run_worker(BadChannelWorker(
        isolated_source,
        {"flat_std": "1e-4", "flat_ratio": "40", "flat_time_check": True,
         "global_flat_check": True, "discrete_level_check": True,
         "parallel": False, "workers": "1"},
    ))
    assert isolated_good == [11] and not isolated_bad
    # The retired 2-second-window options are deliberately ignored so old
    # saved settings cannot reactivate that bad-channel rule.
    legacy_window_signal = .003 * np.sin(2 * np.pi * 20 * np.arange(1000) / 250)
    legacy_window_good, _, _ = run_worker(BadChannelWorker(
        ArraySource(legacy_window_signal[:, None], 250, channel_ids=[9]),
        {"flat_std": "1e-4", "flat_ratio": "30", "parallel": False, "workers": "1",
         "two_sec": True, "valid_window": "1", "ptp": ".01"},
    ))
    assert legacy_window_good == [9]
    high_frequency_good, high_frequency_bad, high_frequency_candidates = run_worker(BadChannelWorker(
        ArraySource(np.full((1000, 1), 2500.0, dtype=np.float32), 250, channel_ids=[10]),
        {"flat_std": "1e-4", "flat_ratio": "30", "parallel": False, "workers": "1",
         "high_frequency_noise_only": True, "high_frequency_noise_check": True,
         "high_frequency_noise_target": "2500", "high_frequency_noise_tolerance": "1",
         "high_frequency_noise_ratio_threshold": "15"},
    ))
    assert not high_frequency_good
    assert "2.5mV" in high_frequency_bad[10]
    assert not high_frequency_candidates
    assert QtAnalysisGUI._brief_channel_review_description(
        "高频噪声/伪迹：数据集中在2.5mV附近 (center=2500, ratio=26.6%)"
    ) == "数据集中在2.5mV"
    assert QtAnalysisGUI._brief_channel_review_description(
        "too few finite samples"
    ) == "有效采样不足"
    assert QtAnalysisGUI._brief_channel_review_description(
        "saturation: bottom=24.1%"
    ) == "贴底饱和"
    car_source = ArraySource(
        np.column_stack([source.data, np.linspace(-.5, .5, source.data.shape[0])]),
        250, channel_ids=[7, 500, 999],
    )
    car_output, car_info = run_worker(LeaveOneOutMedianCARWorker(car_source, {7, 500}))
    assert car_info["good_channel_ids"] == [7, 500]
    assert np.array_equal(
        car_output.data[:, 2], np.asarray(car_source.data[:, 2], dtype=np.float32)
    )
    scoped_good, scoped_bad, scoped_candidates = run_worker(BadChannelWorker(
        ArraySource(np.column_stack([np.zeros(1000), source.data[:, 1]]), 250, channel_ids=[7, 500]),
        {"flat_std": "1e-4", "flat_ratio": "30", "parallel": True, "workers": "2"},
        columns=[1],
    ))
    assert scoped_good == [500] and not scoped_bad and isinstance(scoped_candidates, dict)

    saturation_size = 1000
    saturation_normal = rng.normal(0, .1, saturation_size)
    bottom_saturated = saturation_normal.copy()
    bottom_saturated[:600] = -1.0
    fast_good, fast_bad, _ = run_worker(BadChannelWorker(
        ArraySource(np.column_stack([
            saturation_normal, .9 * saturation_normal, 1.1 * saturation_normal,
            .8 * saturation_normal, bottom_saturated,
        ]), 250, channel_ids=[110, 112, 113, 114, 111]),
        {"flat_std": "1e-4", "flat_ratio": "30", "parallel": False, "workers": "1", "fast_artifact_only": True,
         "fast_artifact_check": True},
    ))
    assert 111 in fast_bad and 110 in fast_good
    fast_single_normal = rng.normal(0, .1, saturation_size)
    top_saturated = fast_single_normal.copy()
    top_saturated[:600] = 1.0
    fast_worker = BadChannelWorker(
        ArraySource(np.column_stack([
            fast_single_normal, .9 * fast_single_normal, 1.1 * fast_single_normal,
            .8 * fast_single_normal, top_saturated,
        ]), 250, channel_ids=[110, 112, 113, 114, 111]),
        {"flat_std": "1e-4", "flat_ratio": "30", "parallel": False, "workers": "1", "fast_artifact_only": True,
         "fast_artifact_check": True, "high_frequency_noise_check": True}, columns=[4],
    )
    fast_single_good, fast_single_bad, fast_single_candidates = run_worker(fast_worker)
    assert (
        111 in fast_single_good and 111 not in fast_single_bad
        and 111 not in fast_single_candidates
        and "top_ratio" not in fast_worker.fast_artifact_rows[0]
        and "top_bad" not in fast_worker.fast_artifact_rows[0]
    )
    assert not fast_worker.high_frequency_noise_rows
    local_saturation = rng.normal(0, .1, saturation_size).astype(np.float32)
    local_saturation[:100] = -1.0
    local_worker = BadChannelWorker(
        ArraySource(np.column_stack([rng.normal(0, .1, saturation_size), local_saturation]), 250, channel_ids=[112, 113]),
        {"flat_std": "1e-4", "flat_ratio": "30", "parallel": False, "workers": "1", "fast_artifact_only": True,
         "fast_artifact_check": True, "saturation_ratio_threshold": "30"},
    )
    local_good, local_bad, _ = run_worker(local_worker)
    local_metrics = {row["channel"]: row for row in local_worker.fast_artifact_rows}[113]
    assert 113 in local_good and 113 not in local_bad and not local_metrics["bottom_bad"]

    task, = run_worker(TaskEpochWorker(source, np.asarray([[300, 2], [600, 2]]), {"epoch_start": -100, "epoch_end": 200, "baseline_start": -100, "baseline_end": 0, "response_start": 0, "response_end": 150, "trials_per_stim": 30, "tags": [2], "aggregate": "mean", "columns": [1]}))
    assert task["channel_ids"] == [500] and len(task["results"][0]["metrics"]) == 1
    assert np.isfinite(task["results"][0]["metrics"][0]["event_lfp_snr_db"])
    vep, = run_worker(TrialVepWorker(source, task["results"][0], 500, task))
    assert vep["qc_applied"] is False and vep["trials"].shape[0] == 1

    # Five 1-second windows are checked in one vectorized operation.  A flat
    # channel, a clipped plateau and a large jump must not pass the 3/5 rule.
    epoch = rng.normal(0, .05, size=(int(6.5 * 250), 4)).astype(np.float32)
    epoch[:, 1] = 0.0
    for window in range(5):
        first = int((1.0 + window) * 250)  # 0.5 s after marker; epoch starts at -0.5 s.
        epoch[first:first + 12, 2] = .75
        epoch[first + 100, 3] = 30.0
    clean, quality = evaluate_task_trial_window_quality(epoch, 250, -500)
    assert clean.shape == (5, 4) and np.all(clean[:, 0])
    assert not np.any(clean[:, 1]) and np.all(quality["flat"][:, 1])
    assert not np.any(clean[:, 2]) and np.all(quality["saturated"][:, 2])
    assert not np.any(clean[:, 3]) and np.all(quality["jump"][:, 3])

    # A three-second response epoch contains W1/W2 but not W3-W5.  Missing
    # windows must be unavailable rather than silently counted as clean, and
    # the two consecutive clean seconds still support the short PSD path.
    short_values = (.05 * np.sin(2 * np.pi * 10 * np.arange(875) / 250)).astype(np.float32)[:, None]
    short_clean, short_quality = evaluate_task_trial_window_quality(short_values, 250, -500)
    assert np.all(short_clean[:2, 0]) and not np.any(short_clean[2:, 0])
    assert np.all(short_quality["available"][:2, 0]) and np.all(short_quality["unavailable"][2:, 0])
    short_source = ArraySource(np.vstack([short_values, short_values[:125]]), 250, channel_ids=[10])
    short_task, = run_worker(TaskEpochWorker(
        short_source, np.asarray([[125, 2]]),
        {"epoch_start": -500, "epoch_end": 3000, "baseline_start": -200,
         "baseline_end": 0, "response_start": 0, "response_end": 300,
         "trials_per_stim": 0, "tags": [2], "aggregate": "mean",
         "columns": [0], "task_target_freq_hz": 10.0, "task_neighbor_bins": 4},
    ))
    short_metric = short_task["results"][0]["metrics"][0]
    assert short_metric["usable_trials"] == 1
    assert short_metric["available_windows"] == 2 and short_metric["clean_windows"] == 2
    assert short_metric["short_epoch_trials"] == 1 and short_metric["spectral_seconds"] == 2.0

    # ITPC reuses the cached per-trial five-window mask and preserves marker
    # phase.  Six 10 Hz trials with identical marker phase must lock near 1.
    itpc_fs, itpc_markers = 100, np.asarray([[1000 + 1000 * index, 2] for index in range(6)])
    itpc_source = ArraySource(np.sin(2 * np.pi * 10 * np.arange(8000) / itpc_fs).astype(np.float32)[:, None], itpc_fs, channel_ids=[10])
    itpc_quality = {
        "columns": np.asarray([0]), "channel_ids": [10], "epoch_start_ms": -500.0,
        "epoch_end_ms": 6000.0, "trial_quality": {
            (2, int(sample)): {"clean_windows": np.ones((5, 1), dtype=bool)}
            for sample, _tag in itpc_markers
        },
    }
    itpc, = run_worker(ItpcWorker(itpc_source, itpc_quality, {
        "tags": [2], "freq_low": 10.0, "freq_high": 10.0, "freq_step": 1.0,
        "cycles": 1.0, "time_start": 500.0, "time_end": 5500.0,
        "baseline_start": 500.0, "baseline_end": 1500.0,
        "sample_rate": 100.0, "min_trials": 4,
    }))
    itpc_result = itpc["results"][0]
    assert np.nanmax(itpc_result["itpc"]) > .99 and np.nanmax(itpc_result["counts"]) == 6

    # The spectral calculation averages independent continuous 1 s windows;
    # it must recover a 1 Hz target without stitching the five windows.
    t = np.arange(250, dtype=float) / 250
    sine_windows = np.stack([np.sin(2 * np.pi * t) for _ in range(5)], axis=0)[:, :, None]
    target_power, neighbor_power = compute_task_window_spectral_metrics(sine_windows, 250, 1.0, 4)
    assert np.all(target_power > neighbor_power * 10)
    # W1/W2 and W4/W5 are separate 2-second PSD segments.  They are not
    # joined through the failed W3 interval, yet their target PSD contributes.
    clean_runs = np.asarray([[True], [True], [False], [True], [True]])
    epoch_for_psd = np.vstack([np.zeros((125, 1)), sine_windows.reshape(-1, 1), np.zeros((125, 1))])
    run_target, run_neighbor, run_samples, run_segments = compute_task_continuous_psd_metrics(
        epoch_for_psd, clean_runs, np.asarray([True]), 250, -500, 1.0, 4,
    )
    assert run_samples[0] == 4 * 250 and run_segments[0] == 2
    assert run_target[0] > run_neighbor[0] * 5

    # External pre-task baseline references use independently QC'd natural
    # 3/4/5 s slices.  A task run is paired only with its exact length.
    baseline_fs = 100.0
    baseline_time = np.arange(int(60 * baseline_fs), dtype=float) / baseline_fs
    baseline_source = ArraySource(
        np.column_stack([
            .08 * np.sin(2 * np.pi * 5 * baseline_time),
            .05 * np.sin(2 * np.pi * 5 * baseline_time + .2),
        ]).astype(np.float32),
        baseline_fs, channel_ids=[10, 11],
    )
    external_reference = compute_external_baseline_reference(
        baseline_source, np.asarray([0, 1]), [10, 11],
        {
            "task_target_freq_hz": 5.0, "task_neighbor_bins": 2,
            "baseline_candidate_count": 8, "baseline_min_valid_segments": 4,
            "baseline_random_seed": 8, "analysis_notch": False,
            "analysis_bandpass": False, "smooth": False,
            "quality_flat_epsilon": 1e-4, "quality_flat_ratio_percent": 30.0,
            "quality_flat_ptp": .01, "quality_available_ratio": .995,
            "quality_saturation_run_samples": 8, "quality_jump_mad_multiplier": 12.0,
            "quality_jump_median_multiplier": 8.0, "quality_jump_flat_floor_multiplier": 10.0,
        },
    )
    assert set(external_reference["references"]) == {3, 4, 5}
    assert np.all(external_reference["references"][5]["eligible"])
    three_window_epoch = np.full((int(5.5 * baseline_fs), 1), np.nan, dtype=np.float32)
    three_window_epoch[:int(1.0 * baseline_fs)] = 0.0
    three_window_epoch[int(1.0 * baseline_fs):int(4.0 * baseline_fs)] = (
        .08 * np.sin(2 * np.pi * 5 * np.arange(int(3 * baseline_fs)) / baseline_fs)
    )[:, None]
    three_clean, _ = evaluate_task_trial_window_quality(three_window_epoch, baseline_fs, -500)
    by_length = compute_task_psd_metrics_by_run_length(
        three_window_epoch, three_clean, np.asarray([True]), baseline_fs, -500, 5.0, 2,
    )
    assert by_length[3][2][0] == int(3 * baseline_fs)
    assert by_length[4][2][0] == 0 and by_length[5][2][0] == 0
    external_task_source = ArraySource(
        (.08 * np.sin(2 * np.pi * 5 * np.arange(800) / baseline_fs)).astype(np.float32)[:, None],
        baseline_fs, channel_ids=[10],
    )
    external_task, = run_worker(TaskEpochWorker(
        external_task_source, np.asarray([[100, 2]]),
        {"epoch_start": -500, "epoch_end": 6000, "baseline_start": -200,
         "baseline_end": 0, "response_start": 0, "response_end": 300,
         "trials_per_stim": 0, "tags": [2], "aggregate": "mean",
         "columns": [0], "task_target_freq_hz": 5.0, "task_neighbor_bins": 2,
         "external_baseline_reference": external_reference},
    ))
    external_metric = external_task["results"][0]["metrics"][0]
    assert external_metric["external_baseline_lengths"] == "5s"
    assert external_metric["external_baseline_matched_seconds"] == 5.0
    assert np.isfinite(external_metric["external_baseline_target_power_change_db"])

    # The independent task/rest test is two-sided and returns the actual time
    # indices supplied by MNE, rather than interpreting its slice output as a
    # boolean mask.
    cluster_task = rng.normal(0, .1, size=(14, 80))
    cluster_rest = rng.normal(0, .1, size=(14, 80))
    cluster_task[:, 30:48] += 1.0
    cluster_output = compute_external_rest_cluster_test(
        cluster_task, cluster_rest, permutations=1000, random_seed=7,
    )
    assert cluster_output["status"] == "ok"
    assert any(item["significant"] and item["first"] == 30 and item["last"] == 47 for item in cluster_output["clusters"])

    tf_task = rng.normal(0, .1, size=(12, 6, 50))
    tf_rest = rng.normal(0, .1, size=(12, 6, 50))
    tf_task[:, 2:4, 20:35] += 1.0
    tf_output = compute_external_rest_tf_cluster_test(tf_task, tf_rest, permutations=1000, random_seed=4)
    assert tf_output["status"] == "ok"
    assert np.all(tf_output["significant_mask"][2:4, 20:35])

    time_external_reference = compute_external_baseline_reference(
        baseline_source, np.asarray([0]), [10],
        {
            "task_target_freq_hz": 5.0, "task_neighbor_bins": 2,
            "baseline_candidate_count": 4, "baseline_min_valid_segments": 3,
            "baseline_random_seed": 9, "analysis_notch": False,
            "analysis_bandpass": False, "smooth": False,
            "quality_flat_epsilon": 1e-4, "quality_flat_ratio_percent": 30.0,
            "quality_flat_ptp": .01, "quality_available_ratio": .995,
            "quality_saturation_run_samples": 8, "quality_jump_mad_multiplier": 12.0,
            "quality_jump_median_multiplier": 8.0, "quality_jump_flat_floor_multiplier": 10.0,
            "epoch_start": -500.0, "epoch_end": 6000.0,
            "baseline_start": -200.0, "baseline_end": 0.0,
            "response_start": 0.0, "response_end": 300.0,
        },
    )
    time_task_source = ArraySource(
        (.08 * np.sin(2 * np.pi * 5 * np.arange(5000) / baseline_fs)).astype(np.float32)[:, None],
        baseline_fs, channel_ids=[10],
    )
    time_task, = run_worker(TaskEpochWorker(
        time_task_source, np.asarray([[300, 2], [600, 2], [900, 2]]),
        {"epoch_start": -500, "epoch_end": 6000, "baseline_start": -200,
         "baseline_end": 0, "response_start": 0, "response_end": 300,
         "trials_per_stim": 0, "tags": [2], "aggregate": "mean",
         "columns": [0], "task_target_freq_hz": 5.0, "task_neighbor_bins": 2,
         "external_baseline_reference": time_external_reference},
    ))
    time_response = time_task["results"][0]["external_time_response"]
    assert time_response is not None and time_response["time_ms"].size > 0
    assert time_response["channels"][10]["task_trials"] == 3
    tf_worker, = run_worker(ExternalTimeFrequencyWorker(
        time_task_source, baseline_source, time_external_reference, time_task["results"][0], 10,
        {
            "epoch_start": -500.0, "epoch_end": 6000.0,
            "baseline_start": -200.0, "baseline_end": 0.0,
            "response_start": 0.0, "response_end": 300.0,
            "analysis_notch": False, "analysis_bandpass": False,
            "analysis_band_low": .5, "analysis_band_high": 300.0,
            "smooth": False, "smooth_window_sec": .02, "zscore": False,
            "quality_settings": {
                "quality_flat_epsilon": 1e-4, "quality_flat_ratio_percent": 30.0,
                "quality_flat_ptp": .01, "quality_available_ratio": .995,
                "quality_saturation_run_samples": 8, "quality_jump_mad_multiplier": 12.0,
                "quality_jump_median_multiplier": 8.0, "quality_jump_flat_floor_multiplier": 10.0,
            },
            "freq_low": 15.0, "freq_high": 25.0, "freq_step": 5.0,
            "cycles": 2.0, "sample_rate_hz": 100.0,
        },
    ))
    assert tf_worker["task_mean_db"].ndim == 2 and tf_worker["comparison"]["status"] == "ok"

    quality_source = ArraySource(
        np.vstack([epoch, rng.normal(0, .05, size=250 * 2 * 4).reshape(250 * 2, 4)]).astype(np.float32),
        250, channel_ids=[10, 11, 12, 13],
    )
    quality_task, = run_worker(TaskEpochWorker(
        quality_source, np.asarray([[125, 2]]),
        {"epoch_start": -500, "epoch_end": 6000, "baseline_start": -200,
         "baseline_end": 0, "response_start": 0, "response_end": 300,
         "trials_per_stim": 0, "tags": [2], "aggregate": "mean",
         "columns": [0, 1, 2, 3], "task_target_freq_hz": 1.0,
         "task_neighbor_bins": 4},
    ))
    quality_metrics = quality_task["results"][0]["metrics"]
    assert quality_metrics[0]["usable_trials"] == 1
    assert quality_metrics[0]["target_freq_hz"] == 1.0
    assert quality_metrics[0]["spectral_seconds"] == 5.0
    assert quality_metrics[0]["spectral_segments"] == 1
    assert quality_metrics[1]["usable_trials"] == 0
    assert quality_metrics[2]["saturated_windows"] == 5
    assert quality_metrics[3]["jump_windows"] == 5

    # Page 4 first runs QC only, then reuses its exact per-trial window
    # decisions for frequency metrics rather than evaluating the windows again.
    qc_only, = run_worker(TaskEpochWorker(
        quality_source, np.asarray([[125, 2]]),
        {"epoch_start": -500, "epoch_end": 6000, "baseline_start": -200,
         "baseline_end": 0, "response_start": 0, "response_end": 300,
         "trials_per_stim": 0, "tags": [2], "aggregate": "mean",
         "columns": [0, 1, 2, 3], "task_target_freq_hz": 1.0,
         "task_neighbor_bins": 4, "quality_only": True, "quality_signature": ("qc",)},
    ))
    assert qc_only["quality_only"] and (2, 125) in qc_only["trial_quality"]
    assert qc_only["results"][0]["metrics"][0]["spectral_seconds"] == 0.0
    reused_qc, = run_worker(TaskEpochWorker(
        quality_source, np.asarray([[125, 2]]),
        {"epoch_start": -500, "epoch_end": 6000, "baseline_start": -200,
         "baseline_end": 0, "response_start": 0, "response_end": 300,
         "trials_per_stim": 0, "tags": [2], "aggregate": "mean",
         "columns": [0, 1, 2, 3], "task_target_freq_hz": 1.0,
         "task_neighbor_bins": 4, "precomputed_trial_quality": qc_only["trial_quality"]},
    ))
    assert reused_qc["results"][0]["metrics"][0]["spectral_seconds"] == 5.0

    # Page-3 Letter browser keeps its legacy epoch/video plotting path.  It
    # must not apply the page-4 LFP five-window or PSD metrics.
    letter_browser_task, = run_worker(TaskEpochWorker(
        quality_source, np.asarray([[125, 4]]),
        {"mode": "Letter", "epoch_start": -500, "epoch_end": 6000,
         "baseline_start": -200, "baseline_end": 0, "response_start": 0,
         "response_end": 300, "trials_per_stim": 0, "tags": [4],
         "aggregate": "mean", "columns": [0], "lfp_task_metrics": False},
    ))
    assert letter_browser_task["results"][0]["metrics"][0]["usable_trials"] == 1
    assert np.isnan(letter_browser_task["results"][0]["metrics"][0]["target_power"])

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "processed.h5"
        gui = QtAnalysisGUI()
        # The plotting facade must cover every result-view primitive with
        # native PyQtGraph items (including the former Matplotlib-only calls).
        primitive_plot = qt_gui_module.PyQtGraphPlot()
        primitive_axis = primitive_plot.add_subplot(111)
        primitive_axis.text(.5, .5, "no data", coordinates="axes", ha="center", va="center")
        primitive_axis.fill_between([0, 1], [0, 0], [1, 1], color="#1f77b4", alpha=.2)
        primitive_axis.axvspan(.2, .4, color="#2e7d32", alpha=.16)
        primitive_axis.set_axis_off()
        primitive_plot.clear()
        assert primitive_plot.plotItem.getAxis("bottom").isVisible()

        no_reference_output = {
            "channel_ids": [3],
            "results": [{
                "tag": 4, "trials": 1,
                "metrics": [{"channel": 3, "trials": 1, "usable_trials": 1}],
                "external_time_response": {"channels": {}, "time_ms": []},
            }],
        }
        gui._populate_lfp_task_result_controls(no_reference_output)
        missing_index = gui.lfp_task_result_metric_box.findData("external_time_response")
        gui.lfp_task_result_metric_box.setCurrentIndex(missing_index)
        gui._render_lfp_task_result_view()
        response = {
            "task_mean": np.asarray([0.0, 1.0, 0.0]), "task_sem": np.asarray([.1, .2, .1]),
            "rest_mean": np.asarray([0.0, .2, 0.0]), "rest_sem": np.asarray([.05, .1, .05]),
            "task_trials": 4, "rest_epochs": 4, "status": "ok",
            "clusters": [{"first": 1, "last": 2, "significant": True}],
        }
        no_reference_output["results"][0]["external_time_response"] = {
            "channels": {3: response}, "time_ms": np.asarray([0.0, 10.0, 20.0]),
        }
        gui._render_lfp_task_result_view()
        assert gui.lfp_analysis_mode_tabs.count() == 2
        assert "静息态" in gui.lfp_analysis_mode_tabs.tabText(0)
        assert "任务分析" in gui.lfp_analysis_mode_tabs.tabText(1)
        assert 165 <= gui.lfp_analysis_mode_tabs.height() <= 500
        assert gui.lfp_result_splitter.orientation() == Qt.Orientation.Horizontal
        preprocess_form = gui.preprocess_run_button.parentWidget().layout()
        one_click_row, _ = preprocess_form.getWidgetPosition(gui.one_click_preprocess_button)
        filter_row, _ = preprocess_form.getWidgetPosition(gui.preprocess_run_button)
        progress_row, _ = preprocess_form.getWidgetPosition(gui.preprocess_progress)
        assert one_click_row < filter_row < progress_row
        assert gui.one_click_preprocess_button.objectName() == "oneClickPreprocess"
        assert gui.preprocess_run_button.objectName() == "filterAction"
        assert gui.preprocess_progress.objectName() == "preprocessProgress"
        assert gui.one_click_preprocess_button.graphicsEffect() is not None
        assert gui.preprocess_run_button.graphicsEffect() is not None
        assert gui.preprocess_progress.graphicsEffect() is not None

        # Letter single-channel epochs are loaded once and then served from
        # the bounded GUI cache when the same channel is reopened.
        class CountingSource:
            def __init__(self):
                self.data = np.arange(4000, dtype=np.float32)[:, None]
                self.metadata = type("Metadata", (), {
                    "rows": 4000, "fs": 250.0, "time_offset": 0.0,
                    "channel_ids": (3,),
                })()
                self.read_count = 0
                self.read_calls = []

            @property
            def loaded(self):
                return True

            def read(self, first, last, column, step=1):
                self.read_count += 1
                self.read_calls.append((first, last, column, step))
                return self.data[first:last:step, column]

        counting_source = CountingSource()
        letter_output = {"columns": np.asarray([0], dtype=int)}
        letter_result = {
            "tag": 4, "t_ms": np.arange(-125, 1500, dtype=float) / 250.0 * 1000.0,
            "used_marker_samples": [500, 2000],
        }
        first_t, first_trials, first_samples = gui._load_letter_channel_trials(
            counting_source, letter_output, letter_result, 0,
        )
        second_t, second_trials, second_samples = gui._load_letter_channel_trials(
            counting_source, letter_output, letter_result, 0,
        )
        assert counting_source.read_count == 2
        assert first_trials is second_trials
        assert np.array_equal(first_t, second_t)
        assert first_samples == second_samples == (500, 2000)

        previous_active_source = gui.active_source
        gui.active_source = counting_source
        gui.alignment_preview_start_spin.setValue(2.0)
        gui.alignment_preview_duration_spin.setValue(3.0)
        gui.refresh_alignment_channel_preview()
        assert counting_source.read_calls[-1] == (500, 1250, 0, 1)
        gui.alignment_preview_full_duration_check.setChecked(True)
        assert not gui.alignment_preview_start_spin.isEnabled()
        assert not gui.alignment_preview_duration_spin.isEnabled()
        gui.refresh_alignment_channel_preview()
        assert counting_source.read_calls[-1] == (0, 4000, 0, 1)
        gui.alignment_preview_full_duration_check.setChecked(False)
        gui.active_source = previous_active_source

        gui.stim_markers = np.asarray([[500, 4], [2000, 5]], dtype=int)
        gui.behavior_matches = {}
        gui.behavior_detection_frame_rate = 30.0
        gui.behavior_detection_video_eeg_offset_sec = 0.0
        browser_time = np.linspace(-500.0, 6000.0, 40)
        browser_output = {
            "results": [{
                "tag": 4, "t_ms": browser_time,
                "wave": np.zeros((browser_time.size, 25), dtype=np.float32),
                "used_marker_samples": [500],
            }],
            "channel_ids": list(range(1, 26)),
            "columns": np.arange(25, dtype=int),
        }
        gui._show_letter_behavior_browser("lfp", browser_output)
        browser = gui._letter_behavior_window
        original_plot_ids = tuple(id(plot) for plot in browser._letter_page_plots)
        browser._letter_page_state["page"] = 1
        browser._letter_render()
        assert tuple(id(plot) for plot in browser._letter_page_plots) == original_plot_ids
        assert browser._letter_page_plots[0]._letter_channel_index > 0
        browser._letter_page_state.update(tag=5, page=0)
        browser._letter_render()
        assert all(plot.isHidden() for plot in browser._letter_page_plots)
        browser.close()

        assert gui._preprocess_action_buttons[0] is gui.preprocess_params_button_var
        assert all(not button.isHidden() for button in gui._preprocess_action_buttons)
        assert gui.preprocess_advanced_parameters.isHidden()
        assert all(button.isHidden() for button in gui._preprocess_expert_action_buttons)
        assert all(widget.isHidden() for widget in gui._preprocess_expert_widgets)
        assert gui.bad_high_frequency_noise_check_var.text().startswith("2.5mV")
        assert gui.bad_high_frequency_noise_target_var.text() == "2500"
        assert gui.preprocessed_preview.full_duration_check.isChecked()
        assert not gui.preprocessed_preview.start_spin.isEnabled()
        assert not gui.preprocessed_preview.end_spin.isEnabled()
        assert not hasattr(gui, "bad_2s_window_var")
        assert not hasattr(gui, "bad_valid_window_var")
        assert not hasattr(gui, "bad_ptp_var")
        assert not hasattr(gui, "bad_flat_std_var")
        assert not hasattr(gui, "bad_flat_ratio_var")
        gui._toggle_preprocess_parameter_area()
        assert all(not button.isHidden() for button in gui._preprocess_action_buttons)
        assert not gui.preprocess_advanced_parameters.isHidden()
        assert all(not button.isHidden() for button in gui._preprocess_expert_action_buttons)
        assert all(not widget.isHidden() for widget in gui._preprocess_expert_widgets)
        gui._toggle_preprocess_parameter_area()
        assert all(not button.isHidden() for button in gui._preprocess_action_buttons)
        assert gui.preprocess_advanced_parameters.isHidden()
        gui._refresh_bad_channel_review_table()
        assert not gui.good_channel_ids and not gui.bad_channel_ids
        assert not gui.bad_channel_fast_artifact_rows
        assert not gui.bad_channel_high_frequency_noise_rows
        gui._bad_channel_result_channel_ids = (7, 500)
        gui.bad_channel_auto_reasons = {500: "saturation: bottom=2.0%"}
        gui.manual_channel_overrides = {7: "good"}
        gui._apply_channel_review_overrides()
        gui._refresh_bad_channel_review_table()
        assert gui.bad_channel_ids == {500} and gui.good_channel_ids == {7}
        assert gui.bad_channel_reason_table.rowCount() == 2
        gui._toggle_bad_channel_review_description_mode(2)
        assert "简要" in gui.bad_channel_reason_table.horizontalHeaderItem(2).text()
        gui._toggle_bad_channel_review_description_mode(2)

        gui._bad_channel_check_completed = True
        assert gui._ica_eligible_channel_ids(source) == {7}
        gui._selected_analysis_batch = None
        gui.source = source
        gui.active_source = source
        remapped_detection_source = ArraySource(
            source.data.copy(), 250, channel_ids=[7, 500], label="remapped-test"
        )
        filtered_display_source = ArraySource(
            (source.data * .5).copy(), 250, channel_ids=[7, 500], label="filtered-test"
        )
        gui.remapped_source = remapped_detection_source
        gui.preprocessed_source = filtered_display_source
        remap_marker = Path(folder) / "mapping-marker.xlsx"
        remap_marker.write_text("mapping-version", encoding="ascii")
        gui.remap_edit.setText(str(remap_marker))
        gui._remapped_detection_signature = gui._remap_signature(str(remap_marker))
        assert gui._detection_source() is remapped_detection_source
        gui.remapped_source = None
        assert gui._detection_source() is source
        gui.remapped_source = remapped_detection_source
        gui.alignment_selected_ids = {7, 500}
        gui._inherit_alignment_selection_for_lfp()
        assert gui.lfp_selected_ids == {7, 500}
        gui.lfp_selected_ids = {500}
        gui._lfp_selection_explicit = True
        gui.alignment_selected_ids = {7}
        gui._inherit_alignment_selection_for_lfp()
        assert gui.lfp_selected_ids == {500}
        assert gui.lfp_task_result_metric_box.findData("local_tag_snr_db") >= 0
        assert gui.lfp_task_result_metric_box.findData("event_lfp_snr_db") >= 0
        gui.open_import_overview()
        gui.open_preprocess_filter_overview()
        assert not gui._import_overview_window.selection_mode
        assert gui._preprocess_filter_selector.selection_mode
        assert gui._import_overview_window is not gui._preprocess_filter_selector
        assert set(gui._preprocess_filter_selector.selection_group_buttons) == {"健康道", "坏道"}
        gui._preprocess_filter_selector._change_selection_group({500}, add=True)
        assert gui._preprocess_filter_selector._selected_channel_ids == {500}
        gui._preprocess_filter_selector._change_selection_group({500}, add=False)
        assert not gui._preprocess_filter_selector._selected_channel_ids
        gui._set_preprocess_filter_channel_ids([500])
        assert gui.preprocess_filter_channels_var.text() == "500"
        assert getattr(gui, "_preprocess_worker", None) is None
        gui._render_external_baseline_qc_details(external_reference)
        assert gui.external_baseline_qc_table.rowCount() == 6
        assert gui.external_baseline_qc_table.item(0, 0).text() == "3 s"
        gui._write_source_h5(output, source)
        partial_output = Path(folder) / "partial_preprocessed.h5"
        partial_filtered_data = source.data.copy()
        partial_filtered_data[:, 0] *= .5
        partial_filtered_source = ArraySource(
            partial_filtered_data, 250, channel_ids=[7, 500], label="partial-filtered-test"
        )
        gui.active_source = partial_filtered_source
        gui.preprocessed_source = partial_filtered_source
        gui._preprocess_filtered_channel_ids = {7}
        gui._bad_channel_check_completed = True
        gui._bad_channel_result_channel_ids = (7, 500)
        gui.bad_channel_auto_reasons = {500: "saturation: bottom=2.0%"}
        gui.manual_channel_overrides = {7: "good"}
        gui._apply_channel_review_overrides()
        assert gui._ica_eligible_channel_ids(partial_filtered_source) == {7}
        record_output = gui._write_preprocessed_h5(partial_output, partial_filtered_source)
        gui._import_overview_window.close()
        gui._preprocess_filter_selector.close()
        gui.close()
        with h5py.File(output, "r") as h5:
            assert h5.attrs["hdf5_compression_filter"] == "blosc:lz4"
            assert h5.attrs["hdf5_compression_shuffle"] == "bitshuffle"
            provenance = read_h5_provenance(h5)
            assert provenance is not None
            assert provenance["data"]["unit"] == "mV"
            assert provenance["operations"][-1]["name"] == "export_hdf5"
        reloaded = LazyH5Source(); meta = reloaded.open(output)
        assert meta.channel_ids == (7, 500) and meta.time_offset == 1.25
        assert meta.provenance is not None and meta.provenance["version"] == 1
        many = reloaded.read_many([(0, 25), (100, 140)], 0)
        assert len(many) == 2
        assert np.array_equal(many[0], reloaded.read(0, 25, 0))
        assert np.array_equal(many[1], reloaded.read(100, 140, 0))
        with h5py.File(partial_output, "r") as h5:
            assert h5["rawData512"].shape == source.data.shape
            assert np.allclose(h5["rawData512"][:, 0], partial_filtered_source.data[:, 0])
            assert np.allclose(h5["rawData512"][:, 1], source.data[:, 1])
            assert h5.attrs["processing_record_csv"] == record_output.name
            assert h5.attrs["filtered_channel_ids"].tolist() == [7]
            qc_snapshot = json.loads(h5.attrs["preprocess_qc_json"])
            assert qc_snapshot["completed"] is True
            assert qc_snapshot["manual_overrides"] == {"7": "good"}
            provenance = read_h5_provenance(h5)
            assert provenance["operations"][-1]["processing_record_csv"] == record_output.name
        with open(record_output, newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
        assert len(records) == 2
        assert records[0]["output_file"] == partial_output.name
        assert records[0]["filtered_channel_ids"] == "7"
        records_by_channel = {int(record["channel"]): record for record in records}
        assert records_by_channel[7]["final_status"] == "healthy"
        assert records_by_channel[500]["automatic_reason"] == "saturation: bottom=2.0%"
        assert records_by_channel[500]["brief_description"] == "贴底饱和"

        reloaded_gui = QtAnalysisGUI()
        reloaded_gui.path_edit.setText(str(partial_output))
        reloaded_gui.load_file()
        assert reloaded_gui.preprocessed_source is reloaded_gui.source
        assert reloaded_gui._bad_channel_check_completed
        assert reloaded_gui.good_channel_ids == {7}
        assert reloaded_gui.bad_channel_ids == {500}
        assert reloaded_gui.bad_channel_reason_table.rowCount() == 2
        assert not reloaded_gui.apply_car_button.isEnabled()
        reloaded_gui.close()

    # ICA needs enough data and at least two channels; constrain iterations to
    # keep this regression check fast while exercising the actual MNE path.
    clean, info = run_worker(IcaWorker(source, [0, 1], {"components": 2, "exclude": [0], "low": 1., "high": 80., "decim": 2, "max_iter": 20}))
    assert clean.metadata.channel_ids == (7, 500) and info["components"] == 2
    app.quit()
    print("Qt core workflow regression: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
