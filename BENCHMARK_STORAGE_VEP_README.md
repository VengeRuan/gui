# Storage and VEP comparison

`benchmark_storage_vep.py` starts from the original BIN, writes eight HDF5
layouts (four compression filters in all-channel and per-channel form), reads
each layout back, and optionally runs a LetterModeData-style epoch comparison.

## Files needed on another computer

Required Python files:

- `benchmark_storage_vep.py`
- `compare_bin_storage.py`

Required input data:

- the original `.bin` file
- the GUI experiment `log.txt`
- the Letter Event CSV used by the GUI

Optional:

- a 512-value channel-quality text file (`0` or `1` per channel)
- MATLAB on `PATH`, only when MATLAB smoke tests are wanted

The benchmark does not import or execute `integrated_pipeline_gui.py`, but the
epoch and marker code mirrors its `build_stim_markers_from_csv`, `make_epochs`,
and `baseline_correct_epochs` logic.

## Install Python packages

```powershell
python -m pip install numpy h5py hdf5plugin pandas matplotlib psutil openpyxl
```

`hdf5plugin` is required for the Blosc/LZ4 + bitshuffle HDF5 comparison.
`matplotlib` is only needed for PNG plots. `psutil` is only needed for RSS
measurements. `openpyxl` is only needed for `storage_summary.xlsx`. Python 3.11 or 3.12 may be needed if a wheel is not yet
available for Python 3.14.

The current benchmark entry point compares native HDF5 only. The older Parquet
helper functions remain in the script for reference, but Parquet is not part of
the eight-layout run.

## Run storage plus VEP comparison

```powershell
python benchmark_storage_vep.py `
  --bin-file "D:\data\20260626_1102_block_1.bin" `
  --log-txt "D:\data\log.txt" `
  --event-csv "D:\data\LetterEvent.csv" `
  --output-dir "D:\data\storage_layout_test" `
  --run-vep `
  --vep-tags "4,5,6,7" `
  --plot-channel 1 `
  --epoch-sample-channels "1,2,17" `
  --epoch-sample-trials 3
```

For a short test, add `--duration-sec 10`. Remove it for the full BIN.
If the source file does not contain `deltaT1`, add for example:
`--deltaT1-ms 6`.

The default epoch parameters are the GUI LetterModeData defaults:

- epoch: `-500` to `800 ms`
- baseline: `-200` to `0 ms`
- response: `0` to `300 ms`
- aggregate: mean

Use `--aggregate median` when the GUI is configured to use median aggregation.

## HDF5 comparison output

The benchmark writes these eight groups:

- `all_channels_gzip1.h5` and `per_channel_gzip1\`
- `all_channels_gzip4.h5` and `per_channel_gzip4\`
- `all_channels_lzf_shuffle.h5` and `per_channel_lzf_shuffle\`
- `all_channels_blosc_lz4_bitshuffle.h5` and `per_channel_blosc_lz4_bitshuffle\`

## VEP output

The output directory contains:

- `vep_metrics.csv`: per-layout, per-tag summary metrics.
- `vep_epoch_compare.csv`: raw and baseline-corrected epoch comparison against
  `all_gzip4` (the benchmark reference). Lossless layouts should have RMSE close to zero and correlation
  close to one.
- `vep_epoch_samples\`: compressed NumPy samples for each layout and tag.
  Each `.npz` contains `raw_epochs`, `corrected_epochs`, `used_markers`,
  `t_ms`, and `channels`.
- `vep_plots\`: average-wave plots and individual epoch sample plots. The
  `*_raw.png` files show the raw epoch; `*_baseline.png` shows the same epoch
  after the GUI baseline correction.
- `storage_vep_benchmark.json`: complete machine-readable report, including
  marker timing, read time, epoch metadata, and all comparison metrics.

Read one saved epoch sample with Python:

```powershell
python -c "import numpy as np; p=r'D:\data\storage_layout_test\vep_epoch_samples\all_gzip4_tag_6.npz'; x=np.load(p); print(x.files); print(x['raw_epochs'].shape, x['corrected_epochs'].shape, x['channels'], x['used_markers'])"
```

The epoch axis order is `[trial, sample, selected_channel]`.

## Important alignment rule

When `--log-txt` and `--event-csv` are supplied, marker samples are built as:

```text
marker_sample = round(fs * (PROGRAM_START - Recording_ON + deltaT1 + Letter_Display_Time_offset))
```

The first Letter trial is the anchor, matching the current GUI behavior. The
video `detection_result.txt` is not needed for this storage/epoch comparison;
it is only needed for the GUI's separate StartFrame/AnimalStart alignment view.
