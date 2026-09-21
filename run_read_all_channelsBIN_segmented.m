function out_mat = run_read_all_channelsBIN_segmented(bin_filename, output_dir, animal, blocknum, segment_start_sec, segment_duration_sec, max_read_mb, source_bin_text, record_date, output_mat_path)
% Parameterized BIN parser for the Python integration GUI.
% It parses the SD-card BIN format into 4 chips x 128 channels, converts ADC
% codes to voltage units, optionally crops a time segment, and saves a MAT file
% containing rawData512 (samples x channels), signal, FS, and time.

if nargin < 2 || isempty(output_dir)
    output_dir = fileparts(bin_filename);
end
if nargin < 3 || isempty(animal)
    animal = 0;
end
if nargin < 4 || isempty(blocknum)
    blocknum = 0;
end
if nargin < 5 || isempty(segment_start_sec)
    segment_start_sec = 0;
end
if nargin < 6 || isempty(segment_duration_sec)
    segment_duration_sec = 0;
end
if nargin < 7 || isempty(max_read_mb) || max_read_mb <= 0
    max_read_mb = inf;
end
if nargin < 8
    source_bin_text = '';
end
if nargin < 9
    record_date = '';
end
if nargin < 10
    output_mat_path = '';
end

if ~exist(bin_filename, 'file')
    error('BIN file does not exist: %s', bin_filename);
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

FS = 6490;
OFFSET = 0;
CH_PER_FRAME = 128;
NUM_CHIPS = 4;
BLOCK_SIZE_BYTES = 64 * 1024;
BLOCK_SIZE_WORDS = BLOCK_SIZE_BYTES / 2;
words_per_frame = 2 + CH_PER_FRAME;
VERIFY_INTERVAL = 100;

MAX_READ_BYTES = max_read_mb * 1024 * 1024;
if isinf(max_read_mb)
    MAX_READ_BYTES = inf;
end

fid = fopen(bin_filename, 'rb');
if fid == -1
    error('Cannot open BIN file: %s', bin_filename);
end
cleanup = onCleanup(@() fclose(fid));

file_info = dir(bin_filename);
file_size = file_info.bytes;

fprintf('BIN file: %s\n', bin_filename);
fprintf('File size: %.2f MB\n', file_size / 1024 / 1024);
fprintf('Searching sync from offset 0x%X...\n', OFFSET);
parser_total_timer = tic;
parser_stage_timer = tic;

dt1 = [];
dt2 = [];
dt1_date = '';
deltaT1 = 0;
deltaT2 = 0;
deltaT1_unit = 'ms';
try
    [dt1, dt2, deltaT1, deltaT2] = func_read_time_info(bin_filename);
    dt1.Format = 'yyyy-MM-dd';
    dt1_date = char(dt1);
    fprintf('Timing metadata: deltaT1 = %.6f ms, deltaT2 = %.6f ms\n', deltaT1, deltaT2);
catch ME
    warning('Could not read BIN timing metadata: %s', ME.message);
end
timing_metadata_sec = toc(parser_stage_timer);
fprintf('TIMING metadata %.3f sec\n', timing_metadata_sec);

%% 1. 全局同步头搜索 (向量化优化)
parser_stage_timer = tic;
fseek(fid, OFFSET, 'bof');
chunk_size_words = 5 * 1024 * 1024; % 10MB chunk
search_chunk = fread(fid, chunk_size_words, 'uint16=>uint16', 0, 'l');

sync_pattern = repmat(uint16(hex2dec('FFFF')), 1, 9);
sync_indices = strfind(search_chunk', sync_pattern);

if ~isempty(sync_indices)
    % 第一个字任意，后9个字FFFF，所以整体起点在 FFFF 前 1 个字(2字节)
    sync_start_pos = max(OFFSET, OFFSET + (sync_indices(1) - 1) * 2 - 2); 
    fprintf('Sync start position: 0x%X\n', sync_start_pos);
else
    error('Sync header was not found in the first 10MB.');
end

%% 2. 读取数据并分配芯片 (Reshape 优化)
sync_search_sec = toc(parser_stage_timer);
fprintf('TIMING sync_search %.3f sec\n', sync_search_sec);
parser_stage_timer = tic;
local_segment_read = segment_duration_sec > 0;
local_first_sample_abs = 1;
target_start_sample = max(1, floor(segment_start_sec * FS) + 1);
target_duration_samples = max(1, floor(segment_duration_sec * FS));
target_end_sample = target_start_sample + target_duration_samples - 1;
first_frame_starts = nan(1, NUM_CHIPS);
local_cycle_start = 0;

if local_segment_read
    probe_cycles = 64;
    probe_words = probe_cycles * BLOCK_SIZE_WORDS * NUM_CHIPS;
    probe_words = min(probe_words, floor((file_size - sync_start_pos) / 2));
    probe_words = floor(probe_words / (BLOCK_SIZE_WORDS * NUM_CHIPS)) * (BLOCK_SIZE_WORDS * NUM_CHIPS);
    if probe_words <= 0
        error('No complete 64KB block can be probed.');
    end
    fseek(fid, sync_start_pos, 'bof');
    probe_data = fread(fid, probe_words, 'uint16=>uint16', 0, 'l');
    probe_cycles_actual = floor(length(probe_data) / (BLOCK_SIZE_WORDS * NUM_CHIPS));
    probe_valid_words = probe_cycles_actual * BLOCK_SIZE_WORDS * NUM_CHIPS;
    probe_reshaped = reshape(probe_data(1:probe_valid_words), BLOCK_SIZE_WORDS, NUM_CHIPS, probe_cycles_actual);
    frame_pattern = [uint16(hex2dec('FFFF')), uint16(hex2dec('0000'))];
    for chip = 1:NUM_CHIPS
        probe_raw = reshape(probe_reshaped(:, chip, :), [], 1);
        sync_idx = strfind(probe_raw', sync_pattern);
        start_search = 1;
        if ~isempty(sync_idx)
            start_search = sync_idx(1) + 10;
        end
        frame_idx = strfind(probe_raw(start_search:end)', frame_pattern);
        if isempty(frame_idx)
            error('Chip %d first frame header was not found in the probe window.', chip);
        end
        first_frame_starts(chip) = start_search + frame_idx(1) - 1;
    end
    clear probe_data probe_reshaped probe_raw;

    frame_buffer = max(FS, VERIFY_INTERVAL);
    local_first_sample_abs = max(1, target_start_sample - frame_buffer);
    local_last_sample_abs = target_end_sample + frame_buffer;
    start_words = first_frame_starts + (local_first_sample_abs - 1) * words_per_frame;
    end_words = first_frame_starts + local_last_sample_abs * words_per_frame - 1;
    local_cycle_start = max(0, floor((min(start_words) - 1) / BLOCK_SIZE_WORDS));
    local_cycle_end = max(local_cycle_start, ceil(max(end_words) / BLOCK_SIZE_WORDS) - 1);
    bytes_to_read = (local_cycle_end - local_cycle_start + 1) * BLOCK_SIZE_BYTES * NUM_CHIPS;
    data_start_pos = sync_start_pos + local_cycle_start * BLOCK_SIZE_BYTES * NUM_CHIPS;
    fseek(fid, data_start_pos, 'bof');
    remaining_bytes = file_size - ftell(fid);
    bytes_to_read = min(bytes_to_read, remaining_bytes);
    words_to_read = floor(bytes_to_read / 2);
    words_to_read = floor(words_to_read / (BLOCK_SIZE_WORDS * NUM_CHIPS)) * (BLOCK_SIZE_WORDS * NUM_CHIPS);
    fprintf('Local segment read: target %.3f-%.3f sec, buffer %.3f sec, cycles %d-%d\n', ...
        segment_start_sec, segment_start_sec + segment_duration_sec, frame_buffer / FS, local_cycle_start, local_cycle_end);
else
    fseek(fid, sync_start_pos, 'bof');
    remaining_bytes = file_size - ftell(fid);
    bytes_to_read = min(remaining_bytes, MAX_READ_BYTES);
    words_to_read = floor(bytes_to_read / 2);
    words_to_read = floor(words_to_read / BLOCK_SIZE_WORDS) * BLOCK_SIZE_WORDS;
end
if words_to_read <= 0
    error('No complete 64KB block can be read.');
end

fprintf('Reading %.2f MB (%d uint16 words)\n', words_to_read * 2 / 1024 / 1024, words_to_read);
all_data = fread(fid, words_to_read, 'uint16=>uint16', 0, 'l');
if isempty(all_data)
    error('No data was read from the BIN file.');
end

total_words = length(all_data);
num_complete_cycles = floor(total_words / (BLOCK_SIZE_WORDS * NUM_CHIPS));
valid_words = num_complete_cycles * BLOCK_SIZE_WORDS * NUM_CHIPS;
fprintf('Complete 64KB block cycles: %d\n', num_complete_cycles);

reshaped_data = reshape(all_data(1:valid_words), BLOCK_SIZE_WORDS, NUM_CHIPS, num_complete_cycles);
chip_raw_data = cell(NUM_CHIPS, 1);
for chip = 1:NUM_CHIPS
    chip_raw_data{chip} = reshape(reshaped_data(:, chip, :), [], 1);
end
clear all_data reshaped_data;
read_split_sec = toc(parser_stage_timer);
fprintf('TIMING read_and_split_chips %.3f sec\n', read_split_sec);
parser_stage_timer = tic;

%% 3. 数据提取与帧头校验 (向量化切片优化)
chip_data = cell(NUM_CHIPS, 1);
frame_counts = zeros(1, NUM_CHIPS);
frame_check_failures = {};
frame_pattern = [uint16(hex2dec('FFFF')), uint16(hex2dec('0000'))];

for chip = 1:NUM_CHIPS
    raw = chip_raw_data{chip};
    if isempty(raw)
        warning('Chip %d has no raw data.', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end

    % 寻找局部同步头
    if local_segment_read
        first_frame_start = round(first_frame_starts(chip) + (local_first_sample_abs - 1) * words_per_frame - local_cycle_start * BLOCK_SIZE_WORDS);
        if first_frame_start < 1 || first_frame_start > length(raw)
            error('Chip %d local frame offset %d is outside the local read window.', chip, first_frame_start);
        end
        fprintf('Chip %d local first frame offset: %d (absolute sample %d)\n', chip, first_frame_start, local_first_sample_abs);
    else
    sync_idx = strfind(raw', sync_pattern);
    start_search = 1;
    if ~isempty(sync_idx)
        sync_start = sync_idx(1);
        start_search = sync_start + 10;
        fprintf('Chip %d local sync offset: %d\n', chip, sync_start);
    else
        warning('Chip %d local sync header was not found.', chip);
    end

    % 寻找第一帧
    frame_idx = strfind(raw(start_search:end)', frame_pattern);
    frame_idx = strfind(raw(start_search:end)', frame_pattern);
    if isempty(frame_idx)
        warning('Chip %d first frame header was not found.', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end
    first_frame_start = start_search + frame_idx(1) - 1;
    fprintf('Chip %d first frame offset: %d\n', chip, first_frame_start);
    end

    % 重组帧矩阵
    total_elements = floor((length(raw) - first_frame_start + 1) / words_per_frame) * words_per_frame;
    total_elements = floor((length(raw) - first_frame_start + 1) / words_per_frame) * words_per_frame;
    if total_elements <= 0
        warning('Chip %d has no complete frame.', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end

    max_frames = total_elements / words_per_frame;
    raw_frames = raw(first_frame_start : first_frame_start + total_elements - 1);
    frames_matrix = reshape(raw_frames, words_per_frame, max_frames);

    chip_data{chip} = frames_matrix(3:end, :);
    frame_counts(chip) = max_frames;

    % 批量帧头校验
    verify_cols = VERIFY_INTERVAL:VERIFY_INTERVAL:max_frames;
    bad_headers_idx = verify_cols(frames_matrix(1, verify_cols) ~= hex2dec('FFFF') | ...
                                  frames_matrix(2, verify_cols) ~= hex2dec('0000'));
    for i = 1:length(bad_headers_idx)
        frm = bad_headers_idx(i);
        header_pos = first_frame_start + (frm - 1) * words_per_frame;
        frame_check_failures{end+1} = struct('chip', chip, 'frame', frm, 'offset', header_pos, ...
            'header', [frames_matrix(1, frm), frames_matrix(2, frm)]); %#ok<AGROW>
    end

    fprintf('Chip %d frames: %d\n', chip, max_frames);
end
clear chip_raw_data;

total_checks = sum(floor(frame_counts(1:NUM_CHIPS) / VERIFY_INTERVAL));
fprintf('Frame header checks: %d, failures: %d\n', total_checks, length(frame_check_failures));
frame_extract_sec = toc(parser_stage_timer);
fprintf('TIMING frame_extract %.3f sec\n', frame_extract_sec);
parser_stage_timer = tic;

%% 4. 转换为电压 & 时间切片准备数据
ADC_TO_MV = 1000 * 5 / 32768;
for chip = 1:NUM_CHIPS
    chip_data{chip} = double(chip_data{chip}) * ADC_TO_MV - 2500;
end

common_frames = min(frame_counts(frame_counts > 0));
if isempty(common_frames) || common_frames <= 0
    error('No valid frames were extracted.');
end

if local_segment_read
    start_sample = target_start_sample - local_first_sample_abs + 1;
else
    start_sample = floor(segment_start_sec * FS) + 1;
end
if start_sample < 1
    start_sample = 1;
end
if segment_duration_sec > 0
    end_sample = min(common_frames, start_sample + target_duration_samples - 1);
else
    end_sample = common_frames;
end
if start_sample > common_frames
    error('Segment start %.3f sec exceeds available duration %.3f sec.', segment_start_sec, common_frames / FS);
end
if end_sample < start_sample
    error('Selected segment is empty.');
end

selected_frames = end_sample - start_sample + 1;
signal = cell(NUM_CHIPS, 1);
rawData512 = zeros(selected_frames, NUM_CHIPS * CH_PER_FRAME);
for chip = 1:NUM_CHIPS
    signal{chip} = chip_data{chip}(:, start_sample:end_sample);
    col_start = (chip - 1) * CH_PER_FRAME + 1;
    col_end = chip * CH_PER_FRAME;
    rawData512(:, col_start:col_end) = signal{chip}';
end

if local_segment_read
    selected_start_sec = (local_first_sample_abs + start_sample - 2) / FS;
else
    selected_start_sec = (start_sample - 1) / FS;
end
time = (0:selected_frames-1) / FS + selected_start_sec;
data_unit = 'mV';
selected_duration_sec = selected_frames / FS;
if local_segment_read
    total_cycles_est = floor((file_size - sync_start_pos) / (BLOCK_SIZE_BYTES * NUM_CHIPS));
    total_frames_est = floor((total_cycles_est * BLOCK_SIZE_WORDS - max(first_frame_starts) + 1) / words_per_frame);
    available_duration_sec = max(total_frames_est, 0) / FS;
else
    available_duration_sec = common_frames / FS;
end
convert_crop_sec = toc(parser_stage_timer);
fprintf('TIMING convert_and_crop %.3f sec\n', convert_crop_sec);

if ~isempty(output_mat_path)
    out_mat = output_mat_path;
else
    segment_tag = sprintf('%0.3fs_%0.3fs', selected_start_sec, selected_duration_sec);
    segment_tag = strrep(segment_tag, '.', 'p');
    save_name = sprintf('Dog_%d_Block-%d_Rec_allChans_segment_%s.h5', animal, blocknum, segment_tag);
    out_mat = fullfile(output_dir, save_name);
end
out_parent = fileparts(out_mat);
if ~isempty(out_parent) && ~exist(out_parent, 'dir')
    mkdir(out_parent);
end

fprintf('Saving HDF5: %s\n', out_mat);
fprintf('DT1_DATE=%s\n', dt1_date);
parser_stage_timer = tic;
if exist(out_mat, 'file')
    delete(out_mat);
end
write_h5_numeric(out_mat, '/FS', FS);
write_h5_numeric(out_mat, '/rawData512', rawData512);
write_h5_numeric(out_mat, '/time', time);
write_h5_numeric(out_mat, '/frame_counts', frame_counts);
write_h5_numeric(out_mat, '/selected_start_sec', selected_start_sec);
write_h5_numeric(out_mat, '/selected_duration_sec', selected_duration_sec);
write_h5_numeric(out_mat, '/available_duration_sec', available_duration_sec);
write_h5_numeric(out_mat, '/segment_start_sec', segment_start_sec);
write_h5_numeric(out_mat, '/segment_duration_sec', segment_duration_sec);
write_h5_numeric(out_mat, '/sync_start_pos', sync_start_pos);
frame_check_failures_matrix = frame_failures_to_matrix(frame_check_failures);
write_h5_numeric(out_mat, '/frame_check_failures', frame_check_failures_matrix);
write_h5_numeric(out_mat, '/frame_check_failure_count', numel(frame_check_failures));
write_h5_numeric(out_mat, '/dt1', datenum(dt1));
write_h5_numeric(out_mat, '/dt2', datenum(dt2));
write_h5_numeric(out_mat, '/deltaT1', deltaT1);
write_h5_numeric(out_mat, '/deltaT2', deltaT2);
write_h5_text(out_mat, '/bin_filename', bin_filename);
write_h5_text(out_mat, '/dt1_date', dt1_date);
write_h5_text(out_mat, '/deltaT1_unit', deltaT1_unit);
write_h5_text(out_mat, '/data_unit', data_unit);
if ~isempty(source_bin_text) || ~isempty(record_date)
    write_h5_text(out_mat, '/source_bin', source_bin_text);
    write_h5_text(out_mat, '/date', record_date);
    write_h5_numeric(out_mat, '/animal', animal);
    write_h5_numeric(out_mat, '/block', blocknum);
end
save_mat_sec = toc(parser_stage_timer);
parser_total_sec = toc(parser_total_timer);
fprintf('TIMING save_h5 %.3f sec\n', save_mat_sec);
fprintf('TIMING total_matlab %.3f sec\n', parser_total_sec);
fprintf('TIMING pct metadata %.1f sync_search %.1f read_split %.1f frame_extract %.1f convert_crop %.1f save_mat %.1f\n', ...
    100 * timing_metadata_sec / parser_total_sec, ...
    100 * sync_search_sec / parser_total_sec, ...
    100 * read_split_sec / parser_total_sec, ...
    100 * frame_extract_sec / parser_total_sec, ...
    100 * convert_crop_sec / parser_total_sec, ...
    100 * save_mat_sec / parser_total_sec);
fprintf('Done. Selected duration %.3f sec, available duration %.3f sec.\n', selected_duration_sec, available_duration_sec);
end

function write_h5_numeric(filename, dataset_name, value)
value = double(value);
if isempty(value)
    return;
end
shape = size(value);
if isscalar(value)
    shape = [1 1];
end
h5create(filename, dataset_name, shape, 'Datatype', 'double');
h5write(filename, dataset_name, value);
end

function matrix = frame_failures_to_matrix(failures)
% Store cell/struct diagnostics as HDF5-compatible numeric rows:
% chip, frame, file-word offset, header word 1, header word 2.
if isempty(failures)
    matrix = zeros(1, 5);
    return;
end
matrix = zeros(numel(failures), 5);
for k = 1:numel(failures)
    item = failures{k};
    matrix(k, :) = [double(item.chip), double(item.frame), double(item.offset), ...
        double(item.header(1)), double(item.header(2))];
end
end

function write_h5_text(filename, dataset_name, value)
bytes = uint8(char(value));
if isempty(bytes)
    bytes = uint8(0);
end
h5create(filename, dataset_name, size(bytes), 'Datatype', 'uint8');
h5write(filename, dataset_name, bytes);
end
