function out_mat = run_read_all_channelsBIN_segmented_v2(bin_filename, output_dir, animal, blocknum, segment_start_sec, segment_duration_sec, max_read_mb)
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

fseek(fid, OFFSET, 'bof');
found_sync = false;
while ftell(fid) < file_size - 20
    w = fread(fid, 1, 'uint16=>uint16', 0, 'l'); %#ok<NASGU>
    if isempty(w)
        break;
    end
    pos_after_first = ftell(fid);
    next9 = fread(fid, 9, 'uint16=>uint16', 0, 'l');
    if length(next9) == 9 && all(next9 == hex2dec('FFFF'))
        found_sync = true;
        break;
    else
        fseek(fid, pos_after_first, 'bof');
    end
end

if ~found_sync
    error('Sync header was not found.');
end

sync_start_pos = ftell(fid) - 20;
fprintf('Sync start position: 0x%X\n', sync_start_pos);

fseek(fid, sync_start_pos, 'bof');
remaining_bytes = file_size - ftell(fid);
bytes_to_read = min(remaining_bytes, MAX_READ_BYTES);
words_to_read = floor(bytes_to_read / 2);
words_to_read = floor(words_to_read / BLOCK_SIZE_WORDS) * BLOCK_SIZE_WORDS;
if words_to_read <= 0
    error('No complete 64KB block can be read.');
end

fprintf('Reading %.2f MB (%d uint16 words)\n', words_to_read * 2 / 1024 / 1024, words_to_read);
all_data = fread(fid, words_to_read, 'uint16=>uint16', 0, 'l');
if isempty(all_data)
    error('No data was read from the BIN file.');
end

total_words = length(all_data);
num_blocks = floor(total_words / BLOCK_SIZE_WORDS);
fprintf('Complete 64KB blocks: %d\n', num_blocks);

chip_block_counts = zeros(1, NUM_CHIPS);
for blk = 0:num_blocks-1
    chip_idx = mod(blk, NUM_CHIPS) + 1;
    chip_block_counts(chip_idx) = chip_block_counts(chip_idx) + 1;
end

chip_raw_data = cell(NUM_CHIPS, 1);
for chip = 1:NUM_CHIPS
    chip_raw_data{chip} = zeros(BLOCK_SIZE_WORDS * chip_block_counts(chip), 1, 'uint16');
end

write_pos = ones(1, NUM_CHIPS);
for blk = 0:num_blocks-1
    chip_idx = mod(blk, NUM_CHIPS) + 1;
    start_idx = blk * BLOCK_SIZE_WORDS + 1;
    end_idx = (blk + 1) * BLOCK_SIZE_WORDS;
    offset = (write_pos(chip_idx) - 1) * BLOCK_SIZE_WORDS + 1;
    chip_raw_data{chip_idx}(offset:offset + BLOCK_SIZE_WORDS - 1) = all_data(start_idx:end_idx);
    write_pos(chip_idx) = write_pos(chip_idx) + 1;
end
clear all_data;

chip_data = cell(NUM_CHIPS, 1);
frame_counts = zeros(1, NUM_CHIPS);
frame_check_failures = {};

for chip = 1:NUM_CHIPS
    raw = chip_raw_data{chip};
    if isempty(raw)
        warning('Chip %d has no raw data.', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end

    sync_start = [];
    for idx = 1:length(raw)-9
        if all(raw(idx+1:idx+9) == hex2dec('FFFF'))
            sync_start = idx;
            break;
        end
    end

    start_search = 1;
    if ~isempty(sync_start)
        start_search = sync_start + 10;
        fprintf('Chip %d local sync offset: %d\n', chip, sync_start);
    else
        warning('Chip %d local sync header was not found.', chip);
    end

    first_frame_start = [];
    for idx = start_search:length(raw)-1
        if raw(idx) == hex2dec('FFFF') && raw(idx+1) == hex2dec('0000')
            first_frame_start = idx;
            break;
        end
    end

    if isempty(first_frame_start)
        warning('Chip %d first frame header was not found.', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end
    fprintf('Chip %d first frame offset: %d\n', chip, first_frame_start);

    data_start = first_frame_start + 2;
    total_words_after = length(raw) - data_start + 1;
    max_frames = floor(total_words_after / words_per_frame);
    if max_frames <= 0
        warning('Chip %d has no complete frame.', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end

    all_channels = zeros(CH_PER_FRAME, max_frames, 'uint16');
    cur_data_pos = data_start;
    for frm = 1:max_frames
        all_channels(:, frm) = raw(cur_data_pos:cur_data_pos + CH_PER_FRAME - 1);

        if mod(frm, VERIFY_INTERVAL) == 0
            header_pos = cur_data_pos - 2;
            h1 = raw(header_pos);
            h2 = raw(header_pos + 1);
            if ~(h1 == hex2dec('FFFF') && h2 == hex2dec('0000'))
                frame_check_failures{end+1} = struct('chip', chip, 'frame', frm, 'offset', header_pos, 'header', [h1, h2]); %#ok<AGROW>
            end
        end

        cur_data_pos = cur_data_pos + words_per_frame;
    end

    chip_data{chip} = all_channels;
    frame_counts(chip) = max_frames;
    fprintf('Chip %d frames: %d\n', chip, max_frames);
end
clear chip_raw_data;

total_checks = sum(floor(frame_counts(1:NUM_CHIPS) / VERIFY_INTERVAL));
fprintf('Frame header checks: %d, failures: %d\n', total_checks, length(frame_check_failures));

ADC_TO_V = 5 / 32768;
for chip = 1:NUM_CHIPS
    chip_data{chip} = double(chip_data{chip}) * ADC_TO_V - 2.5;
end

common_frames = min(frame_counts(frame_counts > 0));
if isempty(common_frames) || common_frames <= 0
    error('No valid frames were extracted.');
end

start_sample = floor(segment_start_sec * FS) + 1;
if start_sample < 1
    start_sample = 1;
end
if segment_duration_sec > 0
    end_sample = min(common_frames, start_sample + floor(segment_duration_sec * FS) - 1);
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

time = (0:selected_frames-1) / FS + (start_sample - 1) / FS;
selected_start_sec = (start_sample - 1) / FS;
selected_duration_sec = selected_frames / FS;
available_duration_sec = common_frames / FS;

segment_tag = sprintf('%0.3fs_%0.3fs', selected_start_sec, selected_duration_sec);
segment_tag = strrep(segment_tag, '.', 'p');
save_name = sprintf('Dog_%d_Block-%d_Rec_allChans_segment_%s.h5', animal, blocknum, segment_tag);
out_mat = fullfile(output_dir, save_name);

fprintf('Saving HDF5: %s\n', out_mat);
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
for chip = 1:NUM_CHIPS
    write_h5_numeric(out_mat, sprintf('/signal_chip%d', chip), signal{chip});
end
write_h5_text(out_mat, '/bin_filename', bin_filename);
write_h5_text(out_mat, '/dt1_date', dt1_date);
write_h5_text(out_mat, '/deltaT1_unit', deltaT1_unit);
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
