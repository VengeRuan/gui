clear all
clc
datafiledir='D:\Animal experiment\Data\PilotStudy2\预实验3_犬1107_1109\';
[fname, pathname] = uigetfile([datafiledir 'rawbin\*.bin'], 'Select a BIN-file');
if fname==0
    return
end
bin_filename=[pathname fname];
animal=1107;
blocknum=100;
savematdata=1;
saveallch=1; %保存全部通道或者部分通道
MAX_READ_BYTES = 500 * 1024 * 1024;   % 最大读取扇区的大小--400 MB
save_chipix=[1 92;1 124;2 9;2 113;3 93;4 97;4 103];  %部分通道先保存成mat文件，供后续SNR等的计算和分析
FS = 6490;                           % 采样率 (Hz)
savefiledir=[datafiledir 'Dog_' num2str(animal) '_Block-' num2str(blocknum)];

% ==================== SD卡储存配置参数 ====================
OFFSET = hex2dec('400000');
FRAME_HEAD = [hex2dec('FFFF'); hex2dec('0000')];       % 帧头
CH_PER_FRAME = 128;                 % 每个芯片每帧通道数
NUM_CHIPS = 4;
BLOCK_SIZE_BYTES = 64 * 1024;       % 64KB
BLOCK_SIZE_WORDS = BLOCK_SIZE_BYTES / 2;  % 32768
words_per_frame = 2 + CH_PER_FRAME; % 130字/帧
save_allchipix=1:128; %全部通道先保存成mat文件，供后续SNR等的计算和分析

% ==================== 滤波配置参数 ====================
FILTER_ENABLE = 0;
LOWPASS_CUTOFF = 300;
HIGHPASS_CUTOFF2 = 0.5;
HIGHPASS_CUTOFF = 1000;
NOTCH_FREQ = 50;
NOTCH_Q = 30;
filter_type1='high';
filter_type2='low';
% =====================================================
% 指定绘图的通道（芯片 -> 通道列表）
plotOn=1;
plot_channels = containers.Map('KeyType', 'double', 'ValueType', 'any');
plot_channels(1) = [92, 124];
plot_channels(2) = [9, 113];
plot_channels(3) = [93, 117];
plot_channels(4) = [97, 103];

% 滤波函数（需提前定义，否则后面的apply_filter会报错）
function data_filtered = apply_filter(data, fs, cutoff, notch_freq, notch_q, type)
    nyq = fs / 2;
    % 陷波滤波器设计
    notch_norm = notch_freq / nyq;
    [b_notch, a_notch] = iirnotch(notch_norm, notch_norm/notch_q);
    % 根据type选择高通或低通
    if strcmp(type, 'high')
        norm_cut = cutoff / nyq;
        [b_main, a_main] = butter(4, norm_cut, 'high');
    elseif strcmp(type, 'low')
        norm_cut = cutoff / nyq;
        [b_main, a_main] = butter(4, norm_cut, 'low');
    else
        error('不支持的滤波类型');
    end
    data_filtered = zeros(size(data));
    for ch = 1:size(data,1)
        x = double(data(ch,:));
        % 先陷波
        y = filtfilt(b_notch, a_notch, x);
        % 再主滤波
        y = filtfilt(b_main, a_main, y);
        data_filtered(ch,:) = y;
    end
end

fid = fopen(bin_filename, 'rb');
if fid == -1, error('无法打开文件'); end
cleanup = onCleanup(@() fclose(fid));
fseek(fid, OFFSET, 'bof');
file_size = dir(bin_filename).bytes;

%% 1. 仅在文件末尾 10 MB 内搜索芯片1同步头（后9字均为FFFF，首字任意）
FRAME_SEARCH_BYTES = 10 * 1024 * 1024;
search_start_pos = max(OFFSET, file_size - FRAME_SEARCH_BYTES);
search_start_pos = search_start_pos + mod(search_start_pos, 2); % uint16 边界对齐

fseek(fid, search_start_pos, 'bof');
search_words = fread(fid, floor((file_size - search_start_pos) / 2), ...
    'uint16=>uint16', 0, 'l');

% 找到连续9个 FFFF；其前一个字为同步头的首字（该字可为任意值）
sync_pattern = repmat(uint16(hex2dec('FFFF')), 1, 9);
sync_indices = strfind(search_words', sync_pattern);
sync_indices = sync_indices(sync_indices > 1); % 确保同步头首字也在搜索缓冲区内

if isempty(sync_indices)
    error('未在文件末尾 %.1f MB 内找到芯片1同步头', ...
        (file_size - search_start_pos) / 1024 / 1024);
end

sync_start_pos = search_start_pos + (sync_indices(1) - 2) * 2; % 同步头起始位置
fprintf('找到芯片1同步头，位置: 0x%X\n', sync_start_pos);

%% 读取时间信息块（0x400000 前512字节）
TIME_BLOCK_OFFSET = OFFSET - 512;   % 0x3FFE00
if TIME_BLOCK_OFFSET >= 0 && TIME_BLOCK_OFFSET + 512 <= file_size
    % 保存当前指针位置（当前在 OFFSET 处）
    current_pos = ftell(fid);
    fseek(fid, TIME_BLOCK_OFFSET, 'bof');
    time_block_data = fread(fid, 512, 'uint8=>uint8');
    fprintf('成功读取时间数据块，位于文件偏移 0x%X (%d 字节)\n', TIME_BLOCK_OFFSET, length(time_block_data));
    
    % 提取两个时间戳（假设前16字节为大端64位微秒时间戳）
    if length(time_block_data) >= 16
        % 第一个时间戳（字节1-8）
        ts1_us = uint64(0);
        for i = 1:8
            ts1_us = bitor(bitshift(ts1_us, 8), uint64(time_block_data(i)));
        end
        % 第二个时间戳（字节9-16）
        ts2_us = uint64(0);
        for i = 9:16
            ts2_us = bitor(bitshift(ts2_us, 8), uint64(time_block_data(i)));
        end
        
        % 转换为秒（微秒→秒）
        ts1_sec = double(ts1_us) / 1e6;
        ts2_sec = double(ts2_us) / 1e6;
        % 加8小时（UTC+8）
        ts1_sec = ts1_sec + 8*3600;
        ts2_sec = ts2_sec + 8*3600;
        
        % 创建 datetime 对象
        dt1 = datetime(ts1_sec, 'ConvertFrom', 'posixtime', 'Format', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
        dt2 = datetime(ts2_sec, 'ConvertFrom', 'posixtime', 'Format', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
        time_diff_sec = abs(ts2_sec - ts1_sec);
        
        fprintf('时间戳1: %s\n', datestr(dt1, 'yyyy-mm-dd HH:MM:SS.FFF'));
        fprintf('时间戳2: %s\n', datestr(dt2, 'yyyy-mm-dd HH:MM:SS.FFF'));
        fprintf('时间差: %.6f 秒\n', time_diff_sec);
    else
        warning('时间块数据长度不足16字节，无法提取时间戳');
        dt1 = []; dt2 = []; time_diff_sec = NaN;
    end
    
    % 显示前32字节的十六进制，供手动分析
    fprintf('前32字节（十六进制）: ');
    for i = 1:min(32, length(time_block_data))
        fprintf('%02X ', time_block_data(i));
        if mod(i,16)==0, fprintf('\n                     '); end
    end
    fprintf('\n');
    
    % 恢复指针到 OFFSET
    fseek(fid, current_pos, 'bof');
else
    warning('时间块位置无效（偏移 0x%X），可能文件不包含时间信息', TIME_BLOCK_OFFSET);
    time_block_data = [];
    dt1 = []; dt2 = []; time_diff_sec = NaN;
end
%% 1.5 读取前一个64KB块（时间信息块）
% 计算同步头所在块的起始偏移（对齐到64KB）
block_start = floor(sync_start_pos / BLOCK_SIZE_BYTES) * BLOCK_SIZE_BYTES;
time_block_offset = block_start - BLOCK_SIZE_BYTES;   % 前一个块的起始偏移
if time_block_offset >= 0 && time_block_offset + BLOCK_SIZE_BYTES <= file_size
    % 保存当前指针位置，以便后续恢复
    current_pos = ftell(fid);
    fseek(fid, time_block_offset, 'bof');
    time_block_data = fread(fid, BLOCK_SIZE_BYTES, 'uint8=>uint8');
    fprintf('成功读取时间数据块，位于文件偏移 0x%X (%d 字节)\n', time_block_offset, length(time_block_data));
    % 尝试解析为Unix时间戳（小端，uint32）
    if length(time_block_data) >= 4
        unix_timestamp = typecast(time_block_data(1:4), 'uint32');
        fprintf('可能的Unix时间戳（小端）: %u (对应 %s)\n', unix_timestamp, datestr(datetime(unix_timestamp, 'ConvertFrom', 'posixtime')));
    else
        unix_timestamp = [];
    end
    % 显示前32字节的十六进制，供手动分析
    fprintf('前32字节（十六进制）: ');
    for i = 1:min(32, length(time_block_data))
        fprintf('%02X ', time_block_data(i));
        if mod(i,16)==0, fprintf('\n                     '); end
    end
    fprintf('\n');
    % 恢复文件指针
    fseek(fid, current_pos, 'bof');
else
    warning('时间块位置无效（偏移 0x%X），可能文件不包含时间信息', time_block_offset);
    time_block_data = [];
    unix_timestamp = [];
end

%% 2. 按64KB块循环读取，分配至芯片（预分配+直接填充）
fseek(fid, sync_start_pos, 'bof');
remaining_bytes = file_size - ftell(fid);
bytes_to_read = min(remaining_bytes, MAX_READ_BYTES);
words_to_read = floor(bytes_to_read / 2);
words_to_read = floor(words_to_read / BLOCK_SIZE_WORDS) * BLOCK_SIZE_WORDS;

fprintf('限制读取数据量: 最多 %.2f MB，实际读取 %d 字 (%.2f MB)\n', ...
    MAX_READ_BYTES/1024/1024, words_to_read, words_to_read*2/1024/1024);

all_data = fread(fid, words_to_read, 'uint16=>uint16', 0, 'l');
if isempty(all_data), error('未能读取到数据'); end
total_words = length(all_data);
fprintf('实际读取到 %d 字数据\n', total_words);
num_blocks = floor(total_words / BLOCK_SIZE_WORDS);

% 统计每个芯片的块数
chip_block_counts = zeros(1, NUM_CHIPS);
for blk = 0:num_blocks-1
    chip_idx = mod(blk, NUM_CHIPS) + 1;
    chip_block_counts(chip_idx) = chip_block_counts(chip_idx) + 1;
end

% 预分配每个芯片的原始数据矩阵
chip_raw_data = cell(NUM_CHIPS, 1);
for chip = 1:NUM_CHIPS
    if chip_block_counts(chip) > 0
        chip_raw_data{chip} = zeros(BLOCK_SIZE_WORDS * chip_block_counts(chip), 1, 'uint16');
    else
        chip_raw_data{chip} = [];
    end
end

% 填充数据
write_pos = ones(1, NUM_CHIPS);
for blk = 0:num_blocks-1
    chip_idx = mod(blk, NUM_CHIPS) + 1;
    start_idx = blk * BLOCK_SIZE_WORDS + 1;
    end_idx = (blk+1) * BLOCK_SIZE_WORDS;
    block_data = all_data(start_idx:end_idx);
    offset = (write_pos(chip_idx) - 1) * BLOCK_SIZE_WORDS + 1;
    chip_raw_data{chip_idx}(offset : offset+BLOCK_SIZE_WORDS-1) = block_data;
    write_pos(chip_idx) = write_pos(chip_idx) + 1;
end

%% 3. 对每个芯片（1,2,3,4）打印同步头和首帧帧头，提取数据并定期校验帧头（静默存储）
chip_data = cell(NUM_CHIPS, 1);
frame_counts = zeros(1, NUM_CHIPS);
VERIFY_INTERVAL = 100;                % 每100帧校验一次帧头
frame_check_failures = {};            % 存储校验失败记录

for chip = 1:NUM_CHIPS
    raw = chip_raw_data{chip};
    if isempty(raw)
        warning('芯片%d 无数据', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end

    % 搜索同步头（后9字均为0xFFFF）
    sync_start = [];
    for idx = 1:length(raw)-9
        if all(raw(idx+1:idx+9) == hex2dec('FFFF'))
            sync_start = idx;
            break;
        end
    end
    if isempty(sync_start)
        warning('芯片%d 未找到同步头', chip);
    else
        fprintf('芯片%d: 同步头本地偏移 %d, 首字=0x%04X\n', chip, sync_start, raw(sync_start));
    end

    % 搜索第一个帧头
    start_search = 1;
    if ~isempty(sync_start)
        start_search = sync_start + 10;
    end
    first_frame_start = [];
    for idx = start_search : length(raw)-1
        if raw(idx) == hex2dec('FFFF') && raw(idx+1) == hex2dec('0000')
            first_frame_start = idx;
            break;
        end
    end
    if isempty(first_frame_start)
        warning('芯片%d 未找到有效帧头', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end
    fprintf('芯片%d: 首帧帧头本地偏移 %d\n', chip, first_frame_start);

    % 提取数据并定期校验
    data_start = first_frame_start + 2;
    total_words_after = length(raw) - data_start + 1;
    if total_words_after < CH_PER_FRAME
        warning('芯片%d 数据不足一帧', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end

    max_frames = floor(total_words_after / words_per_frame);
    if max_frames == 0
        warning('芯片%d 无完整帧', chip);
        chip_data{chip} = zeros(CH_PER_FRAME, 0);
        continue;
    end

    all_channels = zeros(CH_PER_FRAME, max_frames, 'uint16');
    cur_data_pos = data_start;
    for frm = 1:max_frames
        all_channels(:, frm) = raw(cur_data_pos : cur_data_pos+CH_PER_FRAME-1);

        % 定期校验帧头（不打印，仅记录）
        if mod(frm, VERIFY_INTERVAL) == 0
            header_pos = cur_data_pos - 2;
            if header_pos >= 1 && header_pos+1 <= length(raw)
                h1 = raw(header_pos);
                h2 = raw(header_pos+1);
                if ~(h1 == hex2dec('FFFF') && h2 == hex2dec('0000'))
                    frame_check_failures{end+1} = struct(...
                        'chip', chip, ...
                        'frame', frm, ...
                        'offset', header_pos, ...
                        'header', [h1, h2]);
                end
            else
                frame_check_failures{end+1} = struct(...
                    'chip', chip, ...
                    'frame', frm, ...
                    'offset', header_pos, ...
                    'header', [NaN, NaN]);
            end
        end

        cur_data_pos = cur_data_pos + words_per_frame;
    end

    chip_data{chip} = all_channels;
    frame_counts(chip) = max_frames;
    fprintf('芯片%d 提取 %d 帧\n', chip, max_frames);
end

% 输出帧头校验统计
total_checks = sum(floor(frame_counts([1,3,4]) / VERIFY_INTERVAL));
fprintf('\n========== 帧头校验结果 ==========\n');
fprintf('总校验次数: %d, 失败次数: %d\n', total_checks, length(frame_check_failures));
if ~isempty(frame_check_failures)
    fprintf('详细失败记录保存在变量 frame_check_failures 中。\n');
end


%% 4. ADC转电压 (mV)
ADC_TO_mV = 5 / 32768;
for chip = 1:NUM_CHIPS
    chip_data{chip} = double(chip_data{chip}) * ADC_TO_mV - 2.5;
end

% 保存数据
if savematdata
    xx = 1:size(chip_data{4},2);
    time = (xx-1)/FS;
    if saveallch
        signal = chip_data;
        % 使用 -v7.3 支持大于2GB的变量
        save([savefiledir '_Rec_allChans.mat'], 'FS', 'signal', 'time', 'time_block_data', 'unix_timestamp', '-v7.3');
    else
        for savefile = 1:size(save_chipix,1)
            chipix = save_chipix(savefile,1);
            chix = save_chipix(savefile,2);
            signal = chip_data{chipix}(chix,:);
            save([savefiledir '_Rec_chip' num2str(chipix) '_ch' num2str(chix) '.mat'], 'FS', 'signal', 'time');
        end
    end
end

%% 5. 应用滤波
chip_data_filtered = cell(NUM_CHIPS, 1);
if FILTER_ENABLE
    fprintf('正在应用滤波: 低通 %.1f Hz, 陷波 %d Hz\n', LOWPASS_CUTOFF, NOTCH_FREQ);
    chip_data_filtered_high = cell(NUM_CHIPS,1);
    chip_data_filtered_low = cell(NUM_CHIPS,1);
    for chip = 1:NUM_CHIPS
        if ~isempty(chip_data{chip})
            chip_data_filtered_high{chip} = apply_filter(chip_data{chip}, FS, ...
                HIGHPASS_CUTOFF, NOTCH_FREQ, NOTCH_Q, 'high');
            chip_data_filtered_low{chip} = apply_filter(chip_data{chip}, FS, ...
                LOWPASS_CUTOFF, NOTCH_FREQ, NOTCH_Q, 'low');
        else
            chip_data_filtered_high{chip} = [];
            chip_data_filtered_low{chip} = [];
        end
    end
else
    chip_data_filtered = chip_data;
end

%% 6. 绘图
if plotOn
    figure('Name', 'Raw', 'Position', [100,100,1200,500]);
    plot_idx = 1;
    for cid = [1,2,3,4]
        if ~isKey(plot_channels, cid), continue; end
        data_mat = chip_data{cid};
        if isempty(data_mat), continue; end
        [~, n_frames] = size(data_mat);
        ch_list = plot_channels(cid);
        for ch = ch_list
            if plot_idx > 8, break; end
            subplot(4,2,plot_idx);
            plot(1:n_frames, data_mat(ch,:), 'LineWidth',1.2);
            xlabel('采样帧序号'); ylabel('电压 (mV)');
            title(sprintf('芯片%d, 通道%d', cid, ch));
            grid on;
            plot_idx = plot_idx+1;
        end
    end
    sgtitle(sprintf('数据解析 (raw): %s', bin_filename));

    if FILTER_ENABLE
        figure('Name', 'Low-pass', 'Position', [100,100,1200,500]);
        plot_idx = 1;
        for cid = [1,2,3,4]
            if ~isKey(plot_channels, cid), continue; end
            data_mat = chip_data_filtered_low{cid};
            if isempty(data_mat), continue; end
            [~, n_frames] = size(data_mat);
            ch_list = plot_channels(cid);
            for ch = ch_list
                if plot_idx > 8, break; end
                subplot(4,2,plot_idx);
                plot(1:n_frames, data_mat(ch,:), 'LineWidth',1.2);
                xlabel('采样帧序号'); ylabel('电压 (mV)');
                title(sprintf('芯片%d, 通道%d (低通)', cid, ch));
                grid on;
                plot_idx = plot_idx+1;
            end
        end
        sgtitle(sprintf('数据解析 (低通滤波后): %s', bin_filename));

        figure('Name', 'High-pass', 'Position', [100,100,1200,500]);
        plot_idx = 1;
        for cid = [1,2,3,4]
            if ~isKey(plot_channels, cid), continue; end
            data_mat = chip_data_filtered_high{cid};
            if isempty(data_mat), continue; end
            [~, n_frames] = size(data_mat);
            ch_list = plot_channels(cid);
            for ch = ch_list
                if plot_idx > 8, break; end
                subplot(4,2,plot_idx);
                plot(1:n_frames, data_mat(ch,:), 'LineWidth',1.2);
                xlabel('采样帧序号'); ylabel('电压 (mV)');
                title(sprintf('芯片%d, 通道%d (高通)', cid, ch));
                grid on;
                plot_idx = plot_idx+1;
            end
        end
        sgtitle(sprintf('数据解析 (高通滤波后): %s', bin_filename));
    end
end
