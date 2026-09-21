function [dt1, dt2, deltaT1, deltaT2] = func_read_time_info(filename)
% Read timing metadata embedded in the BIN file.
% deltaT1/deltaT2 are returned in milliseconds.

fid = fopen(filename, 'rb');
if fid == -1
    error('Cannot open file: %s', filename);
end
info = dir(filename);
scan_bytes = min(info.bytes, 100 * 1024 * 1024);
scan_start = max(0, info.bytes - scan_bytes);
fseek(fid, scan_start, 'bof');
data = fread(fid, scan_bytes, 'uint8=>uint8');
fclose(fid);

header = uint8([72 101 108 108 111 32 83 68 32 67 97 114 100 32 ...
                118 105 97 32 70 97 116 70 115 33 10]);
idx = strfind(data(:)', header);
if isempty(idx)
    error('Timing header was not found.');
end
base = idx(1) + 25;

off_T1 = 0;
off_A  = 8 + 24;
off_B  = off_A + 8 + 24;
off_T2 = off_B + 8 + 88;

le_uint32 = @(pos) uint32(data(pos)) + ...
                   uint32(data(pos + 1)) * 2^8 + ...
                   uint32(data(pos + 2)) * 2^16 + ...
                   uint32(data(pos + 3)) * 2^24;
le_uint64 = @(pos) uint64(data(pos)) + ...
                   uint64(data(pos + 1)) * 2^8 + ...
                   uint64(data(pos + 2)) * 2^16 + ...
                   uint64(data(pos + 3)) * 2^24 + ...
                   uint64(data(pos + 4)) * 2^32 + ...
                   uint64(data(pos + 5)) * 2^40 + ...
                   uint64(data(pos + 6)) * 2^48 + ...
                   uint64(data(pos + 7)) * 2^56;

T1_raw = le_uint64(base + off_T1);
T2_raw = le_uint64(base + off_T2);

tA1 = le_uint32(base + off_A);
tA2 = le_uint32(base + off_A + 4);
tB1 = le_uint32(base + off_B);
tB2 = le_uint32(base + off_B + 4);

diffA = tA1 - tA2;
diffB = tB1 - tB2;
deltaT1 = double(diffA) / 216.0;
deltaT2 = double(diffB) / 216.0;

ts1_sec = double(T1_raw) / 1e6 + 8 * 3600;
ts2_sec = double(T2_raw) / 1e6 + 8 * 3600;
dt1 = datetime(ts1_sec, 'ConvertFrom', 'posixtime', 'Format', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
dt2 = datetime(ts2_sec, 'ConvertFrom', 'posixtime', 'Format', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
end
