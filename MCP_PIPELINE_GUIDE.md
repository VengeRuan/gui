# v3.0.2 — 主窗口第 2 / 3 页 MCP 扩展

v3.0.2 新增主窗口预处理及时间对齐控制。沿用现有 `animal_peak_mcp_server.py`、SSH 启动脚本及 API v1，**不需要再安装一套 MCP 或新建 Conda 环境**。发布及部署范围仅包括本次 MCP 扩展，不包含其他尚未提交的分析代码修改。

## 开启和选择窗口

1. 更新项目后重启 Qt 程序（不热替换正在计算的程序）。
2. 主窗口左下勾选 **允许 MCP 控制 2/3页**，默认关闭。
3. 同一个 MCP 连接调用 `list_sessions`，选择 `session_kind="pipeline"` 的会话。
4. 先读 `get_state` 和 `describe_controls`。会话列表的文件名仅是注册时快照，**当前数据以 get_state.source 为准**。

原 AnimalStart 窗口会话为 `session_kind="animalstart"`，开关和参数独立，不会因主窗口授权自动授权测试窗口。不要混用两个窗口的 session_id。

## 功能对应

| 页面/功能 | 参数或操作 |
|---|---|
| 导入完整 H5 | `execute(load_h5, options={path, confirm:true})` |
| 映射 Excel、物理通道、滤波模式/截止/阶数/陷波、一键流程勾选 | `set_parameters(group="preprocess", values=…)` |
| 重映射、普通滤波、一键流程 | `execute(remap/filter/one_click, options={confirm:true})` |
| 饱和/2.5mV 附近集中检查阈值、并行参数 | `set_parameters(group="quality", values=…)` |
| 全部已勾选质量检测、单独检测 | `execute(quality_check/saturation_check/concentration_check, options={confirm:true})` |
| 人工健康/坏道/恢复自动 | `execute(review_channels, options={channels:[82], classification:"good"或"bad"或"auto", confirm:true})` |
| CAR | `execute(car, options={confirm:true})` |
| ICA 参数、指定通道执行、清除 | `set_parameters(group="ica", …)`；`execute(ica, options={channels:[…], confirm:true})`；`execute(clear_ica, options={confirm:true})` |
| 流式 LFP/Spike、陷波、重叠、CAR、内存和并发 | `set_parameters(group="stream", …)`；`execute(stream_export, options={confirm:true})` |
| 预处理 H5 + 配对 CSV | `execute(export_preprocessed, options={path:"新绝对路径.h5", confirm:true})` |
| 查看预处理前/后的指定通道和时间段 | `execute(refresh_preprocess, options={channel:82, start_sec:0, end_sec:5, full_duration:false, original:false})` |
| log/事件 CSV/视频 TXT、任务类型 | `set_parameters(group="alignment", values={log_path, events_path, detection_path, task_mode:"Letter"或"Flash"})` |
| 重试自动匹配（保留手动路径，不计算） | `execute(autofill_alignment)` |
| 对齐/Letter 健康通道名单 | `navigate(selection={page:"alignment", channels:[82,83]})`；`channels:null` 表示全部健康通道 |
| 计算对齐 | `execute(align, options={confirm:true, segment_index:0})` |
| 对齐预览通道 | `navigate(selection={channel:82})`，使用物理 ID，不使用列序号 |
| 对齐预览起点/时长/全时长 | `set_parameters(group="preview", values={start_sec:0, duration_sec:5, full_duration:false})`；然后 `execute(refresh_alignment)` |
| Epoch/基线/响应窗、Tag、Trial 数量、平滑/zscore等任务参数 | `set_parameters(group="task", …)` |
| Letter / Flash 任务入口、AnimalStart 测试入口 | `execute(open_letter/open_flash/open_animalstart)`；测试窗口仍需自己开启其 MCP 开关 |
| 预览图 PNG、复制工作站剪贴板、绘制样本 | `get_plot_image(plot="alignment"或"preprocess"或"original")`；`execute(copy_plot, options={plot:…})`；`read_waveform(plot=…)` |
| 对齐标记、CSV Dot/Letter 与行为匹配、日志候选时段、QC | `read_table(table="markers"或"annotations"或"segments"或"qc", offset=0, limit=100)` |

全部字段、范围、枚举以及每个动作可接受的参数均以会话的 `describe_controls` 为准。`set_parameters` 只暂存输入，不自动运行处理。

## 推荐调用顺序

1. `get_state(session_id)`，取得当前 `revision`。
2. 设置参数/通道时带 `expected_revision`，使用返回的新 revision。
3. `execute` 返回 `job_id`，使用 `get_jobs(session_id, job_id)` 等待 `completed` / `failed` / `cancelled`。
4. 检查 job 的 `error`、`messages`、最终状态，再读取结果和图；接口接受命令不等于计算完成。
5. 操作者改参数、换通道、换数据或改 QC 会使旧 revision 失效。重新读取状态，不自动重放超时写操作。

## 安全和结果核对

- 保持原有算法与 Worker、原始/重映射 QC 输入、物理通道和健康通道限制，不通过鼠标位置模拟操作。
- MCP 处理、换数据、人工 QC、对齐和导出需用户授权并传入 `confirm:true`。导入文件已记录同类处理时，还需单独 `repeat_confirm:true`，不静默接受“重复预处理”。
- 参数先整体验证再写入。不接受任意函数名、代码、shell、未知参数或无效通道；不悄悄丢弃不存在/不健康的通道。
- 日志包含多个任务时段：先读 `segments`，明确选择**零基** `segment_index`。结果中的 `program_segment_index` 是原有的**一基**编号。不会擅自选最后一段，也不会按 PROGRAM_END 裁剪数据。
- `markers/annotations` 的 `stale` 和 `get_state.alignment.stale` 表示路径、文件时间戳、数据源、通道/QC 等与上次计算不一致。预览参数变更后须刷新，不会把旧图作为新预览返回。
- CSV Dot 起点仍只取同行 M 列、持续 2 s；不改回 TXT Tag100。对齐失败保留先前有效标记和行为关联，错误写入 job。
- 流式导出保持 GUI 原有 `processed` 输出规则，并要求完整 QC；MCP 不提供静默跳过 QC 的捷径。`export_preprocessed` 仅写新的 H5/CSV，任何一个目标已存在即拒绝。
- 图像包含标题和坐标轴；`read_waveform` 返回当前绘图样本（长图可能抽点），单位沿用界面标签，并不代表物理单位重新标定。复制影响**工作站**剪贴板，本地使用图片应取 PNG。
- `pause/resume/cancel` 需当前 `job_id`，仅在原 Worker 支持时生效，不强杀线程。关闭 MCP 不取消已开始任务，同一窗口重新授权可继续查原 job；重启 GUI 后不保留任务历史。
- 主窗口仍是用户当前会话，不会远程关闭程序、覆盖源文件，或替换仍打开的 AnimalStart 窗口的数据源。原始通道提取/通道修复等专家弹窗内部交互不在本次新增接口范围。

本地验证：`python -m unittest test_pipeline_mcp test_animal_peak_mcp_control test_animal_peak_mcp_bridge test_animal_peak_mcp_protocol test_alignment_log_segments test_alignment_autofill`。新增测试只使用临时合成数据，不执行工作站上的真实实验。

2026-10-27 验证：上述 60 项专项测试全部通过。另执行旧 `test_qt_core_workflows.py`，停在其不存在的 `test-remap.xlsx` 映射模拟用例；已使用 Git HEAD 中未修改的 `PreprocessWorker` 复现相同错误，没有为使测试通过而更改重映射算法或旧测试。
