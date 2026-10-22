# v3.0.0 — AnimalStart 测试窗口 MCP v1

## 版本边界

`v2.1.10-no-mcp`（`b1e4586`）及此前发布版本**没有 MCP 控制能力**。本功能在独立开发分支 `feature/animalstart-mcp-v1` 上新增。可用基线标签取得完全未集成MCP的源码。

新增MCP的版本从`v3.0.0`开始使用3.x编号，后续更新继续递增。此前误用的`v2.1.11-mcp-v1`是早期MCP标签，保留作历史追溯，不再作为最新版本号，也不属于无MCP基线。

## 范围与对接清单

本版控制“测试：AnimalStart 峰谷对齐平均”按钮打开的**整个测试窗口**，不控制主程序其他页面、不导入新的H5、不改变第三页已完成的对齐范围。GUI与MCP复用原有算法/后台Worker；没有任意Python/shell执行接口。

| 界面功能 | MCP 接口/动作 | 约束 |
|---|---|---|
| 会话/文件/参数/状态 | list_sessions、get_state、describe_controls | 明确session_id；不自动选择别的窗口 |
| 搜索/截取/定位/去毛刺/趋势/Tag/Trial上限/额外处理 | set_parameters(extraction)、execute(extract) | 已对齐通道只读；丢弃未保存结果需确认 |
| 峰/谷平均刷新、保留点位调整截取窗 | execute(refresh_average) | 不重新自动找点；其他待修改提取参数须恢复或重提取 |
| 峰谷与旧AnimalStart对照平均名单 | set_trial_selection(peaks) | 原Trial编号；null=全部，[]=全不选 |
| 逐Trial翻页、每页1/4图、曲线显示、分支 | navigate、set_parameters(view) | detail_page从0开始；trial_id自动跳到所在页 |
| 人工点位的数值预览、应用、恢复 | navigate(trial_id/branch)、set_parameters(manual)、execute(apply_manual/restore_manual/clear_manual_preview) | 输入时间等价于图上点选；相对AnimalStart；预览不计入平均 |
| 显示/放大最新平均 | set_parameters(view)、execute(show_average) | 不重新检测、不丢弃人工点位 |
| 完整时长通道二次滤波/恢复 | set_parameters(filter)、execute(filter/restore_filter) | confirm=true；不覆盖源H5；保留点位 |
| 判断窗相似度、阈值、参考、排名、热图 | set_parameters(similarity)、execute(compute_similarity)、navigate(reference_trial_id/rank_trial_id/similarity_tab) | 需趋势提取；相似度不筛选平均 |
| 全部明细、排除原因 | read_table(records)、read_records | 分页；不隐藏无效Trial |
| Letter/Dot/Animal事件平均及基线/响应总长度 | set_parameters(event)、execute(refresh_event) | Dot基线相对Dot，其余相对Letter；duration不能与显式起止同时设置 |
| 事件平均名单 | set_trial_selection(event) | 三方式共用；与峰谷独立；先刷新当前事件组 |
| 统一/事件独立Y轴、Letter前范围、事件图宽 | set_parameters(view)、execute(apply_global_y/auto_global_y/apply_event_y/auto_event_y/apply_letter_range/apply_width/auto_width) | 只改变显示与导出，不裁剪存储数据 |
| 页签、通道/Tag、参数区收起/展开 | navigate、set_parameters(view) | 不增加计算通道，不绕过资格条件 |
| 缩放/平移/Home等价操作 | execute(reset_view，带plot/index/x_range/y_range) | 数值范围代替鼠标动作；固定Y轴不能被缩放改变 |
| 获取任何图/当前详情整页/热图 | get_plot_image | 标题、坐标轴、曲线完整PNG |
| 读取绘图样本 | read_waveform | 每次最多4096点，保留显示单位约定 |
| 复制图片、保存当前PNG | execute(copy_plot/export_plot) | 工作站剪贴板；PNG仅写授权根目录，不覆盖文件 |
| 完整导出及独立Trial PNG选项 | set_parameters(view.individual_png)、execute(export) | confirm=true；新结果目录；保留已完成MCP操作审计 |
| 进度/取消/关闭 | get_jobs、execute(cancel/close) | 取消需当前job_id；丢弃未保存修改需确认 |

尚未打开测试窗口时没有会话。用户先在第三页完成对齐，打开测试窗口；MCP不会悄悄另开数据源或代替主程序完成对齐。

## 工作站启用

1. 更新整套项目并保存结果、重启Qt程序，不热替换正在计算的模块。
2. 工作站`conda activate GUI`后，在项目目录运行`python -m pip install -r requirements-mcp.txt`。按工作站使用者选择，SDK直接安装在GUI环境；关闭MCP时不需要。
3. 测试窗口中设置专用“MCP导出根目录”，勾选“允许MCP控制”，记录会话编号。
4. 仅监听`127.0.0.1`随机端口。凭据位于当前账号`%LOCALAPPDATA%\SD_BIN_Analysis\mcp_sessions`，ACL仅当前账号、SYSTEM和Administrators。不要上传该目录。
5. SSH和GUI必须使用同一Windows账号；不同账号看不到相同注册目录。不要把SSH密码写入脚本。

`start_animal_peak_mcp.cmd`默认使用`D:\software\Anaconda\envs\GUI\python.exe`；也可用`ANIMAL_PEAK_MCP_PYTHON`指定解释器，不依赖非交互SSH的conda激活。

## 本机桌面客户端

参考`mcp_workstation.example.toml`配置本地STDIO MCP：本机启动ssh.exe，远端运行启动脚本。示例中的`YOUR_SSH_ALIAS`必须替换为已配置公钥的SSH别名。保持`BatchMode=yes`和`StrictHostKeyChecking=yes`；首次通过可信方式核对主机指纹，不能关闭校验。

Windows OpenSSH需要`PROGRAMDATA`。MCP客户端可能不继承该变量，表现为普通SSH正常、MCP启动却立即退出；配置示例显式传入`C:\ProgramData`，如本机路径不同请修改。工作站合成数据验证已实际复现并确认此项修复。

没有自动修改用户全局Codex配置。客户端重新加载后须实测工具调用；终端能SSH不等于当前聊天已加载MCP工具。

官方配置说明：<https://learn.chatgpt.com/docs/extend/mcp?surface=cli>。

## 调用顺序

1. list_sessions → get_state → describe_controls，确认会话/文件及参数选项。
2. 每次写操作提供最新expected_revision。人工修改/切换组/任务完成会使旧版本失效；stale_revision时重新读取，不盲目重试写操作。
3. set_parameters仅更新输入；执行对应apply/refresh才应用。事件示例：group=event，values={"alignment":"dot","duration_ms":2000}，随后execute(refresh_event)。
4. 长操作返回job_id，用get_jobs轮询至completed/failed/cancelled，不重新发起相同任务。
5. get_plot_image返回图像块；read_table/read_records/read_waveform分页读取结果。
6. execute(export, options={"directory":"reports","confirm":true})在授权根目录中创建独立目录。未刷新输入不作为已计算参数导出。

SSH/MCP服务断开不会取消Qt进程中的任务。重连可查询原job_id；关闭GUI或退出工作站账号会失去内存状态，**不支持跨GUI重启恢复任务**。同一窗口关闭再开启MCP会更换凭据，但保留最近任务历史。

## 安全与验证边界

- 默认关闭，仅开启时监听和创建私有会话文件；关闭/接受关闭窗口后删除会话文件。
- 请求认证、限长、类型/范围、版本、忙碌检查均先于执行；Qt控件仅在GUI主线程操作。
- 导出限定GUI授权专用根目录，拒绝路径穿越；调用记录不含SSH密码/私钥/会话令牌。
- 区分当前输入与已应用结果；MCP任务失败返回job.error，不弹出阻塞确认框。
- 不自动提升权限，不操作任意应用，不改变主程序对齐资格；AI选点不等同于科学验证。
- 自动测试使用合成数据；真实工作站/桌面客户端连接和真实数据准确性应单独验证，不能仅凭单元测试宣称已部署。

开发验证：安装可选依赖后运行`python -m unittest test_animal_peak_mcp_control test_animal_peak_mcp_bridge test_animal_peak_mcp_protocol`，再运行原有测试。
