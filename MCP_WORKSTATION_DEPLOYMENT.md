# 工作站部署记录 — 2026-10-22

## 实际安装

- 工作站：192.168.110.8，账号 ruanwenjun。
- 项目：`D:\Users\Ruan\gui`。
- 按用户指定，MCP SDK 直接安装在 Conda `GUI` 环境：`D:\software\Anaconda\envs\GUI\python.exe`，MCP 版本 1.26.0。
- 启动脚本 `start_animal_peak_mcp.cmd` 默认使用以上解释器；不使用 `.venv-mcp`。此前创建的隔离验证环境保留但不参与正式启动。
- 初次同步核验 75 个文件，其中 73 个需要更新；后续按用户环境选择更新启动脚本和文档。没有删除数据文件，也没有强制关闭原有工作窗口。
- 被替换的原文件备份：`D:\Users\Ruan\gui\.mcp_deployments\deploy_20261022035552202\backup`。同级 `manifest.json` 保留初始同步前后 SHA256；`deployment_result.json` 记录最终版本和核验结果。

## 实际通过的检查

- SSH 调用正式目录启动脚本 `--diagnose` 成功。
- 正式目录主界面使用 GUI 环境通过隐藏的启动与关闭检查；审计使用独立测试数据库，禁用测试进程的云同步。
- 本机标准 MCP 客户端 → SSH → 正式目录启动脚本 → GUI 环境 Server → 合成数据 Qt 窗口：12 工具发现、LetterStart 参数修改、后台平均计算、任务查询、获取包含标题 PNG 均成功。
- 工作站 GUI 环境执行 21 项 MCP 控制/安全/协议测试，全部通过。
- 所有专用验证进程均已结束；未用真实 EEG 数据进行这些测试。

## 现在怎样使用

1. 下次正常启动更新后的 Qt 程序，完成数据加载和对齐，打开 AnimalStart 测试窗口。
2. 勾选“允许 MCP 控制”，并设置允许的导出目录。
3. 桌面客户端按照 `mcp_workstation.example.toml` 配置 SSH MCP 并重新加载。此次部署没有自动修改 Codex 全局设置。

MCP 使用 stdio 按需启动，不是常驻 Windows 服务；测试客户端退出后，对应 MCP Server 正常退出。GUI 未启用控制时，`list_sessions` 返回空列表是正常行为。完整说明见 `MCP_ANIMALSTART_GUIDE.md`。
