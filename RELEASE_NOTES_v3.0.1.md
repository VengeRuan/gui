# v3.0.1 — 事件对齐绘图与缺失 AnimalStart 兼容

发布日期：2026-10-23。延续 MCP 版本的 `3.x` 编号，不移动或删除已有版本标签。

## 更新内容

- 逐 Trial 子图默认显示 Dot 前 2000 ms 至 AnimalStart，仍可切回 Letter 前查看范围。Dot 起点只使用 CSV M 列；显示范围不改变原有搜索和平均计算。
- 事件对齐平均图逐 Trial 标记显示范围内的其他事件时刻：Dot 灰色、LetterStart 红色、AnimalStart 紫色细线；对齐事件共用零点线。复制与导出保留这些标记。
- 完整结果保存不再生成最大峰／最小谷对齐平均和峰谷时刻分布图，仍保留相关数值数据、明细与审计记录；不删除已经保存的旧文件。
- 时间对齐日志含多个 PROGRAM_START 时间段时，弹窗选择与当前 CSV 对应的时间段，展示 PROGRAM_END 和时长供核对；取消不更改已有对齐结果。PROGRAM_END 不用于裁剪数据。
- 缺少 AnimalStart 时，不再阻止 LetterStart、Dot 平均计算和绘图；AnimalStart 对齐仅纳入具备有效时间的 Trial，无可用 Trial 时跳过对应图片，不补造事件时刻。明确无效的 Trial 仍按原规则排除。

## 验证与部署

- 本地 `test_animal_peak*.py` 回归测试：241 项通过。
- 工作站 Conda `GUI` 环境的缺失事件、事件平均、标记、复制、日志时间段及相关对齐测试：82 项通过。
- 修改后的运行代码已同步工作站 `D:\Users\Ruan\gui`，替换前已备份并核对文件哈希；未关闭用户进程或更改源数据。
- 请保存当前工作并重启整个 GUI 后使用新代码；正在运行的窗口不会自动重载模块。

MCP 使用说明见 [MCP_ANIMALSTART_GUIDE.md](MCP_ANIMALSTART_GUIDE.md)。无 MCP 的历史基线仍为 `v2.1.10-no-mcp`。
