from pywinauto import Desktop

MAIN_TITLE = "1024通道神经电生理信号采集与刺激系统"

desktop = Desktop(backend="uia")

# 连接顶层主程序
main = desktop.window(title=MAIN_TITLE)
main.wait("exists visible ready", timeout=30)

print("已连接主窗口：", main.window_text())
print("主窗口句柄：", hex(main.element_info.handle))
print("\n开始搜索主窗口内部名称包含 Controller 的子控件...\n")

found = []

for i, ctrl in enumerate(main.descendants()):
    try:
        info = ctrl.element_info

        name = getattr(info, "name", "") or ""
        class_name = getattr(info, "class_name", "") or ""
        control_type = getattr(info, "control_type", "") or ""

        # Controller 标题通常会作为 UIA 的 Name 属性
        if "Controller" in name or "Controller" in class_name:
            rect = ctrl.rectangle()

            print("=" * 100)
            print(f"索引        : {i}")
            print(f"Name        : {name!r}")
            print(f"ControlType : {control_type!r}")
            print(f"ClassName   : {class_name!r}")
            print(f"AutomationId: {getattr(info, 'automation_id', '')!r}")
            print(
                f"位置        : left={rect.left}, top={rect.top}, "
                f"right={rect.right}, bottom={rect.bottom}"
            )
            print(f"Handle      : {hex(getattr(info, 'handle', 0))}")

            found.append(ctrl)

    except Exception as e:
        print(f"第 {i} 个控件读取失败：{e}")

print("\n" + "=" * 100)
print(f"共找到 {len(found)} 个可能的 Controller 控件。")