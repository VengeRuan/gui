from pywinauto import Desktop
import win32gui


MAIN_TITLE = "1024通道神经电生理信号采集与刺激系统"

KEYWORDS = [
    "已有重连任务正在执行",
    "5000ms",
    "后重连",
]


# ---------- 先用 UIA 找到嵌入主程序内部的 Controller ----------
desktop = Desktop(backend="uia")

main = desktop.window(title=MAIN_TITLE)
main.wait("exists visible ready", timeout=30)

controller = None

for ctrl in main.descendants():
    try:
        info = ctrl.element_info

        if (
            getattr(info, "name", "") == "Controller"
            and getattr(info, "control_type", "") == "Window"
        ):
            controller = ctrl
            break

    except Exception:
        pass

if controller is None:
    raise RuntimeError("未找到 Controller，请先在主程序中打开它。")

controller_hwnd = controller.element_info.handle

print("Controller HWND：", hex(controller_hwnd))
print("开始通过 Win32 API 枚举其子窗口...")
print("=" * 110)

matches = []


def inspect_hwnd(hwnd, _):
    """EnumChildWindows 的回调函数。"""

    try:
        class_name = win32gui.GetClassName(hwnd)
    except Exception:
        class_name = ""

    try:
        text = win32gui.GetWindowText(hwnd)
    except Exception:
        text = ""

    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    except Exception:
        left, top, right, bottom = 0, 0, 0, 0

    width = right - left
    height = bottom - top

    hit_keywords = [key for key in KEYWORDS if key in text]

    # 显示：
    # 1. 直接命中关键词的控件；
    # 2. 类名包含 Edit / RichEdit / TextBox 的可疑文本控件；
    # 3. 尺寸很大的可疑日志区。
    is_text_control = any(
        key.lower() in class_name.lower()
        for key in ["edit", "richedit", "textbox", "text"]
    )

    is_large_area = width > 800 and height > 100

    if hit_keywords or is_text_control or is_large_area:
        print("\n" + "-" * 110)
        print(f"HWND       : {hex(hwnd)}")
        print(f"ClassName  : {class_name!r}")
        print(f"位置       : left={left}, top={top}, right={right}, bottom={bottom}")
        print(f"尺寸       : width={width}, height={height}")
        print(f"命中关键词 : {hit_keywords}")

        if text:
            print("GetWindowText 读取内容：")
            print(repr(text[:3000]))
        else:
            print("GetWindowText 读取内容：<空>")

    if hit_keywords:
        matches.append(hwnd)

    return True


# 先检查 Controller 本身
inspect_hwnd(controller_hwnd, None)

# 再检查所有原生子窗口
win32gui.EnumChildWindows(controller_hwnd, inspect_hwnd, None)

print("\n" + "=" * 110)
print(f"搜索结束。命中关键词的窗口数量：{len(matches)}")

if matches:
    print("成功：Win32 API 可以读取接收日志。")
    print("后续可定时读取对应 HWND 的 GetWindowText()，并解析 Stim on/off。")
else:
    print("未读到目标文字。该日志区域很可能是完全自定义绘制控件。")
    print("下一步建议改为读取 TCP 通信，而不是 OCR。")