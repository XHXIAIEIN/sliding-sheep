"""一键流水线：截图 -> 识别 -> 求解。
  py scripts/run.py            # 用现有 images/_game.png 跑 识别+求解
  py scripts/run.py --capture  # 先按窗口标题截图(需游戏在前台)，再识别+求解
前置：grid_params.json 已用 app/grid_tuner.html 标定好四角(换关卡/换分辨率才需重标)。
"""
import argparse, cv2
from paths import image_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true", help="先截图刷新 images/_game.png")
    ap.add_argument("--title", default="套住那只羊")
    args = ap.parse_args()

    if args.capture:
        print("== 1/3 截图 ==")
        from capture_window import find_window, grab
        hwnd = find_window(args.title)
        if not hwnd:
            raise SystemExit(f"找不到窗口: {args.title}")
        img, rectinfo, mode = grab(hwnd)
        out = image_path("_game.png")
        cv2.imwrite(str(out), img)
        print(f"  窗口 {rectinfo}  方式 {mode}  -> {out.relative_to(out.parents[1])}")
    else:
        print("== 1/3 截图 == 跳过(用现有 images/_game.png)")

    print("== 2/3 识别 ==")
    import detect_occupancy
    detect_occupancy.main([])

    print("== 3/3 求解 ==")
    import solve_board
    solve_board.main("board.json")


if __name__ == "__main__":
    main()
