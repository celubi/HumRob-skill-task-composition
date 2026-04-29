"""Visualizzazione live della camera RealSense montata sul robot.

Mostra in una finestra OpenCV lo stream a colori della camera. Premere
``q`` o ``ESC`` per uscire.

Esempio
-------
    python3 live_camera.py
    python3 live_camera.py --width 1280 --height 720 --fps 30
    python3 live_camera.py --depth      # mostra anche la depth colorata
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import numpy as np


WINDOW_COLOR = "RealSense - Color (eye-in-hand)"
WINDOW_DEPTH = "RealSense - Depth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--depth", action="store_true",
                   help="abilita anche la visualizzazione della depth")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import pyrealsense2 as rs
        import cv2
    except ImportError as e:
        print(f"[errore] dipendenza mancante: {e}", file=sys.stderr)
        return 1

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height,
                      rs.format.bgr8, args.fps)
    if args.depth:
        cfg.enable_stream(rs.stream.depth, args.width, args.height,
                          rs.format.z16, args.fps)

    try:
        pipe.start(cfg)
    except Exception as e:
        print(f"[errore] impossibile avviare la camera RealSense: {e}",
              file=sys.stderr)
        return 2

    print(f"[live] streaming {args.width}x{args.height}@{args.fps}fps "
          f"(premi 'q' o ESC per uscire)")

    cv2.namedWindow(WINDOW_COLOR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_COLOR, 960, 540)
    if args.depth:
        cv2.namedWindow(WINDOW_DEPTH, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_DEPTH, 960, 540)

    colorizer = rs.colorizer() if args.depth else None

    try:
        while True:
            try:
                frames = pipe.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue

            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            banner = (f"LIVE  {args.width}x{args.height}@{args.fps}fps  "
                      f"{datetime.now().strftime('%H:%M:%S')}")
            cv2.putText(img, banner, (10, img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, banner, (10, img.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_COLOR, img)

            if args.depth:
                depth = frames.get_depth_frame()
                if depth:
                    dimg = np.asanyarray(colorizer.colorize(depth).get_data())
                    cv2.imshow(WINDOW_DEPTH, dimg)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
