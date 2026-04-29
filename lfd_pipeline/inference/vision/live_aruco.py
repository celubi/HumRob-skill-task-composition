"""LiveAruco: stream RealSense + detect ArUco in finestra OpenCV.

Versione semplificata della classe omonima usata in
``lfd_evaluate/generalization_pick.py``: thread di acquisizione che mostra
i frame con overlay (bounding box, axes, posa in BASE) e mantiene
l'ultimo snapshot disponibile a ``snapshot()``.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# helpers SE(3)
# ---------------------------------------------------------------------------
def _to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def _rpy_deg_to_R_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    rd, pd, yd = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(rd), math.sin(rd)
    cp, sp = math.cos(pd), math.sin(pd)
    cy, sy = math.cos(yd), math.sin(yd)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    return Rz @ Ry @ Rx


def _R_to_rpy_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    R = np.asarray(R, float)
    sp = -R[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return float(roll), float(pitch), float(yaw)


# ---------------------------------------------------------------------------
# LiveAruco
# ---------------------------------------------------------------------------
class LiveAruco:
    WINDOW_NAME = "ArUco live (eye-in-hand)"

    def __init__(self,
                 arm,
                 T_ee_cam: np.ndarray,
                 marker_len_m: float,
                 dict_name: str = "DICT_6X6_50",
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30):
        self.arm = arm
        self.T_ee_cam = np.asarray(T_ee_cam, float)
        self.marker_len_m = float(marker_len_m)
        self.dict_name = dict_name
        self.width, self.height, self.fps = width, height, fps

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._latest: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._thread: Optional[threading.Thread] = None

        # imports laziosi (solo quando si avvia il live)
        import pyrealsense2 as rs  # noqa: WPS433
        import cv2                   # noqa: WPS433
        import cv2.aruco as aruco    # noqa: WPS433
        self._rs = rs
        self._cv2 = cv2
        self._aruco = aruco
        self._dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))

    # --------------------------------------------------------------- robot
    def _T_base_ee(self) -> Optional[np.ndarray]:
        code, pose = self.arm.get_position(is_radian=False)
        if code != 0 or pose is None:
            return None
        x_mm, y_mm, z_mm, roll, pitch, yaw = pose[:6]
        R = _rpy_deg_to_R_zyx(roll, pitch, yaw)
        t = np.array([x_mm, y_mm, z_mm], float) / 1000.0
        return _to_T(R, t)

    # --------------------------------------------------------------- detect
    def _detect(self, img_bgr, K, D):
        cv2 = self._cv2
        aruco = self._aruco
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if hasattr(aruco, "ArucoDetector"):
            params = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(self._dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            params = aruco.DetectorParameters_create()
            corners, ids, _ = aruco.detectMarkers(gray, self._dict, parameters=params)

        out: dict[int, np.ndarray] = {}
        annot = img_bgr.copy()
        if ids is None or len(ids) == 0:
            return out, annot

        # filtra marker troppo piccoli
        keep = []
        for i, c in enumerate(corners):
            if cv2.arcLength(c.astype(np.float32), True) / 4.0 >= 6.0:
                keep.append(i)
        if not keep:
            return out, annot
        sel_corners = [corners[i] for i in keep]
        sel_ids = ids[keep]
        aruco.drawDetectedMarkers(annot, sel_corners, sel_ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            sel_corners, self.marker_len_m, K, D)
        for i, mid in enumerate(sel_ids.ravel()):
            rvec = rvecs[i].reshape(3, 1)
            tvec = tvecs[i].reshape(3)
            R, _ = cv2.Rodrigues(rvec)
            out[int(mid)] = _to_T(R, tvec)
            cv2.drawFrameAxes(annot, K, D, rvec, tvec.reshape(3, 1),
                              self.marker_len_m * 0.5)
        return out, annot

    # --------------------------------------------------------------- loop
    def _loop(self, K, D):
        cv2 = self._cv2
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 960, 540)

        while not self._stop_evt.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())

            Tcam_by_id, annot = self._detect(img, K, D)

            T_base_ee = self._T_base_ee()
            ids_sorted = (np.array(sorted(Tcam_by_id.keys()), dtype=int)
                          if Tcam_by_id else np.array([], dtype=int))
            poses_list = []
            T_base_tag_by_id: dict[int, np.ndarray] = {}
            if T_base_ee is not None and ids_sorted.size > 0:
                T_base_cam = T_base_ee @ self.T_ee_cam
                for row, mid in enumerate(ids_sorted):
                    T_base_tag = T_base_cam @ Tcam_by_id[int(mid)]
                    T_base_tag_by_id[int(mid)] = T_base_tag
                    t_mm = T_base_tag[:3, 3] * 1000.0
                    roll, pitch, yaw = _R_to_rpy_zyx(T_base_tag[:3, :3])
                    poses_list.append([t_mm[0], t_mm[1], t_mm[2], roll, pitch, yaw])
                    txt = (f"id:{mid}  x:{t_mm[0]:.0f} y:{t_mm[1]:.0f} z:{t_mm[2]:.0f}  "
                           f"r:{roll:+.2f} p:{pitch:+.2f} y:{yaw:+.2f}")
                    ytxt = 25 + 22 * row
                    cv2.putText(annot, txt, (10, ytxt),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (10, 10, 10), 3, cv2.LINE_AA)
                    cv2.putText(annot, txt, (10, ytxt),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)

            banner = (f"LIVE  detected: {ids_sorted.size}  "
                      f"{datetime.now().strftime('%H:%M:%S')}")
            cv2.putText(annot, banner, (10, annot.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annot, banner, (10, annot.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)

            poses_arr = (np.array(poses_list, float)
                         if poses_list else np.empty((0, 6), float))
            with self._lock:
                self._latest = (ids_sorted.copy(), poses_arr.copy(), annot.copy(),
                                dict(T_base_tag_by_id))

            cv2.imshow(self.WINDOW_NAME, annot)
            cv2.waitKey(1)

        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass

    # --------------------------------------------------------------- public
    def start(self) -> None:
        rs = self._rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        prof = self._pipe.start(cfg)
        vprof = prof.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vprof.get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], float)
        D = np.array(intr.coeffs if intr.coeffs else [0, 0, 0, 0, 0], float)
        for _ in range(10):
            try:
                self._pipe.wait_for_frames(timeout_ms=1000)
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, args=(K, D), daemon=True)
        self._thread.start()
        print(f"[live] camera streaming on '{self.WINDOW_NAME}' "
              f"({self.width}x{self.height}@{self.fps}fps)")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._pipe.stop()
        except Exception:
            pass

    def snapshot(self, retries: int = 30, sleep_s: float = 0.05):
        """Restituisce (ids, poses_mm_rad, annotated_bgr, T_base_tag_by_id)."""
        for _ in range(max(1, retries)):
            with self._lock:
                if self._latest is not None:
                    return self._latest
            time.sleep(sleep_s)
        return (np.array([], dtype=int),
                np.empty((0, 6), float),
                None,
                {})

    def wait_for_marker(self, marker_id: int,
                        timeout_s: float = 30.0,
                        n_stable: int = 5) -> Optional[np.ndarray]:
        """Aspetta che ``marker_id`` venga rilevato per ``n_stable`` frame
        consecutivi e restituisce la sua media ``T_base^tag`` (4x4)."""
        deadline = time.time() + timeout_s
        Ts: list[np.ndarray] = []
        while time.time() < deadline:
            ids, _, _, T_by_id = self.snapshot()
            if int(marker_id) in T_by_id:
                Ts.append(T_by_id[int(marker_id)])
                if len(Ts) >= n_stable:
                    arr = np.stack(Ts[-n_stable:], axis=0)
                    return arr.mean(axis=0)
            else:
                Ts.clear()
            time.sleep(0.05)
        return None
