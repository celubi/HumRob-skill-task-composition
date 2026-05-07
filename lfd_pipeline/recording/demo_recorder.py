import threading
import time
import csv
import re
from xarm.wrapper import XArmAPI
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def next_demo_index(task_dir: Path, task: str) -> int:
    """Ritorna il prossimo indice progressivo per i file <task>_NN.csv"""
    if not task_dir.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(task)}_(\d+)\.csv$")
    max_idx = 0
    for f in task_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """RPY (rad) -> quaternione (qx, qy, qz, qw)."""
    q = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])

class DemoRecorder:
    def __init__(self,
                 arm: XArmAPI,
                 task: str, 
                 out_dir: Path,
                 rate_hz: float,
                 zfill: int):
        
        self.arm = arm
        self.task = task
        self.task_dir = out_dir / task
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.zfill = zfill

        self._recording = threading.Event()
        self._stop = threading.Event()
        self._buffer: list[tuple] = []
        self._t0: float | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # --- public API ---
    def start_thread(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def is_recording(self) -> bool:
        return self._recording.is_set()

    def sample_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def toggle(self) -> str:
        """Start o stop registrazione. Ritorna un messaggio descrittivo."""
        if self._recording.is_set():
            return self._stop_and_save()
        return self._start()

    def discard(self) -> str:
        with self._lock:
            self._recording.clear()
            n = len(self._buffer)
            self._buffer.clear()
            self._t0 = None
        return f"Buffer scartato ({n} campioni)."

    # --- internal ---
    def _start(self) -> str:
        with self._lock:
            self._buffer.clear()
            self._t0 = None
        self._recording.set()
        return "REC ●  registrazione avviata"

    def _stop_and_save(self) -> str:
        self._recording.clear()
        with self._lock:
            samples = list(self._buffer)
            self._buffer.clear()
            self._t0 = None
        if not samples:
            return "Nessun campione registrato; niente da salvare."
        idx = next_demo_index(self.task_dir, self.task)
        fname = f"{self.task}_{str(idx).zfill(self.zfill)}.csv"
        fpath = self.task_dir / fname
        with open(fpath, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw", "gripper"])
            for row in samples:
                writer.writerow(row)
        return f"Salvato {fpath} ({len(samples)} campioni)."

    def _read_pose(self):
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0 or pose is None:
            return None
        x_mm, y_mm, z_mm, roll, pitch, yaw = pose[:6]
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        return x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, qx, qy, qz, qw

    def _read_gripper(self) -> float:
        try:
            code, pos = self.arm.get_gripper_position()
            if code == 0 and pos is not None:
                return float(pos)
        except Exception:
            pass
        return 0.0

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(min(self.dt, next_t - now))
                continue
            next_t += self.dt
            if not self._recording.is_set():
                # tieni il timer allineato in caso di lunghi periodi di idle
                if now - next_t > 1.0:
                    next_t = now + self.dt
                continue
            pose = self._read_pose()
            if pose is None:
                continue
            grip = self._read_gripper()
            with self._lock:
                if self._t0 is None:
                    self._t0 = now
                t_rel = now - self._t0
                self._buffer.append((f"{t_rel:.9f}", *pose, grip))
