import threading
from pynput import keyboard


class KeyboardController:
    """Wrapper su pynput per leggere singoli tasti senza bloccare."""

    def __init__(self):
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def pop(self) -> str | None:
        with self._lock:
            return self._queue.pop(0) if self._queue else None

    def _on_press(self, key) -> None:
        try:
            if key == keyboard.Key.space:
                ch = "space"
            elif hasattr(key, "char") and key.char is not None:
                ch = key.char.lower()
            else:
                return
        except Exception:
            return
        with self._lock:
            self._queue.append(ch)