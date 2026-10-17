"""Optional start/stop push button for ``collect`` (off by default).

``rig.yaml``::

    capture:
      button: {enabled: true, pin: 7, active_low: true, long_press_s: 1.5}   # BOARD pin numbering (Jetson.GPIO)

or ``sensor-rig collect --button gpio:7``. A short press emits ``"toggle"`` (start/stop, like Enter), a long press
(held >= ``long_press_s``) emits ``"abort"``. ``ButtonEdgeDecoder`` is the pure, unit-tested press logic; ``GpioButton``
polls a pin with Jetson.GPIO (imported lazily; NOT RUN ON HARDWARE YET); ``FakeButton`` feeds levels in tests.

Wiring (active low): pin -> momentary switch -> GND, with the pin's pull-up (or an external 10 kOhm to 3.3 V; the
Orin Nano header pins have no software pull-up control in Jetson.GPIO, so use the external resistor).
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class ButtonEdgeDecoder:
    """Turns sampled levels (True = pressed) into 'toggle' / 'abort' events with debouncing."""

    def __init__(self, long_press_s: float = 1.5, debounce_s: float = 0.03):
        self.long_press_s = float(long_press_s)
        self.debounce_s = float(debounce_s)
        self._stable = False
        self._cand: bool | None = None
        self._cand_t = 0.0
        self._down_t: float | None = None
        self._long_sent = False

    def update(self, pressed: bool, t: float) -> str | None:
        pressed = bool(pressed)
        if pressed != self._stable:
            if self._cand != pressed:
                self._cand, self._cand_t = pressed, t
            elif t - self._cand_t >= self.debounce_s:
                self._stable = pressed
                self._cand = None
                if pressed:
                    self._down_t, self._long_sent = t, False
                else:
                    held = t - (self._down_t if self._down_t is not None else t)
                    self._down_t = None
                    if not self._long_sent and held < self.long_press_s:
                        return "toggle"
        else:
            self._cand = None
        if self._stable and self._down_t is not None and not self._long_sent and t - self._down_t >= self.long_press_s:
            self._long_sent = True
            return "abort"      # fires while still held, so the operator gets feedback
        return None


class Button:
    """Polls ``read_level()`` in a thread and calls ``on_event('toggle' | 'abort')``."""

    def __init__(self, on_event: Callable[[str], None], long_press_s: float = 1.5, period_s: float = 0.01):
        self.on_event = on_event
        self.decoder = ButtonEdgeDecoder(long_press_s)
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def read_level(self) -> bool:   # True = pressed
        raise NotImplementedError

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    def start(self) -> None:
        self._open()
        self._thread = threading.Thread(target=self._loop, name="button", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ev = self.decoder.update(self.read_level(), time.monotonic())
            if ev:
                self.on_event(ev)
            time.sleep(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._close()


class FakeButton(Button):
    """``press(seconds)`` holds the button for that long (in a background thread)."""

    def __init__(self, on_event, long_press_s: float = 1.5, period_s: float = 0.005):
        super().__init__(on_event, long_press_s, period_s)
        self.level = False

    def read_level(self) -> bool:
        return self.level

    def press(self, seconds: float) -> None:
        self.level = True
        time.sleep(seconds)
        self.level = False


class GpioButton(Button):  # pragma: no cover - needs a Jetson
    def __init__(self, on_event, pin: int, active_low: bool = True, long_press_s: float = 1.5):
        super().__init__(on_event, long_press_s)
        self.pin = int(pin)
        self.active_low = bool(active_low)
        self._gpio = None

    def _open(self) -> None:
        try:
            import Jetson.GPIO as GPIO
        except ImportError as e:
            raise RuntimeError("the GPIO button needs Jetson.GPIO (sudo apt install python3-jetson-gpio, or pip "
                               "install Jetson.GPIO; user in the 'gpio' group)") from e
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.IN)
        self._gpio = GPIO

    def read_level(self) -> bool:
        v = bool(self._gpio.input(self.pin))
        return (not v) if self.active_low else v

    def _close(self) -> None:
        if self._gpio is not None:
            self._gpio.cleanup(self.pin)


def button_config(rig, override: str | None) -> dict | None:
    """``{"pin", "active_low", "long_press_s"}`` from ``--button gpio:PIN`` or ``rig.capture.button``; None = off."""
    cap = rig.capture if isinstance(getattr(rig, "capture", None), dict) else {}
    cfg = dict(cap.get("button") or {})
    if override:
        if not override.startswith("gpio:"):
            raise ValueError(f"--button must look like gpio:PIN, got {override!r}")
        cfg.update(enabled=True, pin=int(override.split(":", 1)[1]))
    if not cfg.get("enabled"):
        return None
    if cfg.get("pin") is None:
        raise ValueError("capture.button.enabled is true but no pin is set")
    return {"pin": int(cfg["pin"]), "active_low": bool(cfg.get("active_low", True)),
            "long_press_s": float(cfg.get("long_press_s", 1.5))}
