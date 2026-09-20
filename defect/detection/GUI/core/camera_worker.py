"""Background OpenCV camera owner for the interactive camera control page."""

from __future__ import annotations

import ctypes
import math
import platform
import queue
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from LensCamera import camera as camera_tools


PREVIEW_MAX_SIZE = (1600, 900)
IDLE_WAIT_SECONDS = 0.2
FAILED_READ_LIMIT = 20
PROPERTY_READBACK_DELAY_SECONDS = 0.08

# One logical control can have several OpenCV aliases. DirectShow commonly
# exposes white-balance temperature through WHITE_BALANCE_BLUE_U rather than
# WB_TEMPERATURE, so the worker selects the first usable alias at connection.
PROPERTY_ALIASES = {
    "auto_exposure": ("CAP_PROP_AUTO_EXPOSURE",),
    "exposure": ("CAP_PROP_EXPOSURE",),
    # CamSPC exposes its auto-exposure target through the brightness control.
    "exposure_target": ("CAP_PROP_BRIGHTNESS",),
    "auto_white_balance": ("CAP_PROP_AUTO_WB",),
    "white_balance": (
        "CAP_PROP_WB_TEMPERATURE",
        "CAP_PROP_WHITE_BALANCE_BLUE_U",
    ),
    "white_balance_red": ("CAP_PROP_WHITE_BALANCE_RED_V",),
    "hue": ("CAP_PROP_HUE",),
    "saturation": ("CAP_PROP_SATURATION",),
    "contrast": ("CAP_PROP_CONTRAST",),
    "gamma": ("CAP_PROP_GAMMA",),
}

AUTO_PROPERTIES = frozenset(("auto_exposure", "auto_white_balance"))
NEGATIVE_VALUE_PROPERTIES = frozenset(("exposure",))


def property_is_supported(name: str, value: float) -> bool:
    if not math.isfinite(value):
        return False
    if name in AUTO_PROPERTIES:
        return True
    if name in NEGATIVE_VALUE_PROPERTIES:
        return value >= -1000
    return value >= 0


def exposure_milliseconds(value: float) -> float:
    """DirectShow exposure values are normally log2(seconds)."""
    try:
        result = 1000.0 * (2.0 ** float(value))
    except (OverflowError, TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def exposure_backend_value(milliseconds: float) -> float:
    """Convert an editable exposure time in milliseconds to log2(seconds)."""
    try:
        milliseconds = float(milliseconds)
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            return math.nan
        return math.log2(milliseconds / 1000.0)
    except (TypeError, ValueError):
        return math.nan


class CameraWorker(QThread):
    devices_ready = Signal(bool, list, str)
    connected = Signal(dict)
    disconnected = Signal(str)
    failed = Signal(str)
    frame_ready = Signal(QImage)
    stats_updated = Signal(float, int, int)
    properties_updated = Signal(dict)
    property_updated = Signal(str, float, bool, str)
    resolution_updated = Signal(int, int)
    photo_saved = Signal(str)
    recording_changed = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._quitting = False
        self._camera = None
        self._writer = None
        self._recording_path: Path | None = None
        self._latest_frame = None
        self._property_ids: dict[str, int] = {}
        self._property_defaults: dict[str, float] = {}
        self._preview_size = (640, 480)
        self._measured_fps = 0.0
        self._cv2 = None
        self._np = None

    # ---- GUI request API -------------------------------------------------

    def scan(self) -> None:
        self._put("scan")

    def connect_device(
        self, index: int | str, width: int, height: int, pixel_format: str
    ) -> None:
        source = index
        if isinstance(source, str):
            source = source.strip()
            if source.isdecimal():
                source = int(source)
        else:
            source = int(source)
        self._put(
            "connect",
            index=source,
            width=int(width),
            height=int(height),
            pixel_format=str(pixel_format),
        )

    def disconnect_device(self) -> None:
        self._put("disconnect")

    def set_resolution(self, width: int, height: int) -> None:
        self._put("resolution", width=int(width), height=int(height))

    def set_property(self, name: str, value: float) -> None:
        self._put("property", name=str(name), value=float(value))

    def reset_properties(self, names: list[str] | tuple[str, ...]) -> None:
        self._put("reset_properties", names=tuple(names))

    def white_balance_once(self) -> None:
        self._put("white_balance_once")

    def take_photo(self, path: Path | str, width: int, height: int) -> None:
        self._put(
            "photo",
            path=str(path),
            width=int(width),
            height=int(height),
        )

    def start_recording(self, path: Path | str) -> None:
        self._put("start_recording", path=str(path))

    def stop_recording(self) -> None:
        self._put("stop_recording")

    def shutdown(self) -> None:
        self._quitting = True
        self._queue.put({"kind": "shutdown"})

    def _put(self, kind: str, **payload: Any) -> None:
        self._queue.put({"kind": kind, **payload})

    # ---- thread loop -----------------------------------------------------

    def run(self) -> None:
        try:
            self._cv2, self._np = camera_tools.load_image_tools(RuntimeError)
        except Exception as exc:
            self.failed.emit(f"无法加载相机组件: {exc}")
            return

        com_initialized = self._initialize_com()
        frame_count = 0
        stats_started = time.monotonic()
        failed_reads = 0
        try:
            while not self._quitting:
                self._process_commands(block=self._camera is None)
                if self._quitting:
                    break
                if self._camera is None:
                    continue

                ok, frame = self._camera.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= FAILED_READ_LIMIT:
                        self.failed.emit("相机连续读取失败，连接已断开")
                        self._release_camera("读取失败")
                    continue

                failed_reads = 0
                self._latest_frame = frame
                if self._writer is not None:
                    self._writer.write(frame)
                self.frame_ready.emit(self._to_preview_image(frame))

                frame_count += 1
                now = time.monotonic()
                elapsed = now - stats_started
                if elapsed >= 1.0:
                    self._measured_fps = frame_count / elapsed
                    height, width = frame.shape[:2]
                    self.stats_updated.emit(self._measured_fps, width, height)
                    frame_count = 0
                    stats_started = now
        finally:
            self._release_camera(None)
            if com_initialized:
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass

    def _process_commands(self, *, block: bool) -> None:
        try:
            item = self._queue.get(timeout=IDLE_WAIT_SECONDS) if block else self._queue.get_nowait()
        except queue.Empty:
            return

        while item is not None:
            kind = item["kind"]
            if kind == "shutdown":
                self._quitting = True
                return
            handlers = {
                "scan": self._do_scan,
                "connect": self._do_connect,
                "disconnect": self._do_disconnect,
                "resolution": self._do_resolution,
                "property": self._do_property,
                "reset_properties": self._do_reset_properties,
                "white_balance_once": self._do_white_balance_once,
                "photo": self._do_photo,
                "start_recording": self._do_start_recording,
                "stop_recording": self._do_stop_recording,
            }
            try:
                handlers[kind](item)
            except Exception as exc:
                self.failed.emit(f"相机操作失败: {exc}")
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                item = None

    @staticmethod
    def _initialize_com() -> bool:
        if platform.system() != "Windows":
            return False
        try:
            result = ctypes.windll.ole32.CoInitialize(None)
            return result in (0, 1)
        except Exception:
            return False

    # ---- device operations ----------------------------------------------

    def _do_scan(self, _item: dict[str, Any]) -> None:
        try:
            devices = camera_tools.enumerate_cameras(RuntimeError)
            self.devices_ready.emit(
                True,
                devices,
                f"检测到 {len(devices)} 台相机",
            )
        except Exception as exc:
            self.devices_ready.emit(False, [], f"扫描失败: {exc}")

    def _do_connect(self, item: dict[str, Any]) -> None:
        self._release_camera(None)
        source = item["index"]
        camera = camera_tools.open_camera(self._cv2, source)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"无法打开相机 {source}")
        self._camera = camera
        try:
            self._set_pixel_format(item["pixel_format"])
            self._set_capture_size(item["width"], item["height"])
            self._discard_frames(4)

            width, height = camera_tools.frame_size(self._cv2, camera)
            self._preview_size = (width, height)
            properties = self._read_properties()
            fps = float(camera.get(self._cv2.CAP_PROP_FPS))
            backend = ""
            try:
                backend = camera.getBackendName()
            except Exception:
                pass
            self.connected.emit(
                {
                    "index": source,
                    "width": width,
                    "height": height,
                    "fps": fps if math.isfinite(fps) and fps > 0 else None,
                    "backend": backend,
                    "pixel_format": item["pixel_format"],
                    "properties": properties,
                }
            )
        except Exception:
            self._release_camera(None)
            raise

    def _do_disconnect(self, _item: dict[str, Any]) -> None:
        self._release_camera("已断开相机")

    def _release_camera(self, message: str | None) -> None:
        self._stop_writer()
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
        was_connected = self._camera is not None
        self._camera = None
        self._latest_frame = None
        self._property_ids = {}
        self._property_defaults = {}
        if message is not None and was_connected:
            self.disconnected.emit(message)

    def _require_camera(self):
        if self._camera is None:
            raise RuntimeError("相机未连接")
        return self._camera

    def _set_pixel_format(self, pixel_format: str) -> None:
        camera = self._require_camera()
        code = {
            "MJPG": "MJPG",
            "YUY2": "YUY2",
        }.get(pixel_format)
        if code:
            camera.set(
                self._cv2.CAP_PROP_FOURCC,
                self._cv2.VideoWriter_fourcc(*code),
            )

    def _set_capture_size(self, width: int, height: int) -> tuple[int, int]:
        camera = self._require_camera()
        camera.set(self._cv2.CAP_PROP_FRAME_WIDTH, int(width))
        camera.set(self._cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        actual = camera_tools.frame_size(self._cv2, camera)
        self._preview_size = actual
        return actual

    def _do_resolution(self, item: dict[str, Any]) -> None:
        if self._writer is not None:
            raise RuntimeError("录像期间不能切换预览分辨率")
        actual = self._set_capture_size(item["width"], item["height"])
        self._discard_frames(4)
        self.resolution_updated.emit(*actual)

    def _discard_frames(self, count: int) -> None:
        camera = self._require_camera()
        for _ in range(count):
            camera.read()

    # ---- properties -----------------------------------------------------

    def _read_properties(self) -> dict[str, dict[str, Any]]:
        camera = self._require_camera()
        result: dict[str, dict[str, Any]] = {}
        self._property_ids = {}
        self._property_defaults = {}
        for name, aliases in PROPERTY_ALIASES.items():
            candidates = []
            for alias in aliases:
                property_id = getattr(self._cv2, alias, None)
                if property_id is None:
                    continue
                value = float(camera.get(property_id))
                candidates.append((property_id, alias, value))
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if property_is_supported(name, candidate[2])
                ),
                candidates[0] if candidates else None,
            )
            supported = bool(
                selected is not None
                and property_is_supported(name, selected[2])
            )
            value = selected[2] if selected is not None else math.nan
            result[name] = {
                "supported": supported,
                "value": value if math.isfinite(value) else None,
                "backend_property": selected[1] if selected is not None else None,
            }
            if selected is not None:
                self._property_ids[name] = selected[0]
            if supported:
                self._property_defaults[name] = value

        # Some DirectShow drivers report AUTO_EXPOSURE=-1 even though setting
        # the property works. Keep the toggle available and verify on write.
        if "auto_exposure" in self._property_ids:
            result["auto_exposure"]["supported"] = True
        self.properties_updated.emit(result)
        return result

    def _auto_backend_value(self, name: str, enabled: bool) -> float:
        if name == "auto_exposure":
            return 0.75 if enabled else 0.25
        return 1.0 if enabled else 0.0

    def _set_property_value(self, name: str, requested: float) -> tuple[float, bool, str]:
        camera = self._require_camera()
        property_id = self._property_ids.get(name)
        if property_id is None:
            return requested, False, "当前驱动没有提供此属性"
        backend_value = (
            self._auto_backend_value(name, requested >= 0.5)
            if name in AUTO_PROPERTIES
            else requested
        )
        ok = bool(camera.set(property_id, float(backend_value)))
        if ok and name in {"exposure", "exposure_target"}:
            # CamSPC applies these controls asynchronously; an immediate get()
            # can return the previous value and overwrite the GUI incorrectly.
            time.sleep(PROPERTY_READBACK_DELAY_SECONDS)
        actual = float(camera.get(property_id))
        if not math.isfinite(actual):
            actual = backend_value
        if name in AUTO_PROPERTIES:
            actual = 1.0 if requested >= 0.5 else 0.0
        message = "已应用" if ok else "相机驱动拒绝了该值，请选择其他值"
        return actual, ok, message

    def _do_property(self, item: dict[str, Any]) -> None:
        name = item["name"]
        actual, ok, message = self._set_property_value(name, item["value"])
        self.property_updated.emit(name, actual, ok, message)

    def _do_reset_properties(self, item: dict[str, Any]) -> None:
        for name in item["names"]:
            if name not in self._property_defaults:
                continue
            actual, ok, message = self._set_property_value(
                name, self._property_defaults[name]
            )
            self.property_updated.emit(name, actual, ok, message)

    def _do_white_balance_once(self, _item: dict[str, Any]) -> None:
        if "auto_white_balance" not in self._property_ids:
            raise RuntimeError("当前驱动不支持自动白平衡")
        self._set_property_value("auto_white_balance", 1.0)
        camera = self._require_camera()
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            camera.read()
        self._set_property_value("auto_white_balance", 0.0)
        self._read_properties()

    # ---- capture --------------------------------------------------------

    def _do_photo(self, item: dict[str, Any]) -> None:
        camera = self._require_camera()
        if self._writer is not None:
            raise RuntimeError("请先停止录像再拍摄照片")
        previous = camera_tools.frame_size(self._cv2, camera)
        requested = (item["width"], item["height"])
        changed = requested != previous
        frame = None
        try:
            if changed:
                self._set_capture_size(*requested)
                self._discard_frames(4)
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("相机没有返回照片")
            path = Path(item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = self._cv2.imencode(path.suffix or ".png", frame)
            if not ok:
                raise RuntimeError("OpenCV 无法编码照片")
            encoded.tofile(str(path))
            self.photo_saved.emit(str(path))
        finally:
            if changed:
                self._set_capture_size(*previous)
                self._discard_frames(4)
                self.resolution_updated.emit(*previous)

    def _do_start_recording(self, item: dict[str, Any]) -> None:
        camera = self._require_camera()
        if self._writer is not None:
            return
        width, height = camera_tools.frame_size(self._cv2, camera)
        fps = float(camera.get(self._cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 1:
            fps = self._measured_fps if self._measured_fps > 1 else 30.0
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = self._cv2.VideoWriter(
            str(path),
            self._cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法创建录像文件 {path}")
        self._writer = writer
        self._recording_path = path
        self.recording_changed.emit(True, str(path))

    def _do_stop_recording(self, _item: dict[str, Any]) -> None:
        path = str(self._recording_path) if self._recording_path else ""
        self._stop_writer()
        self.recording_changed.emit(False, path)

    def _stop_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None
        self._recording_path = None

    def _to_preview_image(self, frame: Any) -> QImage:
        height, width = frame.shape[:2]
        max_width, max_height = PREVIEW_MAX_SIZE
        scale = min(1.0, max_width / width, max_height / height)
        if scale < 1.0:
            frame = self._cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=self._cv2.INTER_AREA,
            )
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        return QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
