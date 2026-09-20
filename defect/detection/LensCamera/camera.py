import platform
import re
from pathlib import Path


def load_image_tools(error_type=RuntimeError):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise error_type(
            "OpenCV and NumPy are required. Install them with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return cv2, np


def enumerate_windows_cameras(error_type=RuntimeError):
    """Return Windows DirectShow camera devices as index/name records."""
    if platform.system() != "Windows":
        raise error_type("camera name selection is only supported on Windows")

    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError as exc:
        raise error_type(
            "Windows camera enumeration by name requires pygrabber. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    graph = FilterGraph()
    names = graph.get_input_devices()
    return [
        {
            "index": index,
            "name": str(name),
            "source": "DirectShow",
        }
        for index, name in enumerate(names)
    ]


def format_camera_devices(devices):
    if not devices:
        return "  <none>"
    lines = []
    for device in devices:
        path_text = " {}".format(device["device"]) if device.get("device") else ""
        lines.append(
            "  [{index}] {name}{path} ({source})".format(
                path=path_text, **device
            )
        )
    return "\n".join(lines)


def enumerate_linux_cameras(error_type=RuntimeError):
    """Return Linux V4L2 camera devices using sysfs names."""
    if platform.system() != "Linux":
        raise error_type("V4L2 camera enumeration is only supported on Linux")

    def video_index(path):
        match = re.search(r"\d+$", path.name)
        return int(match.group()) if match else -1

    devices = []
    for sys_path in sorted(Path("/sys/class/video4linux").glob("video*"), key=video_index):
        index = video_index(sys_path)
        if index < 0:
            continue
        device_path = Path("/dev") / sys_path.name
        if not device_path.exists():
            continue
        try:
            name = (sys_path / "name").read_text(encoding="utf-8").strip()
        except OSError:
            name = sys_path.name
        try:
            interface_index = int(
                (sys_path / "index").read_text(encoding="ascii").strip()
            )
        except (OSError, ValueError):
            interface_index = None
        devices.append(
            {
                "index": index,
                "name": name,
                "source": "V4L2",
                "device": str(device_path),
                "interface_index": interface_index,
            }
        )
    return devices


def enumerate_cameras(error_type=RuntimeError):
    system = platform.system()
    if system == "Windows":
        return enumerate_windows_cameras(error_type)
    if system == "Linux":
        return enumerate_linux_cameras(error_type)
    raise error_type("camera enumeration is not supported on {}".format(system))


def find_windows_camera_by_name(name_substring, error_type=RuntimeError):
    """Find exactly one Windows DirectShow camera whose name contains substring."""
    query = str(name_substring).strip()
    if not query:
        raise error_type("--camera-name must not be empty")

    devices = enumerate_windows_cameras(error_type)
    query_key = query.casefold()
    matches = [
        device for device in devices if query_key in device["name"].casefold()
    ]

    if not matches:
        raise error_type(
            "no Windows camera name contains {!r}. Available cameras:\n{}".format(
                query, format_camera_devices(devices)
            )
        )

    if len(matches) > 1:
        raise error_type(
            "camera name {!r} matched more than one device. Use a more specific "
            "name. Matches:\n{}".format(query, format_camera_devices(matches))
        )

    selected = dict(matches[0])
    selected["match"] = query
    return selected


def find_camera_by_name(name_substring, error_type=RuntimeError):
    """Find exactly one camera by a case-insensitive device-name substring."""
    query = str(name_substring).strip()
    if not query:
        raise error_type("camera name must not be empty")

    devices = enumerate_cameras(error_type)
    query_key = query.casefold()
    matches = [
        device for device in devices if query_key in device["name"].casefold()
    ]
    # UVC cameras commonly expose a video node and a metadata node with the
    # same name. Interface zero is the normal capture stream.
    if platform.system() == "Linux" and len(matches) > 1:
        primary = [item for item in matches if item.get("interface_index") == 0]
        if len(primary) == 1:
            matches = primary
    if not matches:
        raise error_type(
            "no camera name contains {!r}. Available cameras:\n{}".format(
                query, format_camera_devices(devices)
            )
        )
    if len(matches) > 1:
        raise error_type(
            "camera name {!r} matched more than one device. Use a more specific "
            "name. Matches:\n{}".format(query, format_camera_devices(matches))
        )
    selected = dict(matches[0])
    selected["match"] = query
    return selected


def camera_source(selected):
    """Return the OpenCV source for a camera enumeration record."""
    return selected.get("device", selected["index"])


def open_camera(cv2, camera_index):
    system = platform.system()
    if system == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if camera.isOpened():
            return camera
        camera.release()

    if system == "Linux" and hasattr(cv2, "CAP_V4L2"):
        camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if camera.isOpened():
            return camera
        camera.release()

    return cv2.VideoCapture(camera_index)


def configure_camera(cv2, camera, args):
    if args.frame_width:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    if args.frame_height:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    if getattr(args, "disable_camera_autofocus", False) and hasattr(
        cv2, "CAP_PROP_AUTOFOCUS"
    ):
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)


def frame_size(cv2, camera):
    width = int(round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return width, height
