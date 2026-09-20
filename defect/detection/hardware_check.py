#!/usr/bin/env python3
"""Read-only Linux/Windows hardware connectivity check."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DETECTION_DIR = Path(__file__).resolve().parent
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))


def access_text(path: str) -> str:
    return "read={}, write={}".format(
        os.access(path, os.R_OK), os.access(path, os.W_OK)
    )


def check_camera() -> bool:
    from LensCamera import camera as camera_tools

    cv2, _np = camera_tools.load_image_tools()
    devices = camera_tools.enumerate_cameras()
    print("\n[Camera]\n{}".format(camera_tools.format_camera_devices(devices)))
    if not devices:
        return False

    selected = camera_tools.find_camera_by_name("CamSPC")
    source = camera_tools.camera_source(selected)
    camera = camera_tools.open_camera(cv2, source)
    try:
        if not camera.isOpened():
            print("FAIL: cannot open {}".format(source))
            return False
        ok, frame = camera.read()
        if not ok or frame is None:
            print("FAIL: opened {}, but no frame was returned".format(source))
            return False
        print("OK: {} frame={}x{}".format(source, frame.shape[1], frame.shape[0]))
        return True
    finally:
        camera.release()


def check_lens() -> bool:
    print("\n[LensConnect]")
    try:
        import LensCamera as lens_camera

        count = lens_camera.count_lensconnect_devices()
        print("SDK loaded, detected device count={}".format(count))
        if count < 1:
            return False
        selected = lens_camera.open_lensconnect_usb(0, retries=1, retry_delay=0)
        lens_camera.close()
        print("OK: opened and closed LensConnect device {}".format(selected))
        return True
    except Exception as exc:
        print("FAIL: {}".format(exc))
        for path in sorted(Path("/dev").glob("hidraw*")):
            print("  {} {}".format(path, access_text(str(path))))
        return False


def check_stage() -> bool:
    print("\n[XY stage]")
    try:
        import serial.tools.list_ports
        from motion_controller import MotionController

        ports = list(serial.tools.list_ports.comports())
        usb_ports = [item for item in ports if item.vid is not None]
        for item in usb_ports:
            print("  {} - {} ({})".format(
                item.device, item.description, access_text(item.device)
            ))
        ft232 = next(
            (
                item.device
                for item in usb_ports
                if item.vid == 0x0403 and item.pid == 0x6001
            ),
            None,
        )
        if ft232 is None:
            print("FAIL: FT232 stage adapter was not found")
            return False

        all_ok = True
        for address, name in ((1, "X"), (2, "Y")):
            controller = None
            try:
                controller = MotionController(ft232, slave_address=address)
                status = controller.read_status()
                alarm = controller.read_alarm()
                print(
                    "OK: {} axis Modbus address {} status=0x{:04X} alarm=0x{:04X}".format(
                        name, address, status["raw"], alarm
                    )
                )
            except Exception as exc:
                all_ok = False
                print("FAIL: {} axis address {}: {}".format(name, address, exc))
            finally:
                if controller is not None:
                    controller.close()
        return all_ok
    except Exception as exc:
        print("FAIL: {}".format(exc))
        return False


def main() -> int:
    results = {
        "camera": check_camera(),
        "lens": check_lens(),
        "stage": check_stage(),
    }
    print("\n[Summary]")
    for name, ok in results.items():
        print("{}: {}".format(name, "OK" if ok else "FAIL"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
