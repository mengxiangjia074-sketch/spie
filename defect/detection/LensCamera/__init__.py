"""Lens camera control utilities for LensConnect and OpenCV workflows."""

import argparse
import json
import sys

from . import paths
from .json_io import load_json_object
from .lens_config import (
    DEFAULT_JSON_FILE as _DEFAULT_JSON_FILE,
    MOTOR_ORDER as _MOTOR_ORDER,
    load_target_config,
    parse_device_number,
    parse_position,
    read_device_number,
    read_lens_targets,
    resolve_lens_json_path,
)
from .lens_ports import LensControlError
from .lensconnect_adapter import (
    USB_OPEN_RETRIES,
    USB_OPEN_RETRY_DELAY,
    get_default_controller,
)


SCRIPT_DIR = paths.SCRIPT_DIR
ORIGINAL_CWD = paths.ORIGINAL_CWD
CONTROLLER_DIR = paths.CONTROLLER_DIR
DEFAULT_JSON_FILE = _DEFAULT_JSON_FILE
MOTOR_ORDER = _MOTOR_ORDER


def _controller():
    return get_default_controller()


def __getattr__(name):
    if name in ("CV", "DV", "LensCtrl", "UsbCtrl"):
        return getattr(_controller(), name)
    if name == "MOTORS":
        return _controller().motors
    raise AttributeError(name)


def resolve_user_path(path_text):
    return paths.resolve_path(path_text)


def resolve_json_path(json_file):
    return str(resolve_lens_json_path(json_file))


def load_config(json_path):
    return load_target_config(json_path)


def check_success(retval, action):
    return _controller().check_success(retval, action)


def count_lensconnect_devices():
    return _controller().count_devices()


def choose_lensconnect_device(requested_device, num_devices):
    return _controller().choose_device(requested_device, num_devices)


def open_lensconnect_usb(
    device_number,
    retries=USB_OPEN_RETRIES,
    retry_delay=USB_OPEN_RETRY_DELAY,
):
    return _controller().open_usb(device_number, retries, retry_delay)


def connect(device_number):
    return _controller().connect(device_number)


def close():
    return _controller().close()


def get_last_connected_device_number(default=None):
    return _controller().get_last_connected_device_number(default)


def motor_is_initialized(motor):
    return _controller().motor_is_initialized(motor)


def motor_supported(name, capabilities):
    return _controller().motor_supported(name, capabilities)


def require_motor_supported(name, capabilities):
    return _controller().require_motor_supported(name, capabilities)


def get_motor_range(name):
    return _controller().get_motor_range(name)


def read_current(name):
    return _controller().read_current(name)


def read_ranges(capabilities):
    return _controller().read_ranges(capabilities)


def read_lens_snapshot(capabilities):
    return _controller().read_lens_snapshot(capabilities)


def print_ranges(ranges, targets=None):
    return _controller().print_ranges(ranges, targets)


def validate_targets(targets, ranges):
    return _controller().validate_targets(targets, ranges)


def ensure_ready(name, motor=None, init_if_needed=True):
    if isinstance(motor, bool) and init_if_needed is True:
        init_if_needed = motor
    return _controller().ensure_ready(name, init_if_needed)


def ensure_targets_ready(targets, init_if_needed):
    return _controller().ensure_targets_ready(targets, init_if_needed)


def move_connected_motor(
    name,
    target,
    capabilities,
    settle_seconds=0,
    init_if_needed=None,
    clamp_target=False,
):
    return _controller().move_connected_motor(
        name,
        target,
        capabilities,
        settle_seconds=settle_seconds,
        init_if_needed=init_if_needed,
        clamp_target=clamp_target,
    )


def move_motor(name, target, capabilities, init_if_needed):
    return _controller().move_motor(name, target, capabilities, init_if_needed)


def apply_targets(device_number, targets, init_if_needed):
    return _controller().apply_targets(device_number, targets, init_if_needed)


def show_ranges(device_number):
    return _controller().show_ranges(device_number)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m LensCamera",
        description="Move LensConnect zoom, focus, and iris positions from a JSON file."
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=DEFAULT_JSON_FILE,
        help="Path to JSON file containing zoom/focus/iris.",
    )
    parser.add_argument(
        "--device",
        type=parse_device_number,
        default=None,
        help="LensConnect device number. Use auto to select the first detected device.",
    )
    parser.add_argument(
        "--no-init",
        action="store_true",
        help="Do not initialize a motor automatically when the lens reports it is needed.",
    )
    parser.add_argument(
        "--ranges-only",
        action="store_true",
        help="Only read and print supported zoom/focus/iris ranges; do not move.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    json_path = resolve_json_path(args.json_file)

    try:
        if args.ranges_only:
            device_number = args.device
            if device_number is None:
                try:
                    device_number = read_device_number(load_json_object(json_path))
                except OSError:
                    device_number = None
            show_ranges(device_number)
            return 0

        config_device, targets = load_config(json_path)
        device_number = args.device
        if device_number is None:
            device_number = config_device
        apply_targets(device_number, targets, init_if_needed=not args.no_init)
    except (OSError, json.JSONDecodeError, ValueError, LensControlError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    return 0
