import importlib
import os
import sys
import time

from . import paths
from .lens_config import MOTOR_ORDER
from .lens_ports import LensControlError, LensControllerPort


USB_OPEN_RETRIES = 5
USB_OPEN_RETRY_DELAY = 0.5


class ControllerModules:
    def __init__(self, controller_dir=None):
        self.controller_dir = controller_dir or paths.CONTROLLER_DIR
        self.dll_dir_handle = None
        self.loaded = False
        self.CV = None
        self.DV = None
        self.LensCtrl = None
        self.UsbCtrl = None

    def load(self):
        if self.loaded:
            return self

        if not self.controller_dir.is_dir():
            raise RuntimeError(
                "LensConnect_Controller directory not found inside LensCamera"
            )

        controller_dir_text = str(self.controller_dir)
        if controller_dir_text not in sys.path:
            sys.path.insert(0, controller_dir_text)

        if hasattr(os, "add_dll_directory") and self.dll_dir_handle is None:
            self.dll_dir_handle = os.add_dll_directory(controller_dir_text)

        self.CV = importlib.import_module("ConfigVal")
        self.DV = importlib.import_module("DefVal")
        self.UsbCtrl = importlib.import_module("UsbCtrl")
        self.LensCtrl = importlib.import_module("LensCtrl")

        self.loaded = True
        return self


class LensConnectController(LensControllerPort):
    def __init__(
        self,
        modules=None,
        usb_open_retries=USB_OPEN_RETRIES,
        usb_open_retry_delay=USB_OPEN_RETRY_DELAY,
    ):
        self._modules = modules or ControllerModules()
        self.usb_open_retries = usb_open_retries
        self.usb_open_retry_delay = usb_open_retry_delay
        self.last_connected_device_number = None
        self._motors = None

    @property
    def modules(self):
        return self._modules.load()

    @property
    def CV(self):
        return self.modules.CV

    @property
    def DV(self):
        return self.modules.DV

    @property
    def LensCtrl(self):
        return self.modules.LensCtrl

    @property
    def UsbCtrl(self):
        return self.modules.UsbCtrl

    @property
    def motors(self):
        if self._motors is None:
            cv = self.CV
            lens_ctrl = self.LensCtrl
            self._motors = {
                "zoom": {
                    "name": "zoom",
                    "mask": cv.ZOOM_MASK,
                    "parameter_read": lens_ctrl.ZoomParameterReadSet,
                    "current_read": lens_ctrl.ZoomCurrentAddrReadSet,
                    "init": lens_ctrl.ZoomInit,
                    "move": lens_ctrl.ZoomMove,
                    "min_attr": "zoomMinAddr",
                    "max_attr": "zoomMaxAddr",
                    "current_attr": "zoomCurrentAddr",
                },
                "focus": {
                    "name": "focus",
                    "mask": cv.FOCUS_MASK,
                    "parameter_read": lens_ctrl.FocusParameterReadSet,
                    "current_read": lens_ctrl.FocusCurrentAddrReadSet,
                    "init": lens_ctrl.FocusInit,
                    "move": lens_ctrl.FocusMove,
                    "min_attr": "focusMinAddr",
                    "max_attr": "focusMaxAddr",
                    "current_attr": "focusCurrentAddr",
                },
                "iris": {
                    "name": "iris",
                    "mask": cv.IRIS_MASK,
                    "parameter_read": lens_ctrl.IrisParameterReadSet,
                    "current_read": lens_ctrl.IrisCurrentAddrReadSet,
                    "init": lens_ctrl.IrisInit,
                    "move": lens_ctrl.IrisMove,
                    "min_attr": "irisMinAddr",
                    "max_attr": "irisMaxAddr",
                    "current_attr": "irisCurrentAddr",
                },
            }
        return self._motors

    def motor(self, name_or_motor):
        if isinstance(name_or_motor, str):
            try:
                return self.motors[name_or_motor]
            except KeyError as exc:
                raise LensControlError("unknown lens motor: {}".format(name_or_motor)) from exc

        for motor in self.motors.values():
            if motor is name_or_motor or motor == name_or_motor:
                return motor

        raise LensControlError("unknown lens motor")

    def motor_name(self, name_or_motor):
        motor = self.motor(name_or_motor)
        return motor["name"]

    def check_success(self, retval, action):
        if retval != self.DV.RET_SUCCESS:
            raise LensControlError("{} failed: {}".format(action, retval))

    def count_devices(self):
        retval, num_devices = self.UsbCtrl.UsbGetNumDevices()
        self.check_success(retval, "count LensConnect USB devices")
        return num_devices

    def choose_device(self, requested_device, num_devices):
        if num_devices < 1:
            return requested_device

        if requested_device is None:
            return 0

        if requested_device < 0:
            raise LensControlError("LensConnect device number must not be negative")

        if requested_device < num_devices:
            return requested_device

        if num_devices == 1:
            print(
                "LensConnect device {} was requested, but the system only detected "
                "device 0; using device 0.".format(requested_device)
            )
            return 0

        raise LensControlError(
            "LensConnect device {} was requested, but the system detected {} "
            "device(s); valid device numbers are 0..{}".format(
                requested_device, num_devices, num_devices - 1
            )
        )

    def open_usb(self, device_number, retries=None, retry_delay=None):
        retries = self.usb_open_retries if retries is None else retries
        retry_delay = self.usb_open_retry_delay if retry_delay is None else retry_delay
        last_error = None
        requested_text = "auto" if device_number is None else str(device_number)

        for attempt in range(1, retries + 1):
            self.UsbCtrl.UsbClose()

            try:
                num_devices = self.count_devices()
            except LensControlError as exc:
                num_devices = 0
                last_error = str(exc)

            if num_devices > 0:
                selected_device = self.choose_device(device_number, num_devices)
                retval = self.UsbCtrl.UsbOpen(selected_device)
                if retval == self.DV.RET_SUCCESS:
                    return selected_device

                last_error = "open USB device {} failed: {}".format(
                    selected_device, retval
                )
                self.UsbCtrl.UsbClose()
            else:
                last_error = "no LensConnect USB device detected"

            if attempt < retries:
                time.sleep(retry_delay)

        raise LensControlError(
            "open LensConnect USB device {} failed after {} attempt(s). "
            "Last error: {}. Check that the LensConnect controller is plugged in, "
            "powered, not held by another program, and that the selected device "
            "number matches system enumeration.".format(
                requested_text, retries, last_error
            )
        )

    def connect(self, device_number):
        selected_device = self.open_usb(device_number)
        self.last_connected_device_number = selected_device

        try:
            retval = self.UsbCtrl.UsbSetConfig()
            self.check_success(retval, "set USB configuration")

            retval, capabilities = self.LensCtrl.CapabilitiesRead()
            self.check_success(retval, "read lens capabilities")

            retval, _status2 = self.LensCtrl.Status2ReadSet()
            self.check_success(retval, "read lens initialization status")

            for name, motor in self.motors.items():
                if capabilities & motor["mask"]:
                    retval = motor["parameter_read"]()
                    self.check_success(retval, "read {} parameters".format(name))
                    if self.motor_is_initialized(name):
                        self.read_current(name)

            return capabilities
        except Exception:
            self.UsbCtrl.UsbClose()
            self.last_connected_device_number = None
            raise

    def close(self):
        if self._modules.loaded:
            return self.UsbCtrl.UsbClose()
        return self.DV.RET_SUCCESS if self._modules.loaded else 0

    def get_last_connected_device_number(self, default=None):
        if self.last_connected_device_number is None:
            return default
        return self.last_connected_device_number

    def motor_is_initialized(self, name_or_motor):
        motor = self.motor(name_or_motor)
        return (self.LensCtrl.status2 & motor["mask"]) == self.DV.INIT_COMPLETED

    def motor_supported(self, name, capabilities):
        return bool(capabilities & self.motor(name)["mask"])

    def require_motor_supported(self, name, capabilities):
        if not self.motor_supported(name, capabilities):
            raise LensControlError("connected lens does not support {}".format(name))

    def get_motor_range(self, name):
        motor = self.motor(name)
        return {
            "min": getattr(self.LensCtrl, motor["min_attr"]),
            "max": getattr(self.LensCtrl, motor["max_attr"]),
        }

    def read_current(self, name_or_motor):
        motor = self.motor(name_or_motor)
        name = motor["name"]
        retval = motor["current_read"]()
        self.check_success(retval, "read current {} address".format(name))
        return getattr(self.LensCtrl, motor["current_attr"])

    def read_ranges(self, capabilities):
        ranges = {}
        for name in MOTOR_ORDER:
            motor = self.motor(name)
            if not self.motor_supported(name, capabilities):
                ranges[name] = {"supported": False}
                continue

            initialized = self.motor_is_initialized(name)
            ranges[name] = {
                "supported": True,
                "initialized": initialized,
                "min": getattr(self.LensCtrl, motor["min_attr"]),
                "max": getattr(self.LensCtrl, motor["max_attr"]),
                "current": self.read_current(name) if initialized else None,
            }
        return ranges

    def read_lens_snapshot(self, capabilities):
        return self.read_ranges(capabilities)

    def print_ranges(self, ranges, targets=None):
        print("Available lens ranges:")
        for name in MOTOR_ORDER:
            motor_range = ranges[name]
            if not motor_range["supported"]:
                print("  {}: not supported".format(name))
                continue

            current = motor_range["current"]
            if current is None:
                current = "not initialized"

            text = "  {}: {}..{} (current: {})".format(
                name, motor_range["min"], motor_range["max"], current
            )
            if targets and name in targets:
                text += ", target: {}".format(targets[name])
            print(text)

    def validate_targets(self, targets, ranges):
        errors = []
        for name in MOTOR_ORDER:
            if name not in targets:
                continue

            target = targets[name]
            motor_range = ranges[name]
            if not motor_range["supported"]:
                errors.append("{} is not supported by the connected lens".format(name))
                continue

            min_addr = motor_range["min"]
            max_addr = motor_range["max"]
            if target < min_addr or target > max_addr:
                errors.append(
                    "{} target {} is outside allowed range {}..{}".format(
                        name, target, min_addr, max_addr
                    )
                )

        if errors:
            raise LensControlError("invalid JSON target(s): " + "; ".join(errors))

    def ensure_ready(self, name, init_if_needed):
        if self.motor_is_initialized(name):
            self.read_current(name)
            return

        if not init_if_needed:
            raise LensControlError(
                "{} is not initialized; rerun without --no-init to initialize it".format(
                    name
                )
            )

        print("{}: initializing".format(name))
        retval = self.motor(name)["init"]()
        self.check_success(retval, "initialize {}".format(name))

    def ensure_targets_ready(self, targets, init_if_needed):
        for name in MOTOR_ORDER:
            if name in targets:
                self.ensure_ready(name, init_if_needed)

    def move_connected_motor(
        self,
        name,
        target,
        capabilities,
        settle_seconds=0,
        init_if_needed=None,
        clamp_target=False,
    ):
        self.require_motor_supported(name, capabilities)
        motor_range = self.get_motor_range(name)
        min_addr = motor_range["min"]
        max_addr = motor_range["max"]

        if clamp_target:
            target = int(max(min_addr, min(max_addr, target)))
        elif target < min_addr or target > max_addr:
            raise LensControlError(
                "{} target {} is outside allowed range {}..{}".format(
                    name, target, min_addr, max_addr
                )
            )

        if init_if_needed is not None:
            self.ensure_ready(name, init_if_needed)

        motor = self.motor(name)
        before = self.read_current(name)
        moved = before != target
        if moved:
            retval = motor["move"](target)
            self.check_success(retval, "move {} to {}".format(name, target))

        if settle_seconds > 0:
            time.sleep(settle_seconds)

        actual = self.read_current(name)
        return {
            "target": target,
            "before": before,
            "actual": actual,
            "error": actual - target,
            "moved": moved,
        }

    def move_motor(self, name, target, capabilities, init_if_needed):
        result = self.move_connected_motor(
            name,
            target,
            capabilities,
            init_if_needed=init_if_needed,
        )
        if result["moved"]:
            print("{}: {} -> {}".format(name, result["before"], result["actual"]))
        else:
            print("{}: already at {}".format(name, target))
        return result

    def apply_targets(self, device_number, targets, init_if_needed):
        capabilities = self.connect(device_number)
        try:
            ranges = self.read_ranges(capabilities)
            self.print_ranges(ranges, targets)
            self.validate_targets(targets, ranges)
            self.ensure_targets_ready(targets, init_if_needed)

            moves = {}
            for name in MOTOR_ORDER:
                if name in targets:
                    moves[name] = self.move_motor(
                        name, targets[name], capabilities, init_if_needed
                    )
            return moves
        finally:
            self.close()

    def show_ranges(self, device_number):
        capabilities = self.connect(device_number)
        try:
            self.print_ranges(self.read_ranges(capabilities))
        finally:
            self.close()


_DEFAULT_CONTROLLER = None


def get_default_controller():
    global _DEFAULT_CONTROLLER
    if _DEFAULT_CONTROLLER is None:
        _DEFAULT_CONTROLLER = LensConnectController()
    return _DEFAULT_CONTROLLER
