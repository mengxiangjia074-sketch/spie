import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

try:
    from GUI.core.lens_worker import LensWorker
except ImportError:
    LensWorker = None


@unittest.skipIf(LensWorker is None, "PySide6 or LensConnect dependencies are missing")
class LensWorkerCalibratedPositionTests(unittest.TestCase):
    def test_connect_initializes_each_supported_uninitialized_motor(self):
        worker = LensWorker()
        init_functions = {
            name: mock.Mock(return_value=0) for name in ("zoom", "focus", "iris")
        }
        controller = SimpleNamespace(
            connect=mock.Mock(return_value=0xFFFF),
            motors={name: {"init": fn} for name, fn in init_functions.items()},
        )
        worker.controller = controller
        worker._load_sdk = mock.Mock(return_value={})
        worker._supported = mock.Mock(return_value=True)
        worker._initialized = mock.Mock(side_effect=[False, True, False])
        worker._check = mock.Mock()
        snapshot = {"device": 0, "motors": {}}
        worker._read_snapshot = mock.Mock(return_value=snapshot)
        connected = []
        worker.connected.connect(connected.append)

        worker._do_connect({"index": 0})

        init_functions["zoom"].assert_called_once_with()
        init_functions["focus"].assert_not_called()
        init_functions["iris"].assert_called_once_with()
        self.assertEqual(connected, [snapshot])
        worker._connected = False
        worker.deleteLater()

    def test_calibrated_position_moves_zoom_focus_and_iris_in_order(self):
        worker = LensWorker()
        worker.controller = SimpleNamespace(
            get_motor_range=mock.Mock(return_value={"min": 0, "max": 65535}),
            validate_targets=mock.Mock(),
        )
        worker._sdk = mock.Mock(return_value="sdk")
        worker._supported = mock.Mock(return_value=True)
        worker._move_motor = mock.Mock(
            side_effect=[
                {"actual": 2800},
                {"actual": 4936},
                {"actual": 400},
            ]
        )
        status = {
            "status1": 0,
            "status2": 0,
            "positions": {"zoom": 2800, "focus": 4936, "iris": 400},
        }
        worker._collect_status = mock.Mock(return_value=status)
        finished = []
        worker.calibrated_position_finished.connect(
            lambda ok, message: finished.append((ok, message))
        )

        worker._do_calibrated_position(
            {"position": {"zoom": 2800, "focus": 4936, "iris": 400}}
        )

        self.assertEqual(
            worker._move_motor.call_args_list,
            [
                mock.call("zoom", 2800),
                mock.call("focus", 4936),
                mock.call("iris", 400),
            ],
        )
        self.assertEqual(
            finished,
            [(True, "已转到 Zoom 2800 / Focus 4936 / Iris 400")],
        )
        worker.deleteLater()


if __name__ == "__main__":
    unittest.main()
