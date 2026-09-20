"""Save key calibration results into a shared summary JSON.

Calibration scripts in this directory write their standalone result files
below working_data/.  In addition, every successful run saves one entry to:

    detection/calibration/output/calibrate.json

The shared summary keeps one entry per calibration type and lens zoom name.
A new result with the same ("type", "name") pair overwrites that earlier
result in place; results with the same type but a different zoom name are
kept.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUMMARY_ROOT = Path(__file__).resolve().parent / "output"
SUMMARY_FILENAME = "calibrate.json"
SUMMARY_SCHEMA_VERSION = 1


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def summary_file_path() -> Path:
    """Path of the summary file directly below detection/calibration/output."""
    return SUMMARY_ROOT / SUMMARY_FILENAME


def load_summary() -> dict[str, Any]:
    """Read the summary document, or a fresh empty one."""
    path = summary_file_path()
    if not path.exists():
        return {"schema_version": SUMMARY_SCHEMA_VERSION, "entries": []}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("cannot read calibration summary {}: {}".format(path, exc)) from exc
    if not text.strip():
        return {"schema_version": SUMMARY_SCHEMA_VERSION, "entries": []}

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        backup = backup_invalid_summary(path)
        print(
            "warning: existing calibration summary {} is invalid JSON; "
            "moved it to {} and started a fresh summary".format(path, backup),
            file=sys.stderr,
        )
        return {"schema_version": SUMMARY_SCHEMA_VERSION, "entries": []}

    if not isinstance(document, dict) or not isinstance(
        document.get("entries"), list
    ):
        raise ValueError(
            "existing calibration summary {} has an unexpected structure; "
            "expected an object with an 'entries' list".format(path)
        )
    return document


def backup_invalid_summary(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name("{}_invalid_{}{}".format(
        path.stem,
        timestamp,
        path.suffix,
    ))
    suffix = 2
    while backup.exists():
        backup = path.with_name("{}_invalid_{}_{:02d}{}".format(
            path.stem,
            timestamp,
            suffix,
            path.suffix,
        ))
        suffix += 1
    shutil.move(str(path), str(backup))
    return backup


def zoom_address(zoom: Any) -> int:
    """Parse a LensConnect zoom address."""
    if isinstance(zoom, bool):
        raise ValueError("lens zoom must be an integer address, not a boolean")
    if isinstance(zoom, int):
        return zoom
    elif isinstance(zoom, float) and zoom.is_integer():
        return int(zoom)
    elif isinstance(zoom, str):
        text = zoom.strip()
        if not text:
            raise ValueError("lens zoom must not be empty")
        return int(text, 0)
    raise ValueError("lens zoom must be an integer address")


def zoom_entry_name(zoom: Any) -> str:
    """Return the summary entry name used for a LensConnect zoom address."""
    return "zoom_{:05d}".format(zoom_address(zoom))


def current_lens_position(
    device_number: Any = None,
    motor_names: tuple[str, ...] = ("zoom", "focus"),
) -> dict[str, int]:
    """Read current LensConnect motor addresses using one connection."""
    detection_dir = Path(__file__).resolve().parents[1]
    detection_dir_text = str(detection_dir)
    if detection_dir_text not in sys.path:
        sys.path.insert(0, detection_dir_text)

    import LensCamera as lens_camera

    try:
        capabilities = lens_camera.connect(device_number)
    except Exception as exc:
        raise ValueError("cannot read current lens position: {}".format(exc)) from exc
    try:
        position = {}
        for motor_name in motor_names:
            if not lens_camera.motor_supported(motor_name, capabilities):
                raise ValueError(
                    "connected lens does not support {}".format(motor_name)
                )
            lens_camera.ensure_ready(motor_name, init_if_needed=False)
            position[motor_name] = int(lens_camera.read_current(motor_name))
        return position
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("cannot read current lens position: {}".format(exc)) from exc
    finally:
        lens_camera.close()


def current_lens_zoom(device_number: Any = None) -> int:
    """Read the current LensConnect zoom address without moving the lens."""
    return current_lens_position(
        device_number=device_number,
        motor_names=("zoom",),
    )["zoom"]


def append_entry(
    entry: dict[str, Any],
    *,
    zoom: Any = None,
    name: str | None = None,
    device_number: Any = None,
    include_saved_at_utc: bool = True,
) -> Path:
    """Save one calibration entry, overwriting only the same type and name."""
    if not isinstance(entry, dict):
        raise ValueError("calibration summary entry must be an object")

    path = summary_file_path()
    document = load_summary()
    document["schema_version"] = SUMMARY_SCHEMA_VERSION

    saved_entry = dict(entry)
    if include_saved_at_utc:
        saved_entry.setdefault("saved_at_utc", utc_timestamp())
    else:
        saved_entry.pop("saved_at_utc", None)

    if zoom is None:
        zoom = saved_entry.get("lens_zoom")
    if name is None:
        name = saved_entry.get("name")
    if name is None:
        if zoom is None:
            zoom = current_lens_zoom(device_number=device_number)
        name = zoom_entry_name(zoom)

    saved_entry["name"] = str(name)
    if zoom is not None:
        saved_entry["lens_zoom"] = zoom_address(zoom)

    entry_type = saved_entry.get("type")
    entry_name = saved_entry.get("name")
    if entry_type is None or entry_name is None:
        document["entries"].append(saved_entry)
    else:
        index = _find_last_entry_index(
            document["entries"], entry_type, entry_name
        )
        if index is None:
            document["entries"].append(saved_entry)
        else:
            document["entries"][index] = saved_entry
            document["entries"] = [
                item
                for position, item in enumerate(document["entries"])
                if position == index
                or not _entry_matches(item, entry_type, entry_name)
            ]

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name("{}.tmp".format(path.name))
    with open(str(temp_path), "w", encoding="utf-8") as file_obj:
        json.dump(document, file_obj, indent=2, sort_keys=False)
        file_obj.write("\n")
    temp_path.replace(path)
    return path


def _entry_matches(item: Any, entry_type: Any, entry_name: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == entry_type
        and item.get("name") == entry_name
    )


def _find_last_entry_index(
    entries: list[Any], entry_type: Any, entry_name: Any
) -> int | None:
    """Index of the last entry with the same type and name, if any."""
    for index in range(len(entries) - 1, -1, -1):
        if _entry_matches(entries[index], entry_type, entry_name):
            return index
    return None
