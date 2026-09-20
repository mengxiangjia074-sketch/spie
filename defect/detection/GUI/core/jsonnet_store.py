"""Load jsonnet configs and edit single keys without touching anything else.

The detection scripts read all parameters from jsonnet config files.  The GUI
edits those files with a *surgical* text edit: for each changed key only the
value expression after `"key":` is replaced by a JSON literal.  Comments,
import statements, unmodified expressions (e.g. `lens.frame_width`) and the
overall file layout stay byte-identical.  After writing, the file is
re-loaded with the same loader the scripts use and every value is compared;
on any mismatch the original text is restored and an error is raised.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import GUI_BACKUP_DIR

# LensCamera.settings resolves jsonnet imports (config files cross-reference
# each other via `local lens = import ...`); import it lazily so this module
# can still be imported without the detection package on sys.path.
_settings = None


class ConfigError(RuntimeError):
    pass


def _settings_module():
    global _settings
    if _settings is None:
        from LensCamera import settings as settings_module

        _settings = settings_module
    return _settings


def unwrap(item: Any) -> Any:
    """Mirror of the scripts' setting_value(): unwrap {"value": x} wrappers."""
    if isinstance(item, dict) and "value" in item and set(item) <= {"value", "note"}:
        return item["value"]
    return item


def load_effective(path: Path | str) -> dict[str, Any]:
    """Load a config file with import resolution and unwrapped values."""
    path = Path(path)
    try:
        raw = _settings_module().Settings(path, str(path)).values
    except _settings_module().SettingsError as exc:  # type: ignore[attr-defined]
        raise ConfigError(str(exc)) from exc
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}") from exc
    return {name: unwrap(value) for name, value in raw.items()}


# --------------------------------------------------------------------------
# Surgical text edit
# --------------------------------------------------------------------------

_KEY_LINE = re.compile(r'^(?P<indent>[ \t]*)"(?P<key>[^"\n]+)"[ \t]*:', re.M)
_SELF_DOT_REFERENCE = re.compile(r'\bself\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)')
_SELF_INDEX_REFERENCE = re.compile(r'\bself\s*\[\s*"([^"\n]+)"\s*\]')


def _skip_comment(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")


def _value_span(text: str, search_from: int) -> tuple[int, int] | None:
    """Return (start, end) of the value expression beginning at search_from.

    Tracks bracket depth and string state so multi-line arrays are handled.
    Returns None when the region cannot be determined safely (caller must
    refuse the edit rather than guess).
    """
    n = len(text)
    i = search_from
    while i < n and text[i] in " \t":
        i += 1
    start = i
    depth = 0
    in_string = False
    escape = False
    while i < n:
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            i += 1
            continue
        if char in "[{":
            depth += 1
            i += 1
            continue
        if char in "]}":
            if depth == 0:
                break  # closing brace of the enclosing object
            depth -= 1
            i += 1
            if depth == 0:
                break  # end of a composite value
            continue
        if depth == 0 and (char in ",\n\r"):
            break  # end of a scalar / expression value
        i += 1
    else:
        return None
    end = i
    while end > start and text[end - 1] in " \t\r\n":
        end -= 1
    if end <= start:
        return None
    return start, end


def _render(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"无法序列化配置值 {value!r}: {exc}") from exc


def _rewrite_values(text: str, updates: dict[str, Any]) -> str:
    """Replace each updated key's value expression with a JSON literal."""
    targets = {}  # key -> (value_start, value_end)
    offset = 0
    for match in _KEY_LINE.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if _skip_comment(text[line_start : match.end()]):
            continue
        key = match.group("key")
        if key in updates and key not in targets:
            span = _value_span(text, match.end())
            if span is None:
                raise ConfigError(
                    f"无法定位 {key!r} 的当前值范围, 拒绝修改以保护配置文件"
                )
            targets[key] = span

    missing = set(updates) - set(targets)
    if missing:
        raise ConfigError(
            "配置文件中没有这些设置项: " + ", ".join(sorted(missing))
        )

    result = text
    # Apply replacements from the end so earlier spans stay valid.
    for key in sorted(targets, key=lambda k: targets[k][0], reverse=True):
        start, end = targets[key]
        result = result[:start] + _render(updates[key]) + result[end:]
    return result


def _dependent_keys(text: str, changed_keys: set[str]) -> set[str]:
    """Return keys whose Jsonnet expressions depend on changed object fields."""
    dependencies: dict[str, set[str]] = {}
    for match in _KEY_LINE.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if _skip_comment(text[line_start : match.end()]):
            continue
        key = match.group("key")
        if key in dependencies:
            continue
        span = _value_span(text, match.end())
        if span is None:
            continue
        expression = text[span[0] : span[1]]
        dependencies[key] = set(_SELF_DOT_REFERENCE.findall(expression))
        dependencies[key].update(_SELF_INDEX_REFERENCE.findall(expression))

    affected = set(changed_keys)
    while True:
        newly_affected = {
            key
            for key, references in dependencies.items()
            if key not in affected and references.intersection(affected)
        }
        if not newly_affected:
            return affected
        affected.update(newly_affected)


def set_values(path: Path | str, updates: dict[str, Any]) -> dict[str, Any]:
    """Surgically write ``updates`` into the jsonnet config at *path*.

    Writes literal JSON values (valid jsonnet), validates the result by
    reloading, and restores the original file on any problem.  Returns the
    new effective settings.  A timestamped backup of the previous version is
    kept below working_data/gui_backups/.
    """
    path = Path(path)
    if not updates:
        return load_effective(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")

    original_text = path.read_text(encoding="utf-8")
    old_effective = load_effective(path)

    new_text = _rewrite_values(original_text, updates)
    if new_text == original_text:
        return old_effective

    backup_dir = GUI_BACKUP_DIR / path.name
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, backup_dir / f"{stamp}.jsonnet")

    try:
        path.write_text(new_text, encoding="utf-8", newline="")
        new_effective = load_effective(path)
    except Exception:
        path.write_text(original_text, encoding="utf-8", newline="")
        raise

    problems = []
    missing = object()
    affected_keys = _dependent_keys(original_text, set(updates))

    # Explicitly edited values must match exactly after Jsonnet evaluation.
    for key, expected in updates.items():
        if new_effective.get(key, missing) != expected:
            problems.append(key)

    # Values outside the dependency chain must remain unchanged. Derived
    # fields such as lens_autofocus: self.capture_autofocus are expected to
    # change when their source field changes.
    for key, old_value in old_effective.items():
        if key in affected_keys:
            continue
        if new_effective.get(key, missing) != old_value:
            problems.append(key)
    if problems:
        path.write_text(original_text, encoding="utf-8", newline="")
        raise ConfigError(
            "写入后校验失败, 已恢复原文件; 异常键: " + ", ".join(problems)
        )
    return new_effective
