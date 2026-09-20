"""Data-driven settings form generated from a field schema.

Each Field maps one jsonnet config key to an editor widget.  New config keys
need nothing more than an extra Field entry — the form, change tracking and
surgical saving all work generically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from GUI.core.jsonnet_store import ConfigError

SPIN_RANGE = (-2_000_000_000, 2_000_000_000)
DOUBLE_RANGE = (-1.0e9, 1.0e9)
DECIMALS = 6
LABEL_ALIGN = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


@dataclass(frozen=True)
class Field:
    """One editable config key."""

    key: str
    label: str
    kind: str = "str"
    section: str = "常规"
    tip: str = ""
    options: tuple = ()
    unit: str = ""
    decimals: int = 3
    step: float = 1.0


def F(key: str, label: str, **kwargs: Any) -> Field:
    """Concise Field constructor."""
    return Field(key=key, label=label, **kwargs)


# ---------------------------------------------------------------------------
# Editors: each exposes get_value()/set_value() and emits value_changed.
# ---------------------------------------------------------------------------


class Editor(QWidget):
    value_changed = Signal()

    def get_value(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def set_value(self, value: Any) -> None:  # pragma: no cover
        raise NotImplementedError


class NoWheelSpinBox(QSpinBox):
    """Integer spin box that leaves wheel scrolling to its parent."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Floating-point spin box that leaves wheel scrolling to its parent."""

    def wheelEvent(self, event) -> None:
        event.ignore()


def _blocked(method: Callable) -> Callable:
    def wrapper(self, *args):
        self.blockSignals(True)
        try:
            method(self, *args)
        finally:
            self.blockSignals(False)

    return wrapper


class BoolEditor(Editor):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.check = QCheckBox()
        self.check.toggled.connect(self.value_changed)
        layout.addWidget(self.check)
        layout.addStretch(1)

    def get_value(self):
        return self.check.isChecked()

    @_blocked
    def set_value(self, value):
        self.check.setChecked(bool(value))


def _make_spin(decimals: int, step: float) -> QDoubleSpinBox:
    spin = NoWheelDoubleSpinBox()
    spin.setRange(*DOUBLE_RANGE)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    return spin


class IntEditor(Editor):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spin = NoWheelSpinBox()
        self.spin.setRange(*SPIN_RANGE)
        self.spin.valueChanged.connect(self.value_changed)
        layout.addWidget(self.spin)
        layout.addStretch(1)

    def get_value(self):
        return self.spin.value()

    @_blocked
    def set_value(self, value):
        self.spin.setValue(int(value) if value is not None else 0)


class FloatEditor(Editor):
    def __init__(self, field: Field | None = None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spin = _make_spin(
            field.decimals if field else DECIMALS, field.step if field else 1.0
        )
        self.spin.valueChanged.connect(self.value_changed)
        layout.addWidget(self.spin)
        layout.addStretch(1)

    def get_value(self):
        return round(self.spin.value(), DECIMALS)

    @_blocked
    def set_value(self, value):
        self.spin.setValue(float(value) if value is not None else 0.0)


class StrEditor(Editor):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.textEdited.connect(self.value_changed)
        layout.addWidget(self.edit)

    def get_value(self):
        return self.edit.text()

    @_blocked
    def set_value(self, value):
        self.edit.setText("" if value is None else str(value))


class EnumEditor(Editor):
    EXTRA_SUFFIX = " (配置当前值)"

    def __init__(self, options, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for option in options:
            self.combo.addItem(str(option), option)
        self._base_count = self.combo.count()
        self.combo.currentIndexChanged.connect(self.value_changed)
        layout.addWidget(self.combo)
        layout.addStretch(1)

    def get_value(self):
        return self.combo.currentData()

    @_blocked
    def set_value(self, value):
        index = self.combo.findData(value)
        if index < 0 and value is not None:
            # Config holds a value outside the declared options; show it
            # truthfully instead of silently switching to another option.
            while self.combo.count() > self._base_count:
                self.combo.removeItem(self.combo.count() - 1)
            self.combo.addItem(f"{value}{self.EXTRA_SUFFIX}", value)
            index = self.combo.count() - 1
        if index < 0:
            index = 0
        self.combo.setCurrentIndex(index)


class Vec2Editor(Editor):
    """Two numbers edited as an [x, y] list."""

    def __init__(self, field: Field | None = None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.spins = []
        for _ in range(2):
            spin = _make_spin(
                field.decimals if field else 3, field.step if field else 0.1
            )
            spin.valueChanged.connect(self.value_changed)
            self.spins.append(spin)
            layout.addWidget(spin)
        layout.addStretch(1)

    def get_value(self):
        return [round(spin.value(), DECIMALS) for spin in self.spins]

    @_blocked
    def set_value(self, value):
        values = list(value) if value is not None else [0.0, 0.0]
        for index, spin in enumerate(self.spins):
            spin.setValue(float(values[index]) if index < len(values) else 0.0)


class _ListEditor(Editor):
    sep_tip = "多个值用英文逗号分隔"

    def __init__(self, converter, parent=None):
        super().__init__(parent)
        self._converter = converter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(self.sep_tip)
        self.edit.textEdited.connect(self.value_changed)
        layout.addWidget(self.edit)

    def get_value(self):
        text = self.edit.text().strip()
        if not text or text.lower() in ("null", "none", "[]"):
            return []
        try:
            return [self._converter(item) for item in text.split(",") if item.strip()]
        except ValueError:
            raise ConfigError("列表值格式不正确: " + text)

    @_blocked
    def set_value(self, value):
        if value is None:
            self.edit.setText("")
        elif isinstance(value, (list, tuple)):
            self.edit.setText(", ".join(str(item) for item in value))
        else:
            self.edit.setText(str(value))


class IntListEditor(_ListEditor):
    def __init__(self, parent=None):
        super().__init__(lambda item: int(item.strip(), 0), parent)


class FloatListEditor(_ListEditor):
    def __init__(self, parent=None):
        super().__init__(float, parent)


NULL_TIP = (
    "勾选“自定义”写入具体数值; 取消勾选写入 null, 由脚本使用默认行为\n"
    "(覆盖配置文件中的原表达式)。"
)


class NullWrapEditor(Editor):
    """Wrap another editor: unchecked checkbox means null."""

    def __init__(self, inner: Editor, parent=None):
        super().__init__(parent)
        self.inner = inner
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.check = QCheckBox("自定义")
        self.check.setToolTip(NULL_TIP)
        self.check.toggled.connect(self._sync_enabled)
        self.check.toggled.connect(self.value_changed)
        inner.value_changed.connect(self.value_changed)
        layout.addWidget(self.check)
        layout.addWidget(inner)
        layout.addStretch(1)
        inner.setEnabled(False)

    def _sync_enabled(self, checked: bool):
        self.inner.setEnabled(checked)

    def get_value(self):
        if not self.check.isChecked():
            return None
        return self.inner.get_value()

    @_blocked
    def set_value(self, value):
        if value is None:
            self.check.setChecked(False)
            self.inner.setEnabled(False)
        else:
            self.check.setChecked(True)
            self.inner.setEnabled(True)
            self.inner.set_value(value)


_KIND_FACTORIES: dict[str, Callable[[Field], Editor]] = {
    "bool": lambda f: BoolEditor(),
    "int": lambda f: IntEditor(),
    "float": lambda f: FloatEditor(f),
    "str": lambda f: StrEditor(),
    "enum": lambda f: EnumEditor(f.options),
    "int_list": lambda f: IntListEditor(),
    "float_list": lambda f: FloatListEditor(),
    "vec2": lambda f: Vec2Editor(f),
    "int_null": lambda f: NullWrapEditor(IntEditor()),
    "float_null": lambda f: NullWrapEditor(FloatEditor(f)),
    "str_null": lambda f: NullWrapEditor(StrEditor()),
    "vec2_null": lambda f: NullWrapEditor(Vec2Editor(f)),
}


def build_editor(field: Field) -> Editor:
    factory = _KIND_FACTORIES.get(field.kind)
    if factory is None:
        raise ConfigError(f"未知的字段类型 {field.kind!r} ({field.key})")
    return factory(field)


# ---------------------------------------------------------------------------
# The form itself
# ---------------------------------------------------------------------------


class SettingsForm(QWidget):
    """Grouped editors for one config file, with change tracking."""

    changed = Signal(int)  # number of modified keys

    def __init__(self, fields: list[Field], parent=None):
        super().__init__(parent)
        self.fields = list(fields)
        self.editors: dict[str, Editor] = {}
        self.original: dict[str, Any] = {}
        self._missing: set[str] = set()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        sections: dict[str, tuple[QGroupBox, QFormLayout]] = {}
        for field_spec in self.fields:
            if field_spec.section in sections:
                _, form = sections[field_spec.section]
            else:
                group = QGroupBox(field_spec.section)
                form = QFormLayout(group)
                form.setLabelAlignment(LABEL_ALIGN)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                form.setContentsMargins(8, 4, 8, 8)
                form.setVerticalSpacing(8)
                root.addWidget(group)
                sections[field_spec.section] = (group, form)

            editor = build_editor(field_spec)
            editor.value_changed.connect(self._recount)
            label_text = field_spec.label
            if field_spec.unit:
                label_text += f" ({field_spec.unit})"
            label = QLabel(label_text)
            if field_spec.tip:
                label.setToolTip(field_spec.tip)
                editor.setToolTip(field_spec.tip)
            form.addRow(label, editor)
            self.editors[field_spec.key] = editor
        root.addStretch(1)

    def apply(self, values: dict[str, Any]) -> None:
        """Load effective config values into the editors."""
        self._missing = {
            key for key in self.editors if key not in values
        }
        for key, editor in self.editors.items():
            if key in self._missing:
                editor.setEnabled(False)
                editor.setToolTip("该键不存在于当前配置文件中")
                self.original[key] = None
                continue
            editor.setEnabled(True)
            editor.set_value(values[key])
            self.original[key] = values[key]
        self._recount()

    def collect(self) -> dict[str, Any]:
        return {
            key: editor.get_value()
            for key, editor in self.editors.items()
            if key not in self._missing
        }

    def updates(self) -> dict[str, Any]:
        """Only keys whose value differs from the loaded state."""
        return {
            key: value
            for key, value in self.collect().items()
            if self.original.get(key) != value
        }

    def modified_count(self) -> int:
        return len(self.updates())

    def _recount(self) -> None:
        self.changed.emit(self.modified_count())
