"""Controls of the 1.0 layout (spec §8): segmented choice, settings cards, switches,
slider + number rows, value chips, a sidebar section list and a collapsible card."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import theme
from .flow import FlowLayout


class Segmented(QtWidgets.QWidget):
    """One of 2–5 choices (segmented control): capsule buttons side by side."""

    changed = QtCore.Signal(str)

    def __init__(self, items: Iterable[Tuple[str, str]], parent=None, small: bool = False):
        super().__init__(parent)
        self.setObjectName("hig_segment")
        self.setAttribute(Qt.WA_StyledBackground, True)  # the capsule track
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for key, text in items:
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            b.setFocusPolicy(Qt.TabFocus)
            if small:
                b.setStyleSheet("font-size: 12px; padding: 0 9px;")
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        if self.buttons:
            next(iter(self.buttons.values())).setChecked(True)

    def value(self) -> str:
        return next((k for k, b in self.buttons.items() if b.isChecked()), "")

    def set_value(self, key: str, emit: bool = False):
        b = self.buttons.get(key)
        if b is not None and not b.isChecked():
            b.setChecked(True)
            if emit:
                self.changed.emit(key)


def card(title: Optional[str] = None) -> Tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
    """A settings card (rounded inset group, title above, rows with hairlines)."""
    g = QtWidgets.QGroupBox(title or "")
    if not title:
        g.setProperty("hig_untitled", True)  # no title band above it
    f = QtWidgets.QFormLayout(g)
    f.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    return g, f


def switch(text: str = "", checked: bool = False) -> QtWidgets.QCheckBox:
    cb = QtWidgets.QCheckBox(text)
    cb.setChecked(checked)
    theme.switch(cb)
    return cb


def slider_row(spin: QtWidgets.QDoubleSpinBox, scale: int = 100) -> QtWidgets.QWidget:
    """Continuous value (spec §8): slider + number field kept in step with ``spin``."""
    sl = QtWidgets.QSlider(Qt.Horizontal)
    sl.setRange(int(spin.minimum() * scale), int(spin.maximum() * scale))
    sl.setValue(int(round(spin.value() * scale)))
    sl.valueChanged.connect(lambda v: abs(spin.value() * scale - v) >= 1 and spin.setValue(v / scale))
    spin.valueChanged.connect(lambda v: (sl.blockSignals(True), sl.setValue(int(round(v * scale))), sl.blockSignals(False)))
    return theme.slider_field(sl, spin)


def prominent(btn: QtWidgets.QPushButton) -> QtWidgets.QPushButton:
    return theme.prominent(btn)


def hint(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    theme.hint(lab)
    return lab


def section(text: str) -> QtWidgets.QLabel:
    """Small grey header above a group of sidebar rows."""
    lab = QtWidgets.QLabel(text)
    theme.secondary(lab)
    f = lab.font()
    f.setPixelSize(11)
    f.setBold(True)
    lab.setFont(f)
    lab.setContentsMargins(10, 10, 10, 2)
    return lab


class ChipRow(QtWidgets.QWidget):
    """Values shown as chips (nested in a flow); a click emits the chip's index."""

    clicked = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self, spacing=4)
        self.flow.setContentsMargins(0, 0, 0, 0)

    def set_chips(self, texts: List[str], current: int = -1):
        while self.flow.count():
            w = self.flow.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for i, t in enumerate(texts):
            b = QtWidgets.QPushButton(t)
            b.setCheckable(True)
            b.setChecked(i == current)
            b.setStyleSheet("min-height: 22px; padding: 0 8px; font-size: 12px;")
            b.clicked.connect(lambda _=False, i=i: self.clicked.emit(i))
            self.flow.addWidget(b)
        self.updateGeometry()


class Collapsible(QtWidgets.QWidget):
    """A card that opens and closes with its title (세부 설정)."""

    def __init__(self, title: str, body: QtWidgets.QWidget, opened: bool = False, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.btn = QtWidgets.QToolButton()
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(opened)
        self.btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)
        self.btn.setStyleSheet("QToolButton { font-weight: 600; border: none; background: transparent; }")
        self.btn.toggled.connect(self._toggle)
        lay.addWidget(self.btn)
        self.body = body
        body.setVisible(opened)
        lay.addWidget(body)

    def _toggle(self, on):
        self.btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)


class SideList(QtWidgets.QListWidget):
    """Rows of the sidebar (selection = accent pill); a row may carry a grey note on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        accent = theme.TOKENS[theme.scheme()]["blue"]
        self.setStyleSheet("QListWidget { background: transparent; border: none; }"
                           "QListWidget::item { min-height: 28px; padding: 2px 8px; border-radius: 10px; }"
                           f"QListWidget::item:selected {{ background: {accent}; color: white; }}")

    def set_rows(self, rows: List[Tuple[str, str]], current: int = -1):
        self.blockSignals(True)
        self.clear()
        for text, note in rows:
            it = QtWidgets.QListWidgetItem(f"{text}    {note}" if note else text)
            it.setToolTip(text)
            self.addItem(it)
        if 0 <= current < self.count():
            self.setCurrentRow(current)
        self.blockSignals(False)
        self.setFixedHeight(max(28, self.sizeHintForRow(0) * self.count() + 6) if self.count() else 28)


def color_dot(color: str, size: int = 10) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel()
    lab.setFixedSize(size, size)
    lab.setStyleSheet(f"background: {color}; border-radius: 3px;")
    return lab


def icon(name: str) -> QtGui.QIcon:
    from .hig_widgets import icon as _icon  # noqa: PLC0415

    return _icon(name, 16)


class ZoomChip(QtWidgets.QFrame):
    """The zoom of a canvas as a small chip over it (mockup 1.0.1): magnifier +
    percent only.  Zoom with [ ] or ⌘± as before; a click on the chip fits."""

    def __init__(self, host: Optional[QtWidgets.QWidget], minus=None, plus=None, fit=None, corner: str = "bottom"):
        super().__init__(host)  # host None: an ordinary widget laid out by its parent (graph header)
        self.corner = corner  # "bottom" right (main, Figure) or "top" right (graph)
        self.setObjectName("zoom_chip")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._fit = fit
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 11, 4)
        lay.setSpacing(5)
        ic = QtWidgets.QLabel()
        ic.setPixmap(icon("magnifyingglass").pixmap(13, 13))
        lay.addWidget(ic)
        self.label = QtWidgets.QLabel("100%")
        self.label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        f = self.label.font()
        f.setPixelSize(12)
        f.setWeight(QtGui.QFont.Medium)
        self.label.setFont(f)
        lay.addWidget(self.label)
        self.setToolTip("확대 [ · 축소 ] · 눌러서 화면 맞춤 (⌘0)")
        if fit is not None:
            self.setCursor(Qt.PointingHandCursor)
        self.host = host
        if host is not None:
            host.installEventFilter(self)
            self.place()

    def mouseReleaseEvent(self, e):
        if self._fit is not None and e.button() == Qt.LeftButton:
            self._fit()
        super().mouseReleaseEvent(e)

    def set_zoom(self, k: float):
        self.label.setText(f"{k * 100:.0f}%")
        self.adjustSize()
        self.place()

    def place(self):
        self.adjustSize()
        if self.host is None:
            return
        y = 12 if self.corner == "top" else max(0, self.host.height() - self.height() - 12)
        self.move(max(0, self.host.width() - self.width() - 12), y)
        self.raise_()

    def eventFilter(self, obj, e):
        if obj is self.host and e.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.place()
        return False
