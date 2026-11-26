"""Controls of the 1.0 layout (spec §8): segmented choice, settings cards, switches,
slider + number rows, value chips, a sidebar section list and a collapsible card."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import theme
from .flow import FlowLayout
from ..i18n import _t  # noqa: E402


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
            b.setStyleSheet("min-height: 22px; padding: 0 9px; font-size: 12px; border-radius: 11px;")
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


def sync_items(lst: QtWidgets.QListWidget, rows, check=None):
    """Make the rows of ``lst`` read ``rows`` [(text, tooltip)] without QListWidget.clear().

    1.4.3: clear() deleted every item at once, often inside the click that changed them (a
    release on the Figure preview, a click on a row); PySide then crashed in the item's
    destructor (SIGSEGV / heap corruption: 'the app quits on its own'). Rows are now rewritten
    in place; extra rows are taken out and kept alive until the next call, so nothing is
    deleted while Qt still delivers an event to it. ``check``: {text: Qt.CheckState} for
    check-box rows."""
    old = getattr(lst, "_spare_items", [])
    lst._spare_items = []  # the ones taken out last time may go now
    del old
    for i, (text, tip) in enumerate(rows):
        it = lst.item(i)
        if it is None:
            it = QtWidgets.QListWidgetItem(text)
            if check is not None:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            lst.addItem(it)
        elif it.text() != text:
            it.setText(text)
        it.setToolTip(tip or "")
        if check is not None:
            st = check.get(text, Qt.Checked)
            if it.checkState() != st:
                it.setCheckState(st)
    while lst.count() > len(rows):
        lst._spare_items.append(lst.takeItem(lst.count() - 1))


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
        sync_items(self, [(f"{text}    {note}" if note else text, text) for text, note in rows])
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
        lay.setContentsMargins(10, 0, 11, 0)
        lay.setSpacing(5)
        self.setFixedHeight(26)  # a capsule: the radius (12) stays under half the height, else Qt draws it square
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
        self.setToolTip(_t("확대 ] · 축소 [ · 눌러서 화면 맞춤 (⌘0)"))
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


class WrapCheckBox(QtWidgets.QCheckBox):
    """A check box whose text wraps onto more lines in a narrow box instead of widening it (1.5.3)."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def _opt(self):
        o = QtWidgets.QStyleOptionButton()
        self.initStyleOption(o)
        return o

    def _indicator(self) -> QtCore.QRect:
        return self.style().subElementRect(QtWidgets.QStyle.SE_CheckBoxIndicator, self._opt(), self)

    def _text_x(self) -> int:
        return self._indicator().right() + 1 + 6

    def _text_h(self, w: int) -> int:
        r = QtCore.QRect(0, 0, max(20, w - self._text_x() - 2), 10000)
        return self.fontMetrics().boundingRect(r, int(Qt.TextWordWrap), self.text()).height()

    def minimumSizeHint(self):
        sh = super().minimumSizeHint()
        return QtCore.QSize(min(sh.width(), self._text_x() + 80), sh.height())

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return max(super().sizeHint().height(), self._text_h(w) + 8)

    def paintEvent(self, e):
        p = QtWidgets.QStylePainter(self)
        o = self._opt()
        r = self._indicator()
        if self._text_h(self.width()) > self.fontMetrics().height() * 1.5:
            r.moveTop(4)  # more lines: the box stays beside the first one
        o.rect = r
        p.drawPrimitive(QtWidgets.QStyle.PE_IndicatorCheckBox, o)
        x = self._text_x()
        group = QtGui.QPalette.Active if self.isEnabled() else QtGui.QPalette.Disabled
        p.setPen(self.palette().color(group, QtGui.QPalette.WindowText))
        p.drawText(QtCore.QRect(x, 0, self.width() - x - 2, self.height()),
                   int(Qt.TextWordWrap | Qt.AlignVCenter | Qt.AlignLeft), self.text())


_SHORT_HEADS = {"위 여백": "위", "아래 여백": "아래", "Top margin": "Top", "Bottom margin": "Bottom"}


def tip_choices(page: QtWidgets.QWidget):
    """The whole text of each (maybe cut) choice in its tooltip."""
    for cb in page.findChildren(QtWidgets.QComboBox):
        cb.setToolTip(cb.currentText())


def fit_inspector(page: QtWidgets.QWidget):
    """1.5.3: a page of a fixed-width inspector never gets wider than the box: long choices
    are cut with … (whole text in the tooltip), a field that does not fit beside its title
    goes under it, buttons and wrapped notes shrink."""
    for cb in page.findChildren(QtWidgets.QComboBox):
        cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(6)
        fm = cb.fontMetrics()
        widest = max((fm.horizontalAdvance(cb.itemText(i)) for i in range(cb.count())), default=0)
        # a long choice takes the free width of its row (and is cut only when that is not enough)
        cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding if widest > 120 else QtWidgets.QSizePolicy.Preferred,
                         QtWidgets.QSizePolicy.Fixed)
    tip_choices(page)
    for lay in page.findChildren(QtWidgets.QFormLayout):
        lay.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        for r in range(lay.rowCount()):
            it = lay.itemAt(r, QtWidgets.QFormLayout.FieldRole)
            w = it.widget() if it is not None else None
            if isinstance(w, QtWidgets.QComboBox) and w.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Expanding:
                it.setAlignment(Qt.Alignment())  # not pinned to the row end: it may take the whole field
    for b in page.findChildren(QtWidgets.QPushButton):
        b.setMinimumWidth(0)
        sp = b.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Preferred)
        b.setSizePolicy(sp)
    for lab in page.findChildren(QtWidgets.QLabel):
        if lab.wordWrap():
            lab.setMinimumWidth(0)


class ElideLabel(QtWidgets.QLabel):
    """One line that ends in … when it does not fit (text() keeps the whole message; 1.5.3)."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def minimumSizeHint(self):
        return QtCore.QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setPen(self.palette().color(QtGui.QPalette.WindowText))
        r = self.contentsRect()
        p.drawText(r, int(Qt.AlignVCenter | Qt.AlignRight), self.fontMetrics().elidedText(self.text(), Qt.ElideRight, r.width()))
