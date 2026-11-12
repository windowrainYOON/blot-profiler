"""Layouts that follow the width of their pane: a flow layout (items fill a
line, then continue on the next one) and `reflow`, which turns the rows of
buttons / check boxes of an already built panel into flow layouts and lets
form rows put a long field under its label, so a narrow tools pane shows
everything without sideways scrolling."""

from __future__ import annotations

import shiboken6
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QBoxLayout,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLayout,
    QSizePolicy,
    QWidget,
)


class FlowLayout(QLayout):
    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self):
        return self.minimumSize()  # the width comes from the pane, not the items

    def minimumSize(self):
        # the widest single item
        size = QSize()
        for it in self._visible():
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _visible(self):
        return [it for it in self._items if not (it.widget() and it.widget().isHidden())]

    def _place(self, rect, move):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        gap = self.spacing()
        for it in self._visible():
            hint = it.sizeHint()
            if x > r.x() and x + hint.width() > r.right() + 1:
                x, y, line_h = r.x(), y + line_h + gap, 0
            if move:
                it.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + gap
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def _only_buttons(lay) -> list[QWidget] | None:
    """The widgets of `lay` when they are all buttons / check boxes (spacers
    allowed), else None."""
    widgets = []
    for i in range(lay.count()):
        it = lay.itemAt(i)
        if it.layout() is not None:
            return None
        w = it.widget()
        if w is None:
            continue  # stretch / spacing
        if not isinstance(w, QAbstractButton):
            return None
        widgets.append(w)
    return widgets


def _to_flow(lay, min_n=3) -> FlowLayout | None:
    widgets = _only_buttons(lay)
    if not widgets or len(widgets) < min_n:
        return None
    if isinstance(lay, QGridLayout):  # keep the grid's reading order
        pos = {}
        for w in widgets:
            r, c, _rs, _cs = lay.getItemPosition(lay.indexOf(w))
            pos[w] = (r, c)
        widgets.sort(key=pos.get)
    flow = FlowLayout(spacing=max(lay.spacing(), 4))
    if len(widgets) >= 3 and all(isinstance(w, QCheckBox) for w in widgets):
        # a row of three or more options picked together: multi-select chips
        from . import theme  # noqa: PLC0415

        for w in widgets:
            if not theme.is_auto_choice(w):
                theme.chip(w)
    for w in widgets:
        lay.removeWidget(w)
        if w.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding:
            w.setSizePolicy(QSizePolicy.Policy.Preferred, w.sizePolicy().verticalPolicy())
        flow.addWidget(w)
    return flow


def reflow(root: QWidget, min_n: int = 3):
    """Make the panel `root` (built already) fit narrow widths: rows of three
    (min_n) or more buttons / check boxes become flow layouts, form rows may
    wrap."""
    for w in [root] + root.findChildren(QWidget):
        lay = w.layout()
        if lay is None or w.property('hig_no_reflow'):
            continue  # e.g. an icon-button capsule: one toolbar group, never wrapped
        if isinstance(lay, (QHBoxLayout, QGridLayout)):
            flow = _to_flow(lay, min_n)
            if flow is not None:  # the widget's own layout: swap it
                margins = lay.contentsMargins()
                shiboken6.delete(lay)
                flow.setContentsMargins(margins)
                w.setLayout(flow)
                continue
        _reflow_layout(lay, w, min_n)


def _reflow_layout(lay, owner, min_n=3):
    if isinstance(lay, QFormLayout):
        lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        subs = [lay.itemAt(i).layout() for i in range(lay.count())]
        for sub in subs:
            if sub is None:
                continue
            flow = _to_flow(sub, min_n) if isinstance(sub, (QHBoxLayout, QGridLayout)) else None
            if flow is None:
                _reflow_layout(sub, owner, min_n)
                continue
            row, role = lay.getLayoutPosition(sub)
            lay.takeAt(lay.indexOf(sub))
            sub.setParent(None)
            lay.setLayout(row, role, flow)
        return
    for i in range(lay.count()):
        sub = lay.itemAt(i).layout()
        if sub is None:
            continue
        if isinstance(sub, (QHBoxLayout, QGridLayout)) and isinstance(lay, QBoxLayout):
            flow = _to_flow(sub, min_n)
            if flow is not None:
                stretch = lay.stretch(i)
                lay.takeAt(i)
                sub.setParent(None)
                lay.insertLayout(i, flow, stretch)
                continue
        _reflow_layout(sub, owner, min_n)
