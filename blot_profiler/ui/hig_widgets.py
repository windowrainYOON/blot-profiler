"""Design-system widgets that a style sheet cannot make (macOS tahoe style,
Desktop profile): SF Symbols-style icons, toolbar icon buttons, the capsule
chip switcher that replaces the inspector's tab bar, and the region
splitter and shade (Sidebar | Canvas | Inspector, RegionLayout).

All colours come from `theme.color` (tokens, light / dark) and are re-read
on every paint or on `theme.refresh`, so nothing here holds a fixed colour.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractButton,
    QMenu,
    QSizePolicy,
    QSplitter,
    QSplitterHandle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .flow import FlowLayout

# 24 × 24 stroke glyphs (project/assets/Icons/icons.json of the design
# system; `zoom.region` and `fit` are drawn in the same style because the
# set has no such symbol)
ICONS = {
    'magnifyingglass': '<circle cx="10.5" cy="10.5" r="6.2"/><path d="M15.2 15.2 20 20"/>',
    'plus': '<path d="M12 5v14M5 12h14"/>',
    'trash': ('<path d="M4 7h16M9.5 7V5a1 1 0 0 1 1-1h3a1 1 0 0 1 1 1v2M6 7l.8 12a1.6 1.6 0 0 '
              '0 1.6 1.5h7.2a1.6 1.6 0 0 0 1.6-1.5L18 7"/><path d="M10 11v6M14 11v6"/>'),
    'xmark': '<path d="M6 6l12 12M18 6 6 18"/>',
    'folder': ('<path d="M3.5 7.5A2.5 2.5 0 0 1 6 5h3.2l2 2.2H18a2.5 2.5 0 0 1 2.5 2.5v7.8A2.5 '
               '2.5 0 0 1 18 20H6a2.5 2.5 0 0 1-2.5-2.5z"/>'),
    'zoom.region': ('<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v3M4 15v4a1 1 0 0 0 1 1h3"/>'
                    '<circle cx="14.5" cy="14.5" r="3.6"/><path d="M17.2 17.2 20.5 20.5"/>'),
    'fit': '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/><rect x="8" y="8" width="8" height="8" rx="1.5"/>',
    'arrow.counterclockwise': '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4 4v4.5h4.5"/>',
    'crop': ('<path d="M7 3v12.5A1.5 1.5 0 0 0 8.5 17H21"/>'
             '<path d="M3 7h12.5A1.5 1.5 0 0 1 17 8.5V21"/>'),
    'tray.and.arrow.down': ('<path d="M12 3.5V14M8.3 10.3 12 14l3.7-3.7"/><path d="M4.5 13.5v4A2.5 '
                            '2.5 0 0 0 7 20h10a2.5 2.5 0 0 0 2.5-2.5v-4"/>'),
    # HIG standard symbols (icons.json) used by the ROI editing tools
    'scissors': ('<circle cx="6.5" cy="6.5" r="2.7"/><circle cx="6.5" cy="17.5" r="2.7"/>'
                 '<path d="M8.6 8.2 20 18M8.6 15.8 20 6"/>'),
    'plus.circle': '<circle cx="12" cy="12" r="8.5"/><path d="M12 8v8M8 12h8"/>',
    'arrow.uturn.backward': '<path d="M8.5 5.5 4 10l4.5 4.5"/><path d="M4 10h10a5 5 0 0 1 0 10h-4"/>',
    'arrow.uturn.forward': '<path d="M15.5 5.5 20 10l-4.5 4.5"/><path d="M20 10H10a5 5 0 0 0 0 10h4"/>',
    'chevron.up.chevron.down': '<path d="m7.5 9.5 4.5-4.5 4.5 4.5M7.5 14.5l4.5 4.5 4.5-4.5"/>',
    # not in the set: drawn on the same 24 px grid and stroke
    'minus.circle': '<circle cx="12" cy="12" r="8.5"/><path d="M8 12h8"/>',
    'cursorarrow': '<path d="M6.5 4.5 18 11.2l-5 1.4-2.4 5.1z"/><path d="M13 12.6 17 17"/>',
    'plus.square.dashed': ('<rect x="4" y="4" width="16" height="16" rx="3.5" stroke-dasharray="3 2.4"/>'
                           '<path d="M12 8.5v7M8.5 12h7"/>'),
    'arrow.triangle.merge': ('<path d="M6 4v3.5a5 5 0 0 0 2 4L12 14.5V20"/><path d="M18 4v3.5a5 5 0 0 1-2 4'
                             'L12 14.5"/><path d="M9.5 17.5 12 20l2.5-2.5"/>'),
    'number': '<path d="M10 4 8 20M16 4l-2 16M4.5 9h15M4 15h15"/>',
}


def svg(name: str, color: str, opacity: float = 1.0, stroke: float = 1.7) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-opacity="{opacity:.3f}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>').encode()


def pixmap(name: str, size: int, token: str = 'label', dpr: float = 2.0,
           opacity: float = 1.0) -> QPixmap:
    """The glyph `name` at `size` px in the colour token `token`."""
    c = theme.color(token)
    pm = QPixmap(round(size * dpr), round(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    r = QSvgRenderer(QByteArray(svg(name, c.name(), c.alphaF() * opacity)))
    qp = QPainter(pm)
    r.render(qp, QRectF(0, 0, size, size))
    qp.end()
    return pm


def icon(name: str, size: int = 16, token: str = 'label') -> QIcon:
    ic = QIcon(pixmap(name, size, token))
    ic.addPixmap(pixmap(name, size, 'on-accent'), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(pixmap(name, size, 'label-tertiary'), QIcon.Mode.Disabled)
    return ic


class IconButton(QToolButton):
    """Toolbar icon button (desktop): 28 × 28 with a 16 px glyph, always with a
    tool tip; the same command also lives in a menu or a shortcut."""

    def __init__(self, glyph: str, tip: str, checkable: bool = False, parent=None,
                 token: str = 'label'):
        super().__init__(parent)
        self.setProperty('hig_icon', glyph)
        self.setProperty('hig_icon_token', token)  # 'red' for a destructive action
        self.setToolTip(tip)
        self.setAccessibleName(tip.split('\n')[0])
        self.setCheckable(checkable)
        self.setFixedSize(28, 28)
        self.setIconSize(QSize(16, 16))
        self.refresh_icon()

    def refresh_icon(self):
        self.setIcon(icon(self.property('hig_icon'), 16, self.property('hig_icon_token') or 'label'))


def icon_group(*buttons) -> QWidget:
    """Icon buttons on one capsule (toolbar group: gap 2 inside)."""
    from PySide6.QtWidgets import QHBoxLayout  # noqa: PLC0415

    cap = QWidget()
    cap.setObjectName('hig_capsule')
    cap.setProperty('hig_no_reflow', True)
    cap.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    lay = QHBoxLayout(cap)
    lay.setContentsMargins(3, 3, 3, 3)
    lay.setSpacing(2)
    for b in buttons:
        lay.addWidget(b)
    cap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return cap


# ---- chip tabs ------------------------------------------------------------------

class Chip(QAbstractButton):
    """One capsule chip of the tab switcher: accent pill with white text when
    selected, a quiet fill otherwise; feature tabs show an × on hover."""

    closeClicked = Signal()
    H = 26
    PAD = 10
    X = 14  # close glyph size

    def __init__(self, text: str, closable: bool, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setAutoExclusive(False)
        self.closable = closable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _font(self):
        f = self.font()
        f.setWeight(f.Weight.DemiBold if self.isChecked() else f.Weight.Medium)
        return f

    def sizeHint(self):
        fm = QFontMetrics(self._font())
        w = fm.horizontalAdvance(self.text()) + 2 * self.PAD
        if self.closable:
            w += self.X + 2
        return QSize(w, self.H)

    def _x_rect(self) -> QRectF:
        return QRectF(self.width() - self.PAD - self.X + 4, (self.H - self.X) / 2, self.X, self.X)

    def mouseReleaseEvent(self, ev):
        if self.closable and ev.button() == Qt.MouseButton.LeftButton \
                and self._x_rect().adjusted(-3, -3, 3, 3).contains(ev.position()):
            self.setDown(False)
            self.closeClicked.emit()
            return
        if self.closable and ev.button() == Qt.MouseButton.MiddleButton:
            self.closeClicked.emit()
            return
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        self.update()
        super().mouseMoveEvent(ev)

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, r.height() / 2, r.height() / 2)
        hover = self.underMouse()
        if self.isChecked():
            qp.fillPath(path, theme.color('blue'))
            fg = theme.color('on-accent')
        else:
            qp.fillPath(path, theme.color('fill' if (hover or self.isDown()) else 'fill-tertiary'))
            fg = theme.color('label')
        if self.hasFocus():
            qp.setPen(theme.color('blue'))
            qp.drawPath(path)
        qp.setFont(self._font())
        qp.setPen(fg)
        tr = QRectF(self.PAD, 0, self.width() - 2 * self.PAD
                    - ((self.X + 2) if self.closable else 0), self.H)
        qp.drawText(tr, Qt.AlignmentFlag.AlignCenter, self.text())
        if self.closable:
            x = self._x_rect()
            token = 'on-accent' if self.isChecked() else 'label-secondary'
            op = 1.0 if hover or self.isChecked() else 0.55
            qp.drawPixmap(x.toRect(), pixmap('xmark', self.X, token, opacity=op))
        qp.end()


class ChipTabs(QTabWidget):
    """QTabWidget whose tab bar is replaced by wrapping capsule chips
    (`self.chips`, placed above the pages by the owner).  The QTabWidget API
    (addTab, insertTab, removeTab, currentChanged, tabCloseRequested ...)
    stays as it is."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabBar().hide()
        self.setDocumentMode(True)
        self.chips = QWidget()
        self.chips.setObjectName('hig_chips')
        self._flow = FlowLayout(self.chips, spacing=4)
        self._flow.setContentsMargins(0, 0, 0, 0)
        self._fixed: set[QWidget] = set()
        self.currentChanged.connect(lambda _i: self._sync())

    def set_fixed(self, page: QWidget):
        """`page` cannot be closed (no × on its chip)."""
        self._fixed.add(page)
        self._rebuild()

    def tabInserted(self, index):
        super().tabInserted(index)
        self._rebuild()

    def tabRemoved(self, index):
        super().tabRemoved(index)
        self._rebuild()

    def chip_texts(self) -> list[str]:
        return [c.text() for c in self._chips()]

    def _chips(self) -> list[Chip]:
        return [self._flow.itemAt(i).widget() for i in range(self._flow.count())]

    def _rebuild(self):
        while self._flow.count():
            w = self._flow.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        for i in range(self.count()):
            page = self.widget(i)
            closable = self.tabsClosable() and page not in self._fixed
            chip = Chip(self.tabText(i), closable, self.chips)
            if closable:  # a tool tip that only repeats the label would cover the page
                chip.setToolTip('우클릭 또는 ×: 탭 닫기')
            chip.clicked.connect(lambda _=False, p=page: self.setCurrentWidget(p))
            chip.closeClicked.connect(lambda p=page: self._close(p))
            chip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            chip.customContextMenuRequested.connect(
                lambda pos, c=chip, p=page: self._menu(c, p, pos))
            self._flow.addWidget(chip)
            chip.show()
        self._sync()
        self.chips.updateGeometry()

    def _close(self, page):
        i = self.indexOf(page)
        if i >= 0:
            self.tabCloseRequested.emit(i)

    def _menu(self, chip, page, pos):
        if not chip.closable:
            return
        m = QMenu(chip)
        m.addAction('탭 닫기', lambda: self._close(page))
        m.exec(chip.mapToGlobal(pos))

    def _sync(self):
        cur = self.currentIndex()
        for i, c in enumerate(self._chips()):
            c.setChecked(i == cur)
            c.updateGeometry()
        self._flow.invalidate()


class _PaneHandle(QSplitterHandle):
    """Splitter handle between two regions: the gap between the floating
    boxes, painted as the window's own layer (the sidebar's surface)."""

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.fillRect(self.rect(), theme.region_color('sidebar'))
        qp.end()


class PaneSplitter(QSplitter):
    """The main window's Sidebar | Canvas | Inspector splitter: the canvas
    and inspector boxes float BOX_GAP from each other and the window edge."""

    HANDLE_W = theme.BOX_GAP

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setHandleWidth(self.HANDLE_W)
        g = theme.BOX_GAP
        self.setContentsMargins(0, g, g, g)
        self.setAutoFillBackground(False)

    def createHandle(self):
        return _PaneHandle(self.orientation(), self)


class RegionShade(QWidget):
    """The shade the raised sidebar casts on the canvas: an overlay gradient
    over the canvas' leading edge (crisp at the edge, soft inwards), not a
    drop shadow.  It lets the mouse through and follows its parent's size."""

    def __init__(self, canvas: QWidget, offset: int = 0):
        super().__init__(canvas)
        self.setObjectName('hig_region_shade')
        self.offset = offset  # the part of the shade already drawn by the handle
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        canvas.installEventFilter(self)
        self._fit()

    def eventFilter(self, obj, ev):
        if ev.type() in (ev.Type.Resize, ev.Type.Show, ev.Type.ChildAdded):
            self._fit()
        return False

    def _fit(self):
        p = self.parentWidget()
        self.setGeometry(0, 0, theme.SHADE_W, p.height())
        self.raise_()

    def paintEvent(self, _ev):
        a = theme.region_shade()
        if a <= 0:
            return
        qp = QPainter(self)
        qp.fillRect(self.rect(), theme.shade_gradient(-self.offset, a))
        qp.end()


def inspector(tabs: ChipTabs) -> QWidget:
    """The inspector region (raised surface): the chips on top, the pages
    below, separated by space (gap-section 24), not by a line."""
    pane = QWidget()
    pane.setObjectName('hig_inspector')
    pane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    lay = QVBoxLayout(pane)
    left, top, right, bottom = theme.INSPECTOR_MARGINS
    # the scroll bar sits in the right margin; pages keep 4 px for it (unify)
    lay.setContentsMargins(left, top, right - 8, bottom)
    lay.setSpacing(theme.REGION_GAP)
    lay.addWidget(tabs.chips)
    lay.addWidget(tabs, 1)
    return pane
