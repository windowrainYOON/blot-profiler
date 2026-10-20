"""PySide6 GUI for Blot Profiler."""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from typing import Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import core  # noqa: E402
from . import __version__  # noqa: E402



def _pick_korean_font():
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in have:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_pick_korean_font()

ROW_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#9a6324", "#469990", "#808000"]
ROLE_LABELS = {core.ROLE_TARGET: "측정 (Target)", core.ROLE_CONTROL: "Control (Loading)", core.ROLE_IGNORE: "제외"}
BG_LABELS = {
    "none": "없음",
    "rolling_ball": "Rolling ball (이미지 전체)",
    "local_border": "박스 위·아래 로컬 배경",
    "row_median": "행(가로줄) 레인 사이 배경",
}


def row_color(i: int) -> QtGui.QColor:
    return QtGui.QColor(ROW_COLORS[i % len(ROW_COLORS)])


def to_qimage(img: np.ndarray, lo: float, hi: float, invert: bool = False) -> QtGui.QImage:
    """8-bit preview; with ``invert`` values >= hi are black and <= lo white."""
    if hi <= lo:
        hi = lo + 1
    a = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255)
    if invert:
        a = 255.0 - a
    a = a.astype(np.uint8)
    a = np.ascontiguousarray(a)
    h, w = a.shape
    q = QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return q.copy()


# --------------------------------------------------------------------------
# Image view with draggable band boxes
# --------------------------------------------------------------------------

class BandItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, view: "BlotView", band: core.Band):
        super().__init__()
        self.view = view
        self.band = band
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.label = QtWidgets.QGraphicsSimpleTextItem(self)
        f = QtGui.QFont()
        f.setPointSizeF(9)
        self.label.setFont(f)
        self.label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self._syncing = False
        self.sync_from_band()

    def sync_from_band(self):
        a = self.view.analysis
        x, y, w, h = self.band.box(a.rows)
        self._syncing = True
        self.setRect(0, 0, w, h)
        self.setPos(x, y)
        self._syncing = False
        r = a.rows[self.band.row]
        color = row_color(self.band.row)
        pen = QtGui.QPen(color, 1.5)
        pen.setCosmetic(True)
        if r.role == core.ROLE_CONTROL:
            pen.setStyle(Qt.DashLine)
            pen.setWidthF(2)
        if r.role == core.ROLE_IGNORE:
            pen.setColor(QtGui.QColor("#999999"))
            pen.setStyle(Qt.DotLine)
        self.setPen(pen)
        hl = self.view.highlight_group
        if hl and hl in self.band.groups:
            self.setBrush(QtGui.QColor(255, 220, 0, 90))
        else:
            self.setBrush(Qt.NoBrush)
        self.label.setText(a.band_id(self.band))
        self.label.setBrush(color)
        self.label.setPos(0, 0)
        self.label.setY(-1)
        self.label.setTransform(QtGui.QTransform().translate(0, -14))
        self.label.setVisible(self.view.show_labels)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged and not self._syncing:
            r = self.view.analysis.rows[self.band.row]
            p = self.pos()
            self.band.cx = p.x() + r.width / 2
            self.band.cy = p.y() + r.height / 2
            self.view.band_moved()
        return super().itemChange(change, value)

    # ---- resize mode: drag an edge/corner to change the size of the whole row
    _CURSORS = {
        "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor, "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
        "lt": Qt.SizeFDiagCursor, "rb": Qt.SizeFDiagCursor, "rt": Qt.SizeBDiagCursor, "lb": Qt.SizeBDiagCursor,
    }
    _resize = None

    def _edges_at(self, pos: QtCore.QPointF) -> str:
        if not self.view.resize_mode:
            return ""
        r = self.rect()
        scale = self.view.transform().m11() or 1.0
        mx = min(7.0 / scale, r.width() / 3)
        my = min(7.0 / scale, r.height() / 3)
        e = ""
        if pos.x() <= mx:
            e += "l"
        elif pos.x() >= r.width() - mx:
            e += "r"
        if pos.y() <= my:
            e += "t"
        elif pos.y() >= r.height() - my:
            e += "b"
        return e

    def hoverMoveEvent(self, e):
        edges = self._edges_at(e.pos())
        if edges:
            self.setCursor(self._CURSORS[edges])
        else:
            self.unsetCursor()
        super().hoverMoveEvent(e)

    def hoverLeaveEvent(self, e):
        self.unsetCursor()
        super().hoverLeaveEvent(e)

    def mousePressEvent(self, e):
        edges = self._edges_at(e.pos()) if e.button() == Qt.LeftButton else ""
        if not edges:
            super().mousePressEvent(e)
            return
        a = self.view.analysis
        rows = self.view.linked_rows(self.band)
        self._resize = {
            "edges": edges,
            "rows": {ri: (a.rows[ri].width, a.rows[ri].height) for ri in rows},
            "centres": {id(b): (b.cx, b.cy) for b in a.bands if b.row in rows},
            "box": self.band.box(a.rows),
        }
        e.accept()

    def mouseMoveEvent(self, e):
        if not self._resize:
            super().mouseMoveEvent(e)
            return
        st = self._resize
        a = self.view.analysis
        x, y, w0, h0 = st["box"]
        p = e.scenePos()
        dw = dh = 0
        if "r" in st["edges"]:
            dw = round(p.x() - (x + w0))
        elif "l" in st["edges"]:
            dw = round(x - p.x())
        if "b" in st["edges"]:
            dh = round(p.y() - (y + h0))
        elif "t" in st["edges"]:
            dh = round(y - p.y())
        # the opposite edge stays put: shift the centres by half the size change
        sx = 1 if "r" in st["edges"] else -1
        sy = 1 if "b" in st["edges"] else -1
        for ri, (w, h) in st["rows"].items():
            a.rows[ri].width = max(2, w + dw)
            a.rows[ri].height = max(2, h + dh)
        for b in a.bands:
            c = st["centres"].get(id(b))
            if c is None:
                continue
            w, h = st["rows"][b.row]
            b.cx = c[0] + sx * (a.rows[b.row].width - w) / 2
            b.cy = c[1] + sy * (a.rows[b.row].height - h) / 2
        self.view.refresh_rows(set(st["rows"]))
        self.view.band_moved()
        e.accept()

    def mouseReleaseEvent(self, e):
        if self._resize:
            self._resize = None
            self.view.band_moved()
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def paint(self, painter, option, widget=None):
        # draw selection as thicker outline instead of Qt's dotted rect
        opt = QtWidgets.QStyleOptionGraphicsItem(option)
        selected = bool(opt.state & QtWidgets.QStyle.State_Selected)
        opt.state &= ~QtWidgets.QStyle.State_Selected
        super().paint(painter, opt, widget)
        if selected:
            pen = QtGui.QPen(QtGui.QColor("#1e88ff"), 3)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect())
        if self.view.resize_mode:
            # handles at the corners and edge midpoints (constant size on screen)
            r = self.rect()
            s = 3.0 / (painter.worldTransform().m11() or 1.0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.pen().color())
            for hx in (r.left(), r.center().x(), r.right()):
                for hy in (r.top(), r.center().y(), r.bottom()):
                    if hx == r.center().x() and hy == r.center().y():
                        continue
                    painter.drawRect(QtCore.QRectF(hx - s, hy - s, 2 * s, 2 * s))


class BlotView(QtWidgets.QGraphicsView):
    changed = QtCore.Signal()  # boxes moved/added/removed
    selectionChangedBands = QtCore.Signal()
    contextRequested = QtCore.Signal(QtCore.QPoint, object)  # global pos, BandItem | None
    addRequested = QtCore.Signal(float, float)
    resizeModeExited = QtCore.Signal()  # Esc pressed in resize mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QtGui.QColor("#d9d9d9"))
        self.pix_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.items_: List[BandItem] = []
        self.analysis = core.Analysis()
        self.highlight_group: Optional[str] = None
        self.show_labels = True
        self.resize_mode = False
        self._move_timer = QtCore.QTimer(self)
        self._move_timer.setSingleShot(True)
        self._move_timer.setInterval(150)
        self._move_timer.timeout.connect(self.changed.emit)
        self.scene().selectionChanged.connect(self.selectionChangedBands.emit)

    def set_image(self, qimg: QtGui.QImage, reset_zoom=True):
        pm = QtGui.QPixmap.fromImage(qimg)
        if self.pix_item is None:
            self.pix_item = self.scene().addPixmap(pm)
            self.pix_item.setZValue(-1)
        else:
            self.pix_item.setPixmap(pm)
        self.scene().setSceneRect(QtCore.QRectF(pm.rect()).adjusted(-20, -20, 20, 20))
        if reset_zoom:
            self.fit()

    def fit(self):
        if self.pix_item is not None:
            self.fitInView(self.pix_item, Qt.KeepAspectRatio)

    def set_analysis(self, a: core.Analysis):
        self.analysis = a
        self.rebuild()

    def rebuild(self):
        selected = {id(i.band) for i in self.items_ if i.isSelected()}
        sc = self.scene()
        sc.blockSignals(True)
        for it in self.items_:
            sc.removeItem(it)
        self.items_ = []
        for b in self.analysis.bands:
            it = BandItem(self, b)
            sc.addItem(it)
            if id(b) in selected:
                it.setSelected(True)
            self.items_.append(it)
        sc.blockSignals(False)

    def refresh_items(self):
        for it in self.items_:
            it.sync_from_band()

    def refresh_rows(self, rows):
        for it in self.items_:
            if it.band.row in rows:
                it.sync_from_band()

    def set_resize_mode(self, on: bool):
        self.resize_mode = bool(on)
        for it in self.items_:
            it.unsetCursor()
            it.update()

    def linked_rows(self, band: core.Band) -> set:
        """Rows resized together with ``band``'s row: its own row, rows of bands that
        share a user group with it, and rows of the other selected bands."""
        rows = {band.row}
        if band.groups:
            rows |= {b.row for b in self.analysis.bands if set(b.groups) & set(band.groups)}
        sel = self.selected_bands()
        if any(b is band for b in sel):
            rows |= {b.row for b in sel}
        return rows

    def band_moved(self):
        self._move_timer.start()

    def selected_bands(self) -> List[core.Band]:
        return [i.band for i in self.items_ if i.isSelected()]

    def select_bands(self, bands):
        ids = {id(b) for b in bands}
        self.scene().blockSignals(True)
        for it in self.items_:
            it.setSelected(id(it.band) in ids)
        self.scene().blockSignals(False)
        self.viewport().update()

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier) or not e.pixelDelta().y():
            factor = 1.0015 ** e.angleDelta().y()
            self.scale(factor, factor)
        else:
            super().wheelEvent(e)

    def mouseDoubleClickEvent(self, e):
        p = self.mapToScene(e.position().toPoint())
        item = self.itemAt(e.position().toPoint())
        if isinstance(item, QtWidgets.QGraphicsSimpleTextItem):
            item = item.parentItem()
        if not isinstance(item, BandItem) and self.pix_item is not None:
            self.addRequested.emit(p.x(), p.y())
            return
        super().mouseDoubleClickEvent(e)

    def contextMenuEvent(self, e):
        item = self.itemAt(e.pos())
        if isinstance(item, QtWidgets.QGraphicsSimpleTextItem):
            item = item.parentItem()
        if isinstance(item, BandItem) and not item.isSelected():
            self.select_bands([item.band])
        self.contextRequested.emit(e.globalPos(), item if isinstance(item, BandItem) else None)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.resize_mode:
            self.resizeModeExited.emit()
            return
        sel = [i for i in self.items_ if i.isSelected()]
        step = 5 if e.modifiers() & Qt.ShiftModifier else 1
        moves = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0), Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}
        if sel and e.key() in moves:
            dx, dy = moves[e.key()]
            for it in sel:
                it.band.cx += dx
                it.band.cy += dy
                it.sync_from_band()
            self.band_moved()
            return
        super().keyPressEvent(e)


# --------------------------------------------------------------------------
# Plot canvas
# --------------------------------------------------------------------------

class PlotCanvas(QtWidgets.QWidget):
    """Matplotlib canvas. With ``aspect`` (height / width) the canvas fills the
    available width and its height follows from the width."""

    def __init__(self, parent=None, aspect: Optional[float] = None):
        super().__init__(parent)
        self.aspect = aspect
        self.fig = Figure(figsize=(4, 3) if aspect else (5, 3.5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        if aspect:
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.canvas.setMinimumSize(1, 1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.aspect:
            h = int(round(self.width() * self.aspect))
            if self.canvas.height() != h:
                self.canvas.setFixedHeight(h)
                th = self.toolbar.sizeHint().height() if self.toolbar.isVisible() else 0
                self.setFixedHeight(h + th)

    def plot(self, series: Dict[str, np.ndarray], labels: List[str], conditions: List[str], by_condition: bool,
             ylabel: str, show_points: bool = True, title: str = ""):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if not series:
            ax.text(0.5, 0.5, "표시할 결과가 없습니다", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        names = list(series)
        n_series = len(names)
        width = 0.8 / n_series
        colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        xt_labels = None
        for si, name in enumerate(names):
            vals = np.atleast_1d(np.asarray(series[name], float))
            color = colors[si % len(colors)]
            if vals.size == 1:
                x = np.array([0.0])
                ax.bar(x + (si - (n_series - 1) / 2) * width, vals, width * 0.95, label=name, color=color)
                xt_labels = xt_labels or ["값"]
                continue
            if by_condition:
                cnames, means, sds, ns = core.condition_stats(vals, conditions, labels)
                x = np.arange(len(cnames), dtype=float) + (si - (n_series - 1) / 2) * width
                ax.bar(x, means, width * 0.95, yerr=np.where(np.array(ns) > 1, sds, 0), capsize=3, label=name, color=color, alpha=0.85)
                if show_points:
                    for ci, cn in enumerate(cnames):
                        idx = [i for i in range(len(vals)) if ((conditions[i].strip() if i < len(conditions) else "") or labels[i]) == cn]
                        pts = vals[idx]
                        jitter = np.linspace(-width * 0.2, width * 0.2, len(pts)) if len(pts) > 1 else [0]
                        ax.scatter(x[ci] + np.asarray(jitter), pts, s=12, color="k", zorder=3)
                xt_labels = cnames
            else:
                x = np.arange(len(vals), dtype=float) + (si - (n_series - 1) / 2) * width
                ax.bar(x, vals, width * 0.95, label=name, color=color)
                xt_labels = labels[: len(vals)]
        if xt_labels:
            ax.set_xticks(np.arange(len(xt_labels)))
            ax.set_xticklabels(xt_labels, rotation=45 if max(len(s) for s in xt_labels) > 6 else 0, ha="right" if max(len(s) for s in xt_labels) > 6 else "center")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        if n_series > 1 or names[0]:
            ax.legend(frameon=False, fontsize=8)
        self.canvas.draw_idle()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

def _spin(lo, hi, val, step=1, decimals=None, suffix=""):
    if decimals is None:
        w = QtWidgets.QSpinBox()
    else:
        w = QtWidgets.QDoubleSpinBox()
        w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class ImageDoc:
    """One opened blot image with its own boxes, settings and measurements."""

    def __init__(self, invert: int = 0, bg: Optional[core.BackgroundSettings] = None):
        self.image_path: Optional[str] = None
        self.raw: Optional[np.ndarray] = None
        self.signal: Optional[np.ndarray] = None
        self.corrected: Optional[np.ndarray] = None
        self._corr_key = None
        self.analysis = core.Analysis()
        self.measurements: List[core.Measurement] = []
        self.invert = invert  # index of the "밴드 색" combo
        self.bg = core.BackgroundSettings(**vars(bg)) if bg else core.BackgroundSettings()
        self.ref_lane: Optional[int] = 0
        self.alias = ""

    @property
    def name(self) -> str:
        if self.alias:
            return self.alias
        return os.path.basename(self.image_path) if self.image_path else "(빈 이미지)"


def _doc_attr(name):
    return property(lambda self: getattr(self.doc, name), lambda self, v: setattr(self.doc, name, v))


class MainWindow(QtWidgets.QMainWindow):
    # attributes of the image shown in the view
    image_path = _doc_attr("image_path")
    raw = _doc_attr("raw")
    signal = _doc_attr("signal")
    corrected = _doc_attr("corrected")
    _corr_key = _doc_attr("_corr_key")
    analysis = _doc_attr("analysis")
    measurements = _doc_attr("measurements")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blot Profiler")
        self.resize(1400, 900)
        self.docs: List[ImageDoc] = [ImageDoc()]
        self.cur = 0
        self.formulas: List[Dict[str, object]] = []  # {name, expr, normalize, image}
        self.results: Dict[str, np.ndarray] = {}
        self.result_doc: Dict[str, ImageDoc] = {}
        self._updating = False

        self.view = BlotView()
        self.view.changed.connect(self.on_boxes_changed)
        self.view.selectionChangedBands.connect(self.on_view_selection)
        self.view.contextRequested.connect(self.show_band_menu)
        self.view.addRequested.connect(self.add_band_at)
        self.view.resizeModeExited.connect(lambda: self.resize_mode.setChecked(False))

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_detect_tab(), "1. 검출")
        self.tabs.addTab(self._build_rows_tab(), "2. 행 / 레인")
        self.tabs.addTab(self._build_bands_tab(), "3. 밴드 / 그룹")
        self.tabs.addTab(self._build_results_tab(), "4. 정량 / 그래프")
        self.tabs.setMinimumWidth(520)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        # one tab per opened image
        tb = QtWidgets.QHBoxLayout()
        self.doc_tabs = QtWidgets.QTabBar()
        self.doc_tabs.setTabsClosable(True)
        self.doc_tabs.setExpanding(False)
        self.doc_tabs.setDocumentMode(True)
        self.doc_tabs.setToolTip("열린 이미지. 더블클릭: 별칭(수식에서 쓰는 이름) 변경")
        self.doc_tabs.currentChanged.connect(self.switch_doc)
        self.doc_tabs.tabCloseRequested.connect(self.close_doc)
        self.doc_tabs.tabBarDoubleClicked.connect(self.rename_doc)
        tb.addWidget(self.doc_tabs, 1)
        add_btn = QtWidgets.QPushButton("+ 이미지 추가")
        add_btn.setToolTip("TIF 이미지를 더 열어 함께 정량합니다 (여러 개 선택 가능)")
        add_btn.clicked.connect(self.open_image)
        tb.addWidget(add_btn)
        ll.addLayout(tb)
        ll.addWidget(self.view, 1)
        self.contrast_lo = QtWidgets.QSlider(Qt.Horizontal)
        self.contrast_hi = QtWidgets.QSlider(Qt.Horizontal)
        for s, v in ((self.contrast_lo, 0), (self.contrast_hi, 1000)):
            s.setRange(0, 1000)
            s.setValue(v)
            s.valueChanged.connect(lambda _=None: self.update_display(reset_zoom=False))
        self.show_corrected = QtWidgets.QCheckBox("배경 제거 이미지 표시")
        self.show_corrected.toggled.connect(lambda _=None: self.update_display(reset_zoom=False))
        self.show_labels = QtWidgets.QCheckBox("밴드 ID 표시")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(self.toggle_labels)
        self.resize_mode = QtWidgets.QCheckBox("박스 크기 조절 모드")
        self.resize_mode.setToolTip(
            "켜면 박스 가장자리·모서리를 드래그해 크기를 바꿉니다. 같은 행(과 같은 그룹·선택된 밴드의 행)이 함께 바뀝니다.\n"
            "우클릭 메뉴에서도 켜고 끌 수 있고, Esc로 끝냅니다."
        )
        self.resize_mode.toggled.connect(self.on_resize_mode)
        cl = QtWidgets.QHBoxLayout()
        cl.addWidget(QtWidgets.QLabel("표시 대비"))
        cl.addWidget(self.contrast_lo)
        cl.addWidget(self.contrast_hi)
        cl.addWidget(self.show_corrected)
        cl.addWidget(self.show_labels)
        cl.addWidget(self.resize_mode)
        ll.addLayout(cl)
        hint = QtWidgets.QLabel(
            "더블클릭: 밴드 추가 · 드래그: 이동/범위 선택 · 방향키: 미세 이동 · Delete: 삭제 · 우클릭: 그룹/역할/크기 조절 모드 · ⌘/Ctrl+휠: 확대"
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        ll.addWidget(hint)

        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.setCentralWidget(split)
        self._build_menu()
        self.statusBar().showMessage("TIF 파일을 열어 시작하세요 (파일 → 열기 또는 창에 끌어다 놓기)")
        self.setAcceptDrops(True)

        del_sc = QtGui.QShortcut(QtGui.QKeySequence.Delete, self.view)
        del_sc.activated.connect(self.delete_selected)
        bs_sc = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Backspace), self.view)
        bs_sc.activated.connect(self.delete_selected)
        self.refresh_doc_tabs()

    # ---------------------------------------------------------------- images (documents)
    @property
    def doc(self) -> ImageDoc:
        return self.docs[self.cur]

    def doc_key(self, k: int) -> str:
        return f"I{k + 1}"

    def refresh_doc_tabs(self):
        t = self.doc_tabs
        t.blockSignals(True)
        while t.count():
            t.removeTab(0)
        for k, d in enumerate(self.docs):
            t.addTab(f"{self.doc_key(k)}: {d.name}")
            t.setTabToolTip(k, d.image_path or "")
        t.setCurrentIndex(self.cur)
        t.blockSignals(False)

    def switch_doc(self, k: int):
        if k < 0 or k >= len(self.docs):
            return
        self.cur = k
        d = self.doc
        for w in (self.invert_combo, self.bg_combo, self.bg_radius, self.bg_border):
            w.blockSignals(True)
        self.invert_combo.setCurrentIndex(d.invert)
        self.bg_combo.setCurrentIndex(max(0, self.bg_combo.findData(d.bg.method)))
        self.bg_radius.setValue(d.bg.radius)
        self.bg_border.setValue(d.bg.border)
        for w in (self.invert_combo, self.bg_combo, self.bg_radius, self.bg_border):
            w.blockSignals(False)
        self.on_bg_changed(recompute=False)
        if d.raw is not None:
            self.file_label.setText(f"{os.path.basename(d.image_path)}  ({d.raw.shape[1]}×{d.raw.shape[0]}, max {d.raw.max():.0f})")
            self.setWindowTitle(f"Blot Profiler – {os.path.basename(d.image_path)}")
            self.update_display(reset_zoom=True)
        else:
            self.file_label.setText("-")
            self.setWindowTitle("Blot Profiler")
            if self.view.pix_item is not None:
                self.view.set_image(QtGui.QImage(), reset_zoom=False)
        self.view.highlight_group = None
        self.set_analysis(d.analysis)
        if self.doc_tabs.currentIndex() != k:
            self.doc_tabs.blockSignals(True)
            self.doc_tabs.setCurrentIndex(k)
            self.doc_tabs.blockSignals(False)

    def new_doc(self) -> ImageDoc:
        """Reuse the current tab if it is empty, otherwise add a new one."""
        if self.doc.raw is None:
            return self.doc
        d = ImageDoc(invert=self.doc.invert, bg=self.doc.bg)
        self.docs.append(d)
        self.cur = len(self.docs) - 1
        self.refresh_doc_tabs()
        return d

    def close_doc(self, k: int):
        d = self.docs[k]
        if d.raw is not None and d.analysis.bands:
            r = QtWidgets.QMessageBox.question(self, "이미지 닫기", f"'{d.name}' 이미지를 닫을까요? (저장하지 않은 박스는 사라집니다)")
            if r != QtWidgets.QMessageBox.Yes:
                return
        del self.docs[k]
        for f in self.formulas:
            img = int(f.get("image", 0))
            f["image"] = 0 if img == k else (img - 1 if img > k else img)
        if not self.docs:
            self.docs.append(ImageDoc())
        self.cur = min(self.cur if self.cur < k else max(0, self.cur - 1), len(self.docs) - 1)
        self.refresh_doc_tabs()
        self.switch_doc(self.cur)
        self.fill_formulas()

    def rename_doc(self, k: int):
        if k < 0:
            return
        d = self.docs[k]
        name, ok = QtWidgets.QInputDialog.getText(
            self, "이미지 별칭", f"{self.doc_key(k)}의 별칭 (수식에서 '별칭.Vinculin'처럼 사용, 비우면 파일 이름 표시):", text=d.alias
        )
        if ok:
            d.alias = name.strip()
            self.refresh_doc_tabs()
            self.fill_formulas()
            self.recompute_results()

    def on_resize_mode(self, on):
        self.view.set_resize_mode(on)
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.statusBar().showMessage(
            "박스 크기 조절 모드: 박스 가장자리/모서리를 드래그하세요 (같은 행·그룹이 함께 바뀜). Esc 또는 우클릭 메뉴로 종료"
            if on else "박스 크기 조절 모드를 끝냈습니다"
        )

    # ---------------------------------------------------------------- menus
    def _build_menu(self):
        m = self.menuBar().addMenu("파일")
        act = m.addAction("TIF 열기…", self.open_image)
        act.setShortcut(QtGui.QKeySequence.Open)
        m.addSeparator()
        m.addAction("분석 저장 (.json)…", self.save_project).setShortcut(QtGui.QKeySequence.Save)
        m.addAction("분석 불러오기…", self.load_project)
        m.addSeparator()
        m.addAction("결과 CSV 내보내기…", self.export_csv)
        m.addAction("그래프 저장 (PNG/PDF/SVG)…", self.export_plot)
        m.addAction("박스 표시 이미지 저장…", self.export_overlay)
        v = self.menuBar().addMenu("보기")
        v.addAction("창에 맞추기", self.view.fit).setShortcut("Ctrl+0")
        v.addAction("확대", lambda: self.view.scale(1.25, 1.25)).setShortcut(QtGui.QKeySequence.ZoomIn)
        v.addAction("축소", lambda: self.view.scale(0.8, 0.8)).setShortcut(QtGui.QKeySequence.ZoomOut)
        h = self.menuBar().addMenu("도움말")
        h.addAction("사용법", self.show_help)

    # ---------------------------------------------------------------- tab 1
    def _build_detect_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g_img = QtWidgets.QGroupBox("이미지")
        f = QtWidgets.QFormLayout(g_img)
        self.open_btn = QtWidgets.QPushButton("TIF 파일 열기…")
        self.open_btn.clicked.connect(self.open_image)
        f.addRow(self.open_btn)
        self.invert_combo = QtWidgets.QComboBox()
        self.invert_combo.addItems(["자동 (어두운 밴드/밝은 배경 감지)", "반전 (어두운 밴드)", "반전 안 함 (밝은 밴드, 형광)"])
        self.invert_combo.currentIndexChanged.connect(self.on_image_settings)
        f.addRow("밴드 색", self.invert_combo)
        self.file_label = QtWidgets.QLabel("-")
        self.file_label.setWordWrap(True)
        f.addRow("파일", self.file_label)
        lay.addWidget(g_img)

        g_bg = QtWidgets.QGroupBox("배경 노이즈 제거")
        f = QtWidgets.QFormLayout(g_bg)
        self.bg_combo = QtWidgets.QComboBox()
        for k in core.BACKGROUND_METHODS:
            self.bg_combo.addItem(BG_LABELS[k], k)
        self.bg_combo.setCurrentIndex(1)
        self.bg_combo.currentIndexChanged.connect(self.on_bg_changed)
        f.addRow("방법", self.bg_combo)
        self.bg_radius = _spin(2, 1000, 50, suffix=" px")
        self.bg_radius.setToolTip("Rolling ball 반지름. 밴드 높이보다 충분히 크게 (보통 밴드 높이의 3~5배).")
        self.bg_radius.editingFinished.connect(self.on_bg_changed)
        f.addRow("Rolling ball 반지름", self.bg_radius)
        self.bg_border = _spin(1, 50, 3, suffix=" px")
        self.bg_border.setToolTip("박스 바로 위·아래 띠의 두께. 이 영역의 중앙값을 픽셀당 배경으로 빼줍니다.")
        self.bg_border.editingFinished.connect(self.on_bg_changed)
        f.addRow("위·아래 띠 두께", self.bg_border)
        lay.addWidget(g_bg)

        g_det = QtWidgets.QGroupBox("밴드 자동 검출")
        f = QtWidgets.QFormLayout(g_det)
        self.sens = _spin(0.1, 10.0, 1.0, 0.1, decimals=2)
        self.sens.setToolTip("값을 올리면 흐린 밴드도 찾고, 낮추면 진한 밴드만 찾습니다.")
        f.addRow("민감도", self.sens)
        self.smooth = _spin(0.0, 10.0, 1.5, 0.5, decimals=1, suffix=" px")
        f.addRow("스무딩 (σ)", self.smooth)
        self.min_area = _spin(1, 100000, 30, suffix=" px²")
        f.addRow("최소 밴드 면적", self.min_area)
        self.pad = _spin(0, 100, 3, suffix=" px")
        f.addRow("박스 여백", self.pad)
        self.edge = _spin(0, 2000, 0, suffix=" px")
        self.edge.setToolTip("좌우 가장자리에서 이 거리 안의 물체는 무시합니다 (마커/래더 제외용).")
        f.addRow("좌우 가장자리 제외", self.edge)
        self.fill_lanes = QtWidgets.QCheckBox("밴드가 없는 레인도 박스 생성 (빈 레인 = 0으로 정량)")
        self.fill_lanes.setChecked(True)
        f.addRow(self.fill_lanes)
        self.snap_lanes = QtWidgets.QCheckBox("박스를 레인 중심에 정렬")
        self.snap_lanes.setChecked(True)
        f.addRow(self.snap_lanes)
        b = QtWidgets.QPushButton("밴드 자동 검출")
        b.setStyleSheet("font-weight: bold; padding: 6px;")
        b.clicked.connect(self.detect)
        f.addRow(b)
        b4 = QtWidgets.QPushButton("박스 세로 위치 자동 보정 (밴드에 맞추기)")
        b4.clicked.connect(self.refine_positions)
        f.addRow(b4)
        b2 = QtWidgets.QPushButton("빈 레인 채우기 (모든 행에 같은 레인 박스 추가)")
        b2.clicked.connect(self.fill_missing)
        f.addRow(b2)
        b3 = QtWidgets.QPushButton("모든 박스 지우기")
        b3.clicked.connect(self.clear_all)
        f.addRow(b3)
        lay.addWidget(g_det)
        lay.addStretch(1)
        self.on_bg_changed(recompute=False)
        return w

    # ---------------------------------------------------------------- tab 2
    def _build_rows_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel(
            "같은 가로줄(행)의 밴드는 같은 단백질로 보고 모두 같은 박스 크기로 정량합니다.\n"
            "행 이름(예: Vinculin, pErk)은 수식에서 변수로 사용됩니다."
        ))
        self.rows_table = QtWidgets.QTableWidget(0, 5)
        self.rows_table.setHorizontalHeaderLabels(["행", "이름 (단백질)", "역할", "박스 폭", "박스 높이"])
        self.rows_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.rows_table.verticalHeader().setVisible(False)
        self.rows_table.itemChanged.connect(self.on_row_item_changed)
        self.rows_table.itemSelectionChanged.connect(self.on_row_selected)
        lay.addWidget(self.rows_table, 2)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("선택한 행 삭제")
        b.clicked.connect(self.delete_row)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("선택한 밴드로 새 행 만들기")
        b.setToolTip("이미지에서 선택한 밴드들을 별도의 행(예: 이중 밴드의 위/아래)으로 분리합니다.")
        b.clicked.connect(self.split_row)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("행 박스 높이를 밴드에 맞춤")
        b.clicked.connect(self.autofit_rows)
        hb.addWidget(b)
        lay.addLayout(hb)

        lay.addWidget(QtWidgets.QLabel(
            "레인: 이름을 붙이고, 같은 조건(반복 실험)을 묶으려면 '조건' 칸에 같은 이름을 쓰세요 (그래프에서 평균±SD)."
        ))
        self.lanes_table = QtWidgets.QTableWidget(0, 3)
        self.lanes_table.setHorizontalHeaderLabels(["레인", "이름", "조건 (반복 묶음)"])
        self.lanes_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.lanes_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.lanes_table.verticalHeader().setVisible(False)
        self.lanes_table.itemChanged.connect(self.on_lane_item_changed)
        lay.addWidget(self.lanes_table, 2)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("선택한 레인 삭제")
        b.clicked.connect(self.delete_lane)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("레인 이름 일괄 입력…")
        b.clicked.connect(self.bulk_lane_names)
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)
        return w

    # ---------------------------------------------------------------- tab 3
    def _build_bands_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGroupBox("사용자 그룹")
        gl = QtWidgets.QVBoxLayout(g)
        gl.addWidget(QtWidgets.QLabel(
            "이미지에서 밴드를 선택(드래그/⌘·Shift 클릭)한 뒤 그룹에 넣으세요. 그룹 이름은 수식에서\n"
            "'이름' = 그룹 전체 합(스칼라), 'G_이름' = 레인별 합(벡터)으로 쓸 수 있습니다."
        ))
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setMaximumHeight(130)
        self.group_list.currentItemChanged.connect(self.on_group_selected)
        gl.addWidget(self.group_list)
        hb = QtWidgets.QHBoxLayout()
        for text, fn in (
            ("새 그룹", self.new_group),
            ("이름 변경", self.rename_group),
            ("그룹 삭제", self.delete_group),
            ("선택 밴드 → 그룹에 추가", self.add_selected_to_group),
            ("선택 밴드 그룹에서 빼기", self.remove_selected_from_group),
            ("그룹 밴드 선택", self.select_group_bands),
        ):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        gl.addLayout(hb)
        lay.addWidget(g)

        self.bands_table = QtWidgets.QTableWidget(0, 8)
        self.bands_table.setHorizontalHeaderLabels(["ID", "행", "레인", "Raw", "배경", "Net (정량값)", "Mean/px", "그룹"])
        self.bands_table.horizontalHeader().setSectionResizeMode(7, QtWidgets.QHeaderView.Stretch)
        self.bands_table.verticalHeader().setVisible(False)
        self.bands_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.bands_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.bands_table.itemSelectionChanged.connect(self.on_table_selection)
        lay.addWidget(self.bands_table, 1)
        self.profile = PlotCanvas()
        self.profile.toolbar.hide()
        self.profile.setMaximumHeight(170)
        lay.addWidget(QtWidgets.QLabel("선택한 밴드의 세로 intensity 프로파일 (박스 영역 음영)"))
        lay.addWidget(self.profile)
        return w

    # ---------------------------------------------------------------- tab 4
    def _build_results_tab(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        w = QtWidgets.QWidget()
        scroll.setWidget(w)
        lay = QtWidgets.QVBoxLayout(w)

        g = QtWidgets.QGroupBox("Internal control 정규화")
        f = QtWidgets.QFormLayout(g)
        self.auto_norm = QtWidgets.QCheckBox("측정(Target) 행 ÷ Control 행 (레인별) 자동 계산")
        self.auto_norm.setChecked(True)
        self.auto_norm.toggled.connect(self.recompute_results)
        f.addRow(self.auto_norm)
        self.ref_lane = QtWidgets.QComboBox()
        self.ref_lane.setToolTip("이미지마다 따로 지정합니다 (위쪽 이미지 탭에서 선택한 이미지)")
        self.ref_lane.currentIndexChanged.connect(self.on_ref_lane)
        self.ref_lane_label = QtWidgets.QLabel("기준 레인 (=1.0)")
        f.addRow(self.ref_lane_label, self.ref_lane)
        lay.addWidget(g)

        g = QtWidgets.QGroupBox("사용자 수식 (그룹 / 행 / 밴드 / 이미지 간 더하기·빼기·곱하기·나누기)")
        gl = QtWidgets.QVBoxLayout(g)
        self.formula_table = QtWidgets.QTableWidget(0, 4)
        self.formula_table.setHorizontalHeaderLabels(["이름", "수식", "기본 이미지", "기준 레인 정규화"])
        self.formula_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.formula_table.verticalHeader().setVisible(False)
        self.formula_table.setMinimumHeight(110)
        self.formula_table.setMaximumHeight(160)
        self.formula_table.itemChanged.connect(self.on_formula_changed)
        gl.addWidget(self.formula_table)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("수식 추가")
        b.clicked.connect(self.add_formula)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("수식 삭제")
        b.clicked.connect(self.delete_formula)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("사용 가능한 변수 보기")
        b.clicked.connect(self.show_variables)
        hb.addWidget(b)
        hb.addStretch(1)
        gl.addLayout(hb)
        self.formula_msg = QtWidgets.QLabel(FORMULA_HINT)
        self.formula_msg.setStyleSheet("color: #888;")
        self.formula_msg.setWordWrap(True)
        gl.addWidget(self.formula_msg)
        lay.addWidget(g)

        hb = QtWidgets.QHBoxLayout()
        self.by_condition = QtWidgets.QCheckBox("조건별 평균 ± SD")
        self.by_condition.toggled.connect(self.replot)
        hb.addWidget(self.by_condition)
        self.show_points = QtWidgets.QCheckBox("개별 값 점 표시")
        self.show_points.setChecked(True)
        self.show_points.toggled.connect(self.replot)
        hb.addWidget(self.show_points)
        hb.addWidget(QtWidgets.QLabel("그래프 항목:"))
        self.series_list = QtWidgets.QListWidget()
        self.series_list.setMaximumHeight(60)
        self.series_list.setFlow(QtWidgets.QListView.LeftToRight)
        self.series_list.setWrapping(True)
        self.series_list.itemChanged.connect(self.replot)
        hb.addWidget(self.series_list, 1)
        lay.addLayout(hb)

        self.plot = PlotCanvas(aspect=0.75)  # 4:3, height follows the width
        lay.addWidget(self.plot)
        self.results_table = QtWidgets.QTableWidget(0, 0)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_table.setMinimumHeight(220)
        lay.addWidget(self.results_table, 1)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("CSV 내보내기…")
        b.clicked.connect(self.export_csv)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("그래프 저장…")
        b.clicked.connect(self.export_plot)
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)
        return scroll

    # ---------------------------------------------------------------- image
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        for p in paths:
            self.open_path(p)

    def open_path(self, p: str):
        if p.lower().endswith(".json"):
            self.load_project(p)
        else:
            self.load_image(p)

    def open_image(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "TIF 파일 열기 (여러 개 선택 가능)", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        for p in paths:
            self.load_image(p)

    def load_image(self, path: str, detect=True, doc: Optional[ImageDoc] = None):
        """Open ``path`` in ``doc`` (default: a new tab, or the current one if it is empty)."""
        try:
            raw = core.load_image(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "열기 실패", f"이미지를 열 수 없습니다:\n{e}")
            return
        d = doc or self.new_doc()
        d.raw = raw
        d.image_path = path
        d.analysis = core.Analysis()
        d.measurements = []
        self._make_signal(d)
        self.refresh_doc_tabs()
        self.switch_doc(self.docs.index(d))
        if detect:
            self.detect()

    def _invert_choice(self, doc: Optional[ImageDoc] = None):
        return {0: None, 1: True, 2: False}[(doc or self.doc).invert]

    def _make_signal(self, doc: Optional[ImageDoc] = None):
        d = doc or self.doc
        if d.raw is None:
            return
        d.signal = core.to_signal(d.raw, self._invert_choice(d))
        d._corr_key = None

    def bg_settings(self, doc: Optional[ImageDoc] = None) -> core.BackgroundSettings:
        return (doc or self.doc).bg

    def get_corrected(self, doc: Optional[ImageDoc] = None) -> Optional[np.ndarray]:
        d = doc or self.doc
        if d.signal is None:
            return None
        bg = d.bg
        key = (bg.method == "rolling_ball", bg.radius if bg.method == "rolling_ball" else None)
        if key != d._corr_key:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                d.corrected = core.corrected_signal(d.signal, bg)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            d._corr_key = key
        return d.corrected

    def on_image_settings(self):
        self.doc.invert = self.invert_combo.currentIndex()
        self._make_signal()
        self.update_display(reset_zoom=False)
        self.remeasure()

    def on_bg_changed(self, *_, recompute=True):
        m = self.bg_combo.currentData()
        self.bg_radius.setEnabled(m == "rolling_ball")
        self.bg_border.setEnabled(m == "local_border")
        if not recompute:
            return
        self.doc.bg = core.BackgroundSettings(method=m, radius=self.bg_radius.value(), border=self.bg_border.value())
        if self.signal is not None:
            if self.show_corrected.isChecked():
                self.update_display(reset_zoom=False)
            self.remeasure()

    def update_display(self, reset_zoom=False):
        """Preview with a white background and black bands, whatever the input polarity."""
        if self.signal is None:
            return
        img = self.get_corrected() if self.show_corrected.isChecked() else self.signal
        lo_p, hi_p = np.percentile(img, [0.05, 99.95])
        span = hi_p - lo_p
        lo = lo_p + span * self.contrast_lo.value() / 1000.0 * 0.9
        hi = lo_p + span * (0.1 + 0.9 * self.contrast_hi.value() / 1000.0)
        self.view.set_image(to_qimage(img, lo, hi, invert=True), reset_zoom=reset_zoom)

    def toggle_labels(self, on):
        self.view.show_labels = on
        self.view.refresh_items()

    # ---------------------------------------------------------------- detection / editing
    def detect(self):
        if self.signal is None:
            self.statusBar().showMessage("먼저 TIF 파일을 여세요")
            return
        corr = self.get_corrected()
        settings = core.DetectSettings(
            invert=self._invert_choice(), smooth_sigma=self.smooth.value(), sensitivity=self.sens.value(),
            min_area=self.min_area.value(), pad=self.pad.value(), exclude_edge=self.edge.value(),
            fill_missing_lanes=self.fill_lanes.isChecked(), snap_to_lanes=self.snap_lanes.isChecked(),
        )
        old = self.analysis
        names = [r.name for r in old.rows]
        a = core.detect_bands(corr, settings, names if names else None)
        # keep user-entered lane labels / roles / formulas when re-detecting
        for i, r in enumerate(a.rows):
            if i < len(old.rows):
                r.role = old.rows[i].role
        if old.lane_labels and len(old.lane_labels) == a.n_lanes:
            a.lane_labels = list(old.lane_labels)
            a.lane_conditions = list(old.lane_conditions)
        a.groups = list(old.groups)
        if a.rows and not any(r.role == core.ROLE_CONTROL for r in a.rows):
            a.rows[0].role = core.ROLE_CONTROL  # loading control is usually the top row; user can change
        self.set_analysis(a)
        self.statusBar().showMessage(f"{len(a.rows)}개 행, {a.n_lanes}개 레인, {len(a.bands)}개 박스를 찾았습니다. 필요하면 박스를 직접 수정하세요.")

    def set_analysis(self, a: core.Analysis):
        self.analysis = a
        a.ensure_lane_labels()
        self.view.set_analysis(a)
        self.refresh_all()

    def refresh_all(self):
        self.fill_rows_table()
        self.fill_lanes_table()
        self.fill_groups()
        self.fill_formulas()
        self.remeasure()

    def on_boxes_changed(self):
        # box sizes may have been changed by dragging in resize mode
        for i, r in enumerate(self.analysis.rows):
            for col, attr in ((3, "width"), (4, "height")):
                sp = self.rows_table.cellWidget(i, col)
                if sp is not None and sp.value() != getattr(r, attr):
                    sp.blockSignals(True)
                    sp.setValue(getattr(r, attr))
                    sp.blockSignals(False)
        self.remeasure()

    def add_band_at(self, x, y):
        core.add_band(self.analysis, x, y)
        self.view.rebuild()
        self.fill_rows_table()
        self.fill_lanes_table()
        self.remeasure()

    def delete_selected(self):
        sel = self.view.selected_bands()
        if not sel:
            return
        ids = {id(b) for b in sel}
        self.analysis.bands = [b for b in self.analysis.bands if id(b) not in ids]
        # drop rows that became empty
        for ri in reversed(range(len(self.analysis.rows))):
            if not any(b.row == ri for b in self.analysis.bands):
                self.analysis.remove_row(ri)
        self.view.rebuild()
        self.fill_rows_table()
        self.remeasure()

    def fill_missing(self):
        n = core.fill_missing_lanes(self.analysis)
        self.view.rebuild()
        self.remeasure()
        self.statusBar().showMessage(f"{n}개 박스를 추가했습니다")

    def refine_positions(self):
        corr = self.get_corrected()
        if corr is None:
            return
        n = core.refine_vertical(corr, self.analysis)
        self.view.refresh_items()
        self.remeasure()
        self.statusBar().showMessage(f"{n}개 박스 위치를 보정했습니다")

    def clear_all(self):
        a = core.Analysis(groups=list(self.analysis.groups))
        self.set_analysis(a)

    def show_band_menu(self, gpos, item):
        sel = self.view.selected_bands()
        menu = QtWidgets.QMenu(self)
        if sel:
            gm = menu.addMenu(f"선택한 밴드 {len(sel)}개를 그룹에 추가")
            for g in self.analysis.groups:
                gm.addAction(g, lambda g=g: self._add_to_group(sel, g))
            gm.addSeparator()
            gm.addAction("새 그룹…", lambda: self._add_to_group(sel, self.new_group()))
            in_groups = sorted({g for b in sel for g in b.groups})
            if in_groups:
                rm = menu.addMenu("그룹에서 빼기")
                for g in in_groups:
                    rm.addAction(g, lambda g=g: self._remove_from_group(sel, g))
            rows = sorted({b.row for b in sel})
            if len(rows) == 1:
                ro = menu.addMenu(f"행 '{self.analysis.rows[rows[0]].name}' 역할")
                for role in core.ROLES:
                    ro.addAction(ROLE_LABELS[role], lambda role=role, r=rows[0]: self.set_row_role(r, role))
            menu.addAction("선택한 밴드로 새 행 만들기", self.split_row)
            menu.addSeparator()
            menu.addAction("삭제", self.delete_selected)
        else:
            menu.addAction("여기에 밴드 추가", lambda: self.add_band_at(*self._scene_pos(gpos)))
        menu.addSeparator()
        act = menu.addAction("박스 크기 조절 모드 (가장자리 드래그)")
        act.setCheckable(True)
        act.setChecked(self.resize_mode.isChecked())
        act.toggled.connect(self.resize_mode.setChecked)
        menu.exec(gpos)

    def _scene_pos(self, gpos):
        p = self.view.mapToScene(self.view.mapFromGlobal(gpos))
        return p.x(), p.y()

    # ---------------------------------------------------------------- rows / lanes
    def fill_rows_table(self):
        t = self.rows_table
        t.blockSignals(True)
        t.setRowCount(len(self.analysis.rows))
        for i, r in enumerate(self.analysis.rows):
            it = QtWidgets.QTableWidgetItem(f"R{i + 1}")
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setForeground(row_color(i))
            t.setItem(i, 0, it)
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(r.name))
            cb = QtWidgets.QComboBox()
            for role in core.ROLES:
                cb.addItem(ROLE_LABELS[role], role)
            cb.setCurrentIndex(core.ROLES.index(r.role))
            cb.currentIndexChanged.connect(lambda _=None, i=i, cb=cb: self.set_row_role(i, cb.currentData()))
            t.setCellWidget(i, 2, cb)
            for col, attr in ((3, "width"), (4, "height")):
                sp = _spin(2, 5000, getattr(r, attr), suffix=" px")
                sp.valueChanged.connect(lambda v, i=i, attr=attr: self.set_row_size(i, attr, v))
                t.setCellWidget(i, col, sp)
        t.blockSignals(False)
        t.resizeColumnToContents(2)

    def set_row_role(self, i, role):
        if i >= len(self.analysis.rows):
            return
        self.analysis.rows[i].role = role
        self.view.refresh_items()
        cb = self.rows_table.cellWidget(i, 2)
        if cb and cb.currentData() != role:
            cb.blockSignals(True)
            cb.setCurrentIndex(core.ROLES.index(role))
            cb.blockSignals(False)
        self.recompute_results()

    def set_row_size(self, i, attr, v):
        setattr(self.analysis.rows[i], attr, int(v))
        self.view.refresh_items()
        self.remeasure()

    def on_row_item_changed(self, item):
        if item.column() == 1:
            name = item.text().strip() or f"Row {item.row() + 1}"
            self.analysis.rows[item.row()].name = name
            self.remeasure()

    def on_row_selected(self):
        rows = {i.row() for i in self.rows_table.selectedIndexes()}
        if rows:
            self.view.select_bands([b for b in self.analysis.bands if b.row in rows])

    def _selected_row_indices(self):
        return sorted({i.row() for i in self.rows_table.selectedIndexes()})

    def delete_row(self):
        rows = self._selected_row_indices()
        if not rows:
            rows = sorted({b.row for b in self.view.selected_bands()})
        for r in reversed(rows):
            self.analysis.remove_row(r)
        self.view.rebuild()
        self.fill_rows_table()
        self.remeasure()

    def split_row(self):
        sel = self.view.selected_bands()
        if not sel:
            self.statusBar().showMessage("이미지에서 새 행으로 옮길 밴드를 먼저 선택하세요")
            return
        src = self.analysis.rows[sel[0].row]
        name, ok = QtWidgets.QInputDialog.getText(self, "새 행", "새 행(단백질) 이름:", text=f"{src.name}_2")
        if not ok:
            return
        self.analysis.rows.append(core.Row(name=name.strip() or f"Row {len(self.analysis.rows) + 1}", width=src.width, height=src.height, role=src.role))
        new = len(self.analysis.rows) - 1
        for b in sel:
            b.row = new
        for ri in reversed(range(len(self.analysis.rows))):
            if not any(b.row == ri for b in self.analysis.bands):
                self.analysis.remove_row(ri)
        self.analysis.sort()
        self.view.rebuild()
        self.fill_rows_table()
        self.remeasure()

    def autofit_rows(self):
        corr = self.get_corrected()
        if corr is None:
            return
        for ri, r in enumerate(self.analysis.rows):
            hs = []
            for b in self.analysis.bands:
                if b.row != ri:
                    continue
                prof = core.lane_profile(corr, self.analysis, b)
                if prof.size < 3 or prof.max() <= 0:
                    continue
                p = prof - np.median(prof)
                above = np.nonzero(p > 0.25 * p.max())[0]
                if above.size:
                    hs.append(above.max() - above.min() + 1)
            if hs:
                r.height = int(np.median(hs)) + 2 * self.pad.value()
        self.view.refresh_items()
        self.fill_rows_table()
        self.remeasure()

    def fill_lanes_table(self):
        a = self.analysis
        a.ensure_lane_labels()
        t = self.lanes_table
        t.blockSignals(True)
        t.setRowCount(a.n_lanes)
        for i in range(a.n_lanes):
            it = QtWidgets.QTableWidgetItem(str(i + 1))
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            t.setItem(i, 0, it)
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(a.lane_labels[i]))
            t.setItem(i, 2, QtWidgets.QTableWidgetItem(a.lane_conditions[i]))
        t.blockSignals(False)
        self.ref_lane.blockSignals(True)
        self.ref_lane.clear()
        self.ref_lane.addItem("정규화 안 함 (비율 그대로)", None)
        for i in range(a.n_lanes):
            self.ref_lane.addItem(f"{i + 1}: {a.lane_labels[i]}", i)
        idx = 0 if self.doc.ref_lane is None else self.ref_lane.findData(self.doc.ref_lane)
        self.ref_lane.setCurrentIndex(idx if idx >= 0 else min(1, self.ref_lane.count() - 1))
        self.ref_lane.blockSignals(False)
        self.ref_lane_label.setText(f"기준 레인 (=1.0) · {self.doc_key(self.cur)}")

    def on_ref_lane(self, *_):
        self.doc.ref_lane = self.ref_lane.currentData()
        self.recompute_results()

    def on_lane_item_changed(self, item):
        a = self.analysis
        a.ensure_lane_labels()
        if item.column() == 1:
            a.lane_labels[item.row()] = item.text()
            self.fill_lanes_table()
        elif item.column() == 2:
            a.lane_conditions[item.row()] = item.text()
        self.recompute_results()

    def delete_lane(self):
        lanes = sorted({i.row() for i in self.lanes_table.selectedIndexes()}, reverse=True)
        a = self.analysis
        for lane in lanes:
            a.bands = [b for b in a.bands if b.lane != lane]
            for b in a.bands:
                if b.lane > lane:
                    b.lane -= 1
            if lane < len(a.lane_labels):
                del a.lane_labels[lane]
            if lane < len(a.lane_conditions):
                del a.lane_conditions[lane]
        self.view.rebuild()
        self.fill_lanes_table()
        self.remeasure()

    def bulk_lane_names(self):
        a = self.analysis
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "레인 이름", "한 줄에 하나씩 (왼쪽 레인부터). 탭이나 쉼표 뒤에 조건을 쓸 수 있습니다:\n예) IgG 12h, IgG",
            "\n".join(f"{l}, {c}" if c else l for l, c in zip(a.lane_labels, a.lane_conditions)),
        )
        if not ok:
            return
        for i, line in enumerate([l for l in text.splitlines() if l.strip()][: a.n_lanes]):
            parts = [p.strip() for p in line.replace("\t", ",").split(",", 1)]
            a.lane_labels[i] = parts[0]
            a.lane_conditions[i] = parts[1] if len(parts) > 1 else ""
        self.fill_lanes_table()
        self.recompute_results()

    # ---------------------------------------------------------------- groups
    def fill_groups(self):
        cur = self.group_list.currentItem().text() if self.group_list.currentItem() else None
        self.group_list.blockSignals(True)
        self.group_list.clear()
        for g in self.analysis.groups:
            n = sum(1 for b in self.analysis.bands if g in b.groups)
            it = QtWidgets.QListWidgetItem(f"{g}  ({n}개 밴드)")
            it.setData(Qt.UserRole, g)
            self.group_list.addItem(it)
            if g == cur:
                self.group_list.setCurrentItem(it)
        self.group_list.blockSignals(False)

    def current_group(self) -> Optional[str]:
        it = self.group_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def on_group_selected(self, *_):
        self.view.highlight_group = self.current_group()
        self.view.refresh_items()

    def new_group(self) -> Optional[str]:
        name, ok = QtWidgets.QInputDialog.getText(self, "새 그룹", "그룹 이름 (예: IgG, pErk_total):")
        name = name.strip()
        if not ok or not name:
            return None
        if name not in self.analysis.groups:
            self.analysis.groups.append(name)
        self.fill_groups()
        for i in range(self.group_list.count()):
            if self.group_list.item(i).data(Qt.UserRole) == name:
                self.group_list.setCurrentRow(i)
        return name

    def rename_group(self):
        g = self.current_group()
        if not g:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "이름 변경", "새 이름:", text=g)
        name = name.strip()
        if not ok or not name or name == g:
            return
        self.analysis.groups[self.analysis.groups.index(g)] = name
        for b in self.analysis.bands:
            b.groups = [name if x == g else x for x in b.groups]
        self.fill_groups()
        self.remeasure()

    def delete_group(self):
        g = self.current_group()
        if not g:
            return
        self.analysis.groups.remove(g)
        for b in self.analysis.bands:
            if g in b.groups:
                b.groups.remove(g)
        self.view.highlight_group = None
        self.fill_groups()
        self.remeasure()

    def _add_to_group(self, bands, g):
        if not g:
            return
        if g not in self.analysis.groups:
            self.analysis.groups.append(g)
        for b in bands:
            if g not in b.groups:
                b.groups.append(g)
        self.fill_groups()
        self.view.refresh_items()
        self.remeasure()

    def _remove_from_group(self, bands, g):
        for b in bands:
            if g in b.groups:
                b.groups.remove(g)
        self.fill_groups()
        self.view.refresh_items()
        self.remeasure()

    def add_selected_to_group(self):
        sel = self.view.selected_bands()
        g = self.current_group() or (self.new_group() if sel else None)
        if not sel:
            self.statusBar().showMessage("이미지에서 밴드를 먼저 선택하세요")
            return
        self._add_to_group(sel, g)

    def remove_selected_from_group(self):
        g = self.current_group()
        if g:
            self._remove_from_group(self.view.selected_bands(), g)

    def select_group_bands(self):
        g = self.current_group()
        if g:
            self.view.select_bands([b for b in self.analysis.bands if g in b.groups])

    # ---------------------------------------------------------------- formulas
    def fill_formulas(self):
        t = self.formula_table
        t.blockSignals(True)
        t.setRowCount(len(self.formulas))
        for i, f in enumerate(self.formulas):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(str(f.get("name", ""))))
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(str(f.get("expr", ""))))
            cb = QtWidgets.QComboBox()
            cb.setToolTip("수식에서 접두어 없이 쓴 이름(예: Vinculin)이 가리키는 이미지. 다른 이미지는 I2.Vinculin 처럼 씁니다.")
            for k, d in enumerate(self.docs):
                cb.addItem(f"{self.doc_key(k)}: {d.name}", k)
            cb.setCurrentIndex(min(int(f.get("image", 0)), len(self.docs) - 1))
            cb.currentIndexChanged.connect(lambda _=None, i=i, cb=cb: self.set_formula_image(i, cb.currentData()))
            t.setCellWidget(i, 2, cb)
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if f.get("normalize", "1") == "1" else Qt.Unchecked)
            t.setItem(i, 3, it)
        t.blockSignals(False)
        t.resizeColumnToContents(2)

    def set_formula_image(self, i, k):
        if i < len(self.formulas):
            self.formulas[i]["image"] = int(k)
            self.recompute_results()

    def add_formula(self):
        a = self.analysis
        ctrl = next((core._safe_name(r.name) for r in a.rows if r.role == core.ROLE_CONTROL), "Control")
        tgt = next((core._safe_name(r.name) for r in a.rows if r.role == core.ROLE_TARGET), "Target")
        self.formulas.append({"name": f"수식{len(self.formulas) + 1}", "expr": f"{tgt} / {ctrl}", "normalize": "1", "image": self.cur})
        self.fill_formulas()
        self.recompute_results()

    def delete_formula(self):
        rows = sorted({i.row() for i in self.formula_table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.formulas[r]
        self.fill_formulas()
        self.recompute_results()

    def on_formula_changed(self, item):
        f = self.formulas[item.row()]
        if item.column() == 0:
            f["name"] = item.text()
        elif item.column() == 1:
            f["expr"] = item.text()
        elif item.column() == 3:
            f["normalize"] = "1" if item.checkState() == Qt.Checked else "0"
        self.recompute_results()

    def show_variables(self):
        lines = []
        for k, d in enumerate(self.docs):
            if d.raw is None:
                continue
            a = d.analysis
            v = core.formula_variables(a, d.measurements)
            alias = f" / {core._safe_name(d.alias)}" if d.alias else ""
            lines.append(f"■ {self.doc_key(k)}{alias}: {d.name}")
            lines.append("  행 (레인별 벡터): " + ", ".join(f"{core._safe_name(r.name)} (R{i + 1})" for i, r in enumerate(a.rows)))
            for g in a.groups:
                s = core._safe_name(g)
                lines.append(f"  그룹 {s} (전체 합 = {v.get(s, 0):.4g}),  G_{s} (레인별 합)")
            lines.append("")
        lines.append("개별 밴드: R<행>L<레인>  예) R2L3")
        lines.append("다른 이미지의 변수: I2.Vinculin, I2.G_IgG, I2.R2L3 (또는 별칭.Vinculin)")
        lines.append("접두어 없는 이름은 수식의 '기본 이미지' 것을 씁니다.")
        lines.append("\n연산: + - * / ** ( )   함수: sum, mean, max, min, log2, log10, sqrt")
        lines.append("\n예시:\n  pErk / Erk                    (레인별 인산화 비율)\n  (pErk_upper + pErk_lower) / Vinculin\n  G_IgG / Vinculin\n"
                     "  (I2.pErk / I2.Vinculin) / (I1.Erk / I1.Vinculin)   (두 블롯 간)\n  I2.G_IgG / I1.G_IgG")
        QtWidgets.QMessageBox.information(self, "수식에 쓸 수 있는 변수", "\n".join(lines))

    # ---------------------------------------------------------------- measurement / results
    def measure_doc(self, d: ImageDoc):
        if d.signal is None or not d.analysis.rows:
            d.measurements = []
        else:
            d.measurements = core.measure(d.signal, d.analysis, d.bg, self.get_corrected(d))

    def remeasure(self):
        self.measure_doc(self.doc)
        self.fill_bands_table()
        self.fill_groups()
        self.recompute_results()

    def fill_bands_table(self):
        a = self.analysis
        t = self.bands_table
        t.blockSignals(True)
        t.setRowCount(len(self.measurements))
        by_id = {a.band_id(b): b for b in a.bands}
        for i, m in enumerate(self.measurements):
            b = by_id.get(m.band_id)
            vals = [
                m.band_id, a.rows[m.row].name, a.lane_labels[m.lane] if m.lane < len(a.lane_labels) else str(m.lane + 1),
                f"{m.raw:.1f}", f"{m.background:.1f}", f"{m.net:.1f}", f"{m.mean:.2f}", ", ".join(b.groups) if b else "",
            ]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if c == 0:
                    it.setForeground(row_color(m.row))
                if 3 <= c <= 6:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, c, it)
        t.blockSignals(False)

    def on_table_selection(self):
        ids = {self.bands_table.item(i.row(), 0).text() for i in self.bands_table.selectedIndexes() if self.bands_table.item(i.row(), 0)}
        bands = [b for b in self.analysis.bands if self.analysis.band_id(b) in ids]
        self.view.select_bands(bands)
        self.draw_profile(bands[:1])

    def on_view_selection(self):
        sel = self.view.selected_bands()
        self.draw_profile(sel[:1])
        if sel:
            self.statusBar().showMessage(f"선택: {', '.join(self.analysis.band_id(b) for b in sel[:12])}{' …' if len(sel) > 12 else ''}")

    def draw_profile(self, bands):
        fig = self.profile.fig
        fig.clear()
        if bands and self.signal is not None:
            b = bands[0]
            corr = self.get_corrected()
            prof = core.lane_profile(corr, self.analysis, b)
            h = self.analysis.rows[b.row].height
            ax = fig.add_subplot(111)
            ax.plot(prof, color="k", lw=1)
            ax.axvspan(h, 2 * h, color=ROW_COLORS[b.row % len(ROW_COLORS)], alpha=0.2)
            if self.bg_combo.currentData() in ("local_border", "row_median"):
                m = next((m for m in self.measurements if m.band_id == self.analysis.band_id(b)), None)
                if m and m.area:
                    ax.axhline(m.background / m.area, color="r", ls="--", lw=0.8)
            ax.set_title(self.analysis.band_id(b), fontsize=8)
            ax.tick_params(labelsize=7)
        self.profile.canvas.draw_idle()

    def loaded_docs(self) -> List[ImageDoc]:
        return [d for d in self.docs if d.raw is not None]

    def recompute_results(self, *_):
        self.results = {}
        self.result_doc = {}
        msgs = []
        multi = len(self.loaded_docs()) > 1

        def put(name, vals, doc):
            self.results[name] = np.atleast_1d(np.asarray(vals, float))
            self.result_doc[name] = doc

        for k, d in enumerate(self.docs):
            a = d.analysis
            if not d.measurements or not self.auto_norm.isChecked():
                continue
            prefix = f"[{self.doc_key(k)}] " if multi else ""
            controls = [i for i, r in enumerate(a.rows) if r.role == core.ROLE_CONTROL]
            if not controls:
                msgs.append(f"{prefix}Control 행이 없습니다: '2. 행 / 레인' 탭에서 loading control 행의 역할을 Control로 지정하세요.")
                continue
            c = controls[0]
            for i, r in enumerate(a.rows):
                if r.role == core.ROLE_TARGET:
                    put(f"{prefix}{r.name}/{a.rows[c].name}", core.normalize_to_control(a, d.measurements, i, c, d.ref_lane), d)
        image_vars = [core.formula_variables(d.analysis, d.measurements) for d in self.docs]
        aliases = [d.alias for d in self.docs]
        for f in self.formulas:
            name = f.get("name") or f.get("expr")
            expr = str(f.get("expr", "")).strip()
            if not expr:
                continue
            base = min(int(f.get("image", 0)), len(self.docs) - 1)
            d = self.docs[base]
            try:
                val = core.evaluate(expr, core.project_variables(image_vars, aliases, base))
                if f.get("normalize", "1") == "1" and np.ndim(val) == 1:
                    val = core.scale_to_reference(val, d.ref_lane)
                put(name, val, d)
            except core.FormulaError as e:
                msgs.append(f"{name}: {e}")
            except Exception as e:  # pragma: no cover - defensive
                msgs.append(f"{name}: {e}")
        self.formula_msg.setText("\n".join(msgs) if msgs else FORMULA_HINT)
        self.formula_msg.setStyleSheet("color: #c0392b;" if msgs else "color: #888;")
        # series checklist (keep previous check states)
        prev = {self.series_list.item(i).text(): self.series_list.item(i).checkState() for i in range(self.series_list.count())}
        self.series_list.blockSignals(True)
        self.series_list.clear()
        for name in self.results:
            it = QtWidgets.QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(prev.get(name, Qt.Checked))
            self.series_list.addItem(it)
        self.series_list.blockSignals(False)
        self.fill_results_table()
        self.replot()

    def _result_docs(self) -> List[ImageDoc]:
        out = []
        for name in self.results:
            d = self.result_doc[name]
            if d not in out:
                out.append(d)
        return out or [self.doc]

    def fill_results_table(self):
        names = list(self.results)
        docs = self._result_docs()
        t = self.results_table
        n = max(d.analysis.n_lanes for d in docs)
        if len(docs) == 1:
            label_cols = ["레인", "조건"]
        else:
            label_cols = ["#"] + [f"{self.doc_key(self.docs.index(d))} 레인" for d in docs]
        t.clear()
        t.setColumnCount(len(names) + len(label_cols))
        t.setHorizontalHeaderLabels(label_cols + names)
        t.setRowCount(max(n, 1 if names else 0))
        for i in range(n):
            if len(docs) == 1:
                a = docs[0].analysis
                cells = [a.lane_labels[i] if i < len(a.lane_labels) else str(i + 1),
                         a.lane_conditions[i] if i < len(a.lane_conditions) else ""]
            else:
                cells = [str(i + 1)] + [d.analysis.lane_labels[i] if i < len(d.analysis.lane_labels) else "" for d in docs]
            for c, v in enumerate(cells):
                t.setItem(i, c, QtWidgets.QTableWidgetItem(v))
        off = len(label_cols)
        for c, name in enumerate(names):
            vals = self.results[name]
            if vals.size == 1:
                t.setItem(0, c + off, QtWidgets.QTableWidgetItem(f"{vals[0]:.4g}"))
                continue
            for i in range(min(n, vals.size)):
                it = QtWidgets.QTableWidgetItem("" if not np.isfinite(vals[i]) else f"{vals[i]:.4g}")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, c + off, it)
        t.resizeColumnsToContents()

    def replot(self, *_):
        chosen = [self.series_list.item(i).text() for i in range(self.series_list.count()) if self.series_list.item(i).checkState() == Qt.Checked]
        series = {k: self.results[k] for k in chosen if k in self.results}
        # don't mix scalars and lane vectors in one chart
        if any(np.size(v) > 1 for v in series.values()):
            series = {k: v for k, v in series.items() if np.size(v) > 1}
        # lane names / conditions of the image the first series comes from
        d = self.result_doc[next(iter(series))] if series else self.doc
        a = d.analysis
        n = max([np.size(v) for v in series.values()] + [a.n_lanes])
        labels = list(a.lane_labels) + [f"Lane {i + 1}" for i in range(len(a.lane_labels), n)]
        conds = list(a.lane_conditions) + [""] * (n - len(a.lane_conditions))
        any_ref = any(self.result_doc[k].ref_lane is not None for k in series)
        ylabel = "상대 intensity" if any_ref else "정량값"
        self.plot.plot(series, labels, conds, self.by_condition.isChecked(), ylabel, self.show_points.isChecked())

    # ---------------------------------------------------------------- save / export
    def project_dict(self):
        return {
            "app": "Blot Profiler",
            "version": __version__,
            "format": 2,
            "detect": {
                "sensitivity": self.sens.value(), "smooth": self.smooth.value(), "min_area": self.min_area.value(),
                "pad": self.pad.value(), "edge": self.edge.value(),
            },
            "images": [
                {
                    "image_path": d.image_path, "alias": d.alias, "invert": d.invert, "background": vars(d.bg),
                    "reference_lane": d.ref_lane, "analysis": d.analysis.to_dict(),
                }
                for d in self.docs if d.raw is not None
            ],
            "formulas": self.formulas,
        }

    def save_project(self):
        base = os.path.splitext(self.image_path)[0] + "_analysis.json" if self.image_path else "analysis.json"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "분석 저장", base, "JSON (*.json)")
        if not p:
            return
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.project_dict(), f, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(f"저장했습니다: {p}")

    def _find_image(self, img, project_path):
        if img and os.path.exists(img):
            return img
        cand = os.path.join(os.path.dirname(project_path), os.path.basename(img or ""))
        if img and os.path.exists(cand):
            return cand
        QtWidgets.QMessageBox.information(self, "이미지 찾기", f"원본 이미지를 찾을 수 없습니다:\n{img}\n이미지 파일을 선택하세요.")
        img, _ = QtWidgets.QFileDialog.getOpenFileName(self, "TIF 파일 열기", "", "Images (*.tif *.tiff *.png *.jpg)")
        return img or None

    def load_project(self, p=None):
        if not p:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "분석 불러오기", "", "JSON (*.json)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if "images" in d:
                images = d["images"]
                formulas = d.get("formulas", [])
            else:  # format 1: a single image, formulas stored in the analysis
                images = [{k: d.get(k) for k in ("image_path", "invert", "background", "reference_lane", "analysis")}]
                formulas = [dict(f, image=0) for f in d.get("analysis", {}).get("formulas", [])]
            det = d.get("detect", {})
            self.sens.setValue(det.get("sensitivity", 1.0))
            self.smooth.setValue(det.get("smooth", 1.5))
            self.min_area.setValue(det.get("min_area", 30))
            self.pad.setValue(det.get("pad", 3))
            self.edge.setValue(det.get("edge", 0))
            docs = []
            for im in images:
                path = self._find_image(im.get("image_path"), p)
                if not path:
                    continue
                doc = ImageDoc(invert=int(im.get("invert") or 0), bg=core.BackgroundSettings(**(im.get("background") or {})))
                doc.raw = core.load_image(path)
                doc.image_path = path
                doc.alias = im.get("alias") or ""
                doc.ref_lane = im.get("reference_lane", 0)
                doc.analysis = core.Analysis.from_dict(im.get("analysis") or {})
                doc.analysis.formulas = []
                doc.analysis.ensure_lane_labels()
                self._make_signal(doc)
                self.measure_doc(doc)
                docs.append(doc)
            if not docs:
                return
            self.docs = docs
            self.formulas = [dict(f) for f in formulas]
            self.cur = 0
            self.refresh_doc_tabs()
            self.switch_doc(0)
            self.statusBar().showMessage(f"불러왔습니다: {p}")
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "불러오기 실패", str(e))

    def export_csv(self):
        docs = self.loaded_docs()
        if not any(d.measurements for d in docs):
            self.statusBar().showMessage("내보낼 결과가 없습니다")
            return
        base = os.path.splitext(self.image_path)[0] + "_quant.csv" if self.image_path else "quant.csv"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "CSV 내보내기", base, "CSV (*.csv)")
        if not p:
            return
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["# Blot Profiler", __version__])
            for d in docs:
                k = self.doc_key(self.docs.index(d))
                w.writerow([f"# image {k}", d.alias, d.image_path or "", "background", BG_LABELS[d.bg.method],
                            "radius", d.bg.radius, "border", d.bg.border])
            w.writerow([])
            w.writerow(["[Bands]"])
            w.writerow(["image", "band_id", "row", "protein", "role", "lane", "lane_label", "condition", "x", "y", "width", "height",
                        "raw_intensity", "background", "net_intensity", "mean_per_px", "groups"])
            for d in docs:
                a = d.analysis
                k = self.doc_key(self.docs.index(d))
                by_id = {a.band_id(b): b for b in a.bands}
                for m in d.measurements:
                    b = by_id[m.band_id]
                    x, y, wd, ht = b.box(a.rows)
                    r = a.rows[m.row]
                    w.writerow([k, m.band_id, m.row + 1, r.name, r.role, m.lane + 1, a.lane_labels[m.lane], a.lane_conditions[m.lane],
                                x, y, wd, ht, f"{m.raw:.3f}", f"{m.background:.3f}", f"{m.net:.3f}", f"{m.mean:.4f}", ";".join(b.groups)])
            w.writerow([])
            w.writerow(["[Results]"])
            names = list(self.results)
            w.writerow(["series", "image", "reference_lane", "lane", "lane_label", "condition", "value"])
            for n in names:
                d = self.result_doc[n]
                a = d.analysis
                k = self.doc_key(self.docs.index(d))
                ref = "" if d.ref_lane is None else d.ref_lane + 1
                v = self.results[n]
                if v.size == 1:
                    w.writerow([n, k, ref, "", "", "", f"{v[0]:.6g}"])
                    continue
                for i in range(v.size):
                    w.writerow([n, k, ref, i + 1, a.lane_labels[i] if i < len(a.lane_labels) else "",
                                a.lane_conditions[i] if i < len(a.lane_conditions) else "", f"{v[i]:.6g}" if np.isfinite(v[i]) else ""])
            cond_rows = []
            for n in names:
                v = self.results[n]
                a = self.result_doc[n].analysis
                if v.size < 2 or not any(c.strip() for c in a.lane_conditions):
                    continue
                cn, means, sds, ns = core.condition_stats(v, a.lane_conditions, a.lane_labels)
                cond_rows += [[c_, n, f"{m_:.6g}", f"{s_:.6g}", k_] for c_, m_, s_, k_ in zip(cn, means, sds, ns)]
            if cond_rows:
                w.writerow([])
                w.writerow(["[By condition]"])
                w.writerow(["condition", "series", "mean", "sd", "n"])
                w.writerows(cond_rows)
            w.writerow([])
            w.writerow(["[Formulas]"])
            for fml in self.formulas:
                w.writerow([fml.get("name"), fml.get("expr"), "base image", self.doc_key(int(fml.get("image", 0)))])
        self.statusBar().showMessage(f"CSV 저장: {p}")

    def export_plot(self):
        base = os.path.splitext(self.image_path)[0] + "_plot.png" if self.image_path else "plot.png"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "그래프 저장", base, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if p:
            self.plot.fig.savefig(p, dpi=300, bbox_inches="tight")
            self.statusBar().showMessage(f"그래프 저장: {p}")

    def export_overlay(self):
        if self.view.pix_item is None:
            return
        base = os.path.splitext(self.image_path)[0] + "_boxes.png" if self.image_path else "boxes.png"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "박스 표시 이미지 저장", base, "PNG (*.png)")
        if not p:
            return
        self.view.scene().clearSelection()
        rect = self.view.pix_item.boundingRect()
        img = QtGui.QImage(int(rect.width()) * 2, int(rect.height()) * 2, QtGui.QImage.Format_ARGB32)
        img.fill(Qt.white)
        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.scene().render(painter, QtCore.QRectF(img.rect()), rect)
        painter.end()
        img.save(p)
        self.statusBar().showMessage(f"이미지 저장: {p}")

    def show_help(self):
        QtWidgets.QMessageBox.information(self, "사용법", HELP_TEXT)


FORMULA_HINT = "예: (pErk_upper + pErk_lower) / Vinculin  ·  G_IgG / Vinculin  ·  R2L3 - R2L1  ·  I2.Kras / I1.Kras (이미지 간)"

HELP_TEXT = """Blot Profiler 사용 순서

1. 검출 탭: TIF 파일을 열면 밴드를 자동으로 찾습니다.
   · 배경 제거 방법을 고르고(기본: Rolling ball) 민감도를 조절해 다시 검출할 수 있습니다.
   · 밴드가 없는 레인에도 같은 위치에 박스를 만들어 0에 가까운 값으로 정량합니다.

2. 박스 수정: 이미지에서 박스를 드래그해 옮기고, 더블클릭으로 추가, Delete로 삭제합니다.
   같은 가로줄(행)의 박스는 항상 같은 크기입니다. 크기는 행/레인 탭에서 바꾸거나,
   우클릭 → '박스 크기 조절 모드'를 켜고 박스 가장자리·모서리를 드래그합니다.
   이때 같은 행, 같은 사용자 그룹에 속한 밴드의 행, 함께 선택된 밴드의 행이 같이 바뀝니다. (Esc: 모드 종료)

   여러 이미지: '+ 이미지 추가'로 TIF를 여러 개 열면 위쪽 탭으로 전환합니다.
   탭을 더블클릭하면 별칭(수식에서 쓰는 이름)을 붙일 수 있습니다.

3. 행 / 레인 탭: 각 행에 단백질 이름을 붙이고 역할을 정합니다.
   · 측정 (Target): 정량할 단백질 (예: pErk, Kras)
   · Control (Loading): internal control (예: Vinculin, GAPDH, β-actin)
   · 제외: 계산에서 뺍니다.
   레인 이름과 조건(반복 실험 묶음)을 입력합니다.

4. 밴드 / 그룹 탭: 밴드를 선택해 사용자 그룹을 만듭니다. (이중 밴드 합치기, 처리군 묶기 등)

5. 정량 / 그래프 탭:
   · Target ÷ Control 을 레인별로 계산하고 기준 레인을 1로 정규화합니다.
   · 수식으로 행·그룹·밴드를 더하고 빼고 곱하고 나눌 수 있습니다.
   · 이미지 간 계산: I2.Kras / I1.Kras, I2.G_IgG / I1.G_IgG 처럼 'I<번호>.' 또는 '별칭.'을 붙입니다.
     접두어 없는 이름은 그 수식의 '기본 이미지'에서 가져옵니다.
   · CSV와 그래프(PNG/PDF/SVG)로 내보낼 수 있습니다.

정량값(Net) = 박스 안 픽셀 intensity 합 − 배경.
16-bit TIF 원본을 쓰면 가장 정확합니다 (스크린샷/JPEG는 포화와 압축 때문에 부정확할 수 있습니다).
"""


class App(QtWidgets.QApplication):
    """Handles files opened from Finder ("Open With" / dropping on the Dock icon)."""

    def __init__(self, argv):
        super().__init__(argv)
        self.window: Optional[MainWindow] = None
        self.pending: List[str] = []

    def event(self, e):
        if e.type() == QtCore.QEvent.FileOpen:
            path = e.file()
            if self.window is None:
                self.pending.append(path)
            else:
                self.window.open_path(path)
            return True
        return super().event(e)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = App(argv)
    app.setApplicationName("Blot Profiler")
    app.setOrganizationName("BlotProfiler")
    w = MainWindow()
    app.window = w
    w.show()
    for p in app.pending:
        w.open_path(p)
    args = [a for a in argv[1:] if not a.startswith("-psn")]
    if args and os.path.exists(args[0]):
        if args[0].lower().endswith(".json"):
            w.load_project(args[0])
        else:
            w.load_image(args[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
