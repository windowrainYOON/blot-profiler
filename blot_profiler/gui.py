"""PySide6 GUI for Blot Profiler."""

from __future__ import annotations

import csv
import json
import os
import re
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
from . import figure  # noqa: E402
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
        self.setBrush(Qt.NoBrush)
        self.label.setText(self.view.ids.get(id(self.band), ""))
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
    filesDropped = QtCore.Signal(list)  # image/analysis files dropped on the view
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
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.pix_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.marker_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self.marker_caption = None
        self.items_: List[BandItem] = []
        self.analysis = core.Analysis()
        self.ids: Dict[int, str] = {}  # id(band) -> band ID
        self.show_labels = True
        self.resize_mode = False
        self._move_timer = QtCore.QTimer(self)
        self._move_timer.setSingleShot(True)
        self._move_timer.setInterval(150)
        self._move_timer.timeout.connect(self.changed)
        self.scene().selectionChanged.connect(self.selectionChangedBands)

    def set_image(self, qimg: QtGui.QImage, reset_zoom=True):
        pm = QtGui.QPixmap.fromImage(qimg)
        if self.pix_item is None:
            self.pix_item = self.scene().addPixmap(pm)
            self.pix_item.setZValue(-1)
        else:
            self.pix_item.setPixmap(pm)
        self._update_rect()
        if reset_zoom:
            self.fit()

    def set_marker(self, qimg: Optional[QtGui.QImage]):
        """Show the marker (ladder) image left of the blot, same height, for reference."""
        sc = self.scene()
        for it in (self.marker_item, self.marker_caption):
            if it is not None:
                sc.removeItem(it)
        self.marker_item = self.marker_caption = None
        if qimg is not None and not qimg.isNull():
            pm = QtGui.QPixmap.fromImage(qimg)
            gap = max(12, int(0.03 * pm.width()))
            self.marker_item = sc.addPixmap(pm)
            self.marker_item.setZValue(-1)
            self.marker_item.setPos(-(pm.width() + gap), 0)
            self.marker_caption = sc.addSimpleText("마커 (참고용)")
            self.marker_caption.setBrush(QtGui.QColor("#1565c0"))
            self.marker_caption.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            self.marker_caption.setPos(-(pm.width() + gap), -22)
        self._update_rect()

    def content_rect(self) -> QtCore.QRectF:
        r = QtCore.QRectF()
        for it in (self.pix_item, self.marker_item):
            if it is not None and not it.pixmap().isNull():
                r = r.united(it.sceneBoundingRect())
        return r

    def _update_rect(self):
        r = self.content_rect()
        if not r.isEmpty():
            self.scene().setSceneRect(r.adjusted(-20, -30, 20, 20))

    def fit(self):
        r = self.content_rect()
        if not r.isEmpty():
            self.fitInView(r, Qt.KeepAspectRatio)

    def set_analysis(self, a: core.Analysis):
        self.analysis = a
        self.rebuild()

    def rebuild(self):
        selected = {id(i.band) for i in self.items_ if i.isSelected()}
        sc = self.scene()
        self.ids = self.analysis.band_ids()
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
        self.ids = self.analysis.band_ids()
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
        """Rows resized together with ``band``'s row: its own protein group and the
        rows of the other selected bands."""
        rows = {band.row}
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

    @staticmethod
    def _paths(e):
        md = e.mimeData()
        if not md.hasUrls():
            return []
        exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".json")
        return [u.toLocalFile() for u in md.urls() if u.toLocalFile().lower().endswith(exts)]

    def dragEnterEvent(self, e):
        if self._paths(e):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if self._paths(e):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        paths = self._paths(e)
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)
        else:
            super().dropEvent(e)

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
            if self.pix_item.sceneBoundingRect().contains(p):  # not on the marker beside the blot
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


    def plot_bars(self, keys: List[str], series: Dict[str, Dict[str, object]], ylabel: str, show_points: bool = True):
        """Bars over named lanes; ``series[name]`` has mean, sd (or None) and points (per-image values or None)."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        if not series or not keys:
            ax.text(0.5, 0.5, "표시할 결과가 없습니다", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        n_series = len(series)
        width = 0.8 / n_series
        scalar = all(np.size(s["mean"]) == 1 for s in series.values())
        for si, (name, s) in enumerate(series.items()):
            mean = np.atleast_1d(np.asarray(s["mean"], float))
            x = np.arange(mean.size, dtype=float) + (si - (n_series - 1) / 2) * width
            sd = s.get("sd")
            yerr = None if sd is None else np.where(np.asarray(s["n"]) > 1, sd, 0)
            ax.bar(x, mean, width * 0.95, yerr=yerr, capsize=3, label=name, color=colors[si % len(colors)], alpha=0.85)
            pts = s.get("points")
            if show_points and pts:
                for pi, p in enumerate(pts):
                    off = (pi - (len(pts) - 1) / 2) * width * 0.4 / max(len(pts), 1)
                    ax.scatter(x[: len(p)] + off, p[: mean.size], s=12, color="k", zorder=3)
        labels = ["값"] if scalar else list(keys)
        ax.set_xticks(np.arange(len(labels)))
        long = max(len(k) for k in labels) > 6
        ax.set_xticklabels(labels, rotation=45 if long else 0, ha="right" if long else "center")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        self.canvas.draw_idle()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Formula builder: operators are typed, every operand is picked from a list
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\W\d]\w*(?:\.\w+)?|\d+(?:\.\d*)?|\.\d+|\*\*|[-+*/()]")
_OP_KEYS = {Qt.Key_Plus: "+", Qt.Key_Minus: "-", Qt.Key_Asterisk: "*", Qt.Key_Slash: "/",
            Qt.Key_ParenLeft: "(", Qt.Key_ParenRight: ")"}


def tokenize(expr: str) -> List[str]:
    return _TOKEN_RE.findall(expr or "")


class FormulaDialog(QtWidgets.QDialog):
    """Edit a formula without typing names.

    ``terms`` is ``[(category, [(label, token), ...]), ...]``; ``variables`` is
    used for the live preview.
    """

    def __init__(self, parent, title: str, terms, variables, formula: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(620)
        self.terms = [(c, items) for c, items in terms if items]
        self.variables = variables
        formula = formula or {}
        self.tokens: List[str] = tokenize(str(formula.get("expr", "")))

        lay = QtWidgets.QVBoxLayout(self)
        f = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(str(formula.get("name", "")))
        f.addRow("이름", self.name)
        lay.addLayout(f)

        self.display = QtWidgets.QLineEdit()
        self.display.setReadOnly(True)
        font = self.display.font()
        font.setPointSizeF(font.pointSizeF() * 1.3)
        self.display.setFont(font)
        self.display.setPlaceholderText("아래에서 항목을 넣고, + - * / ( ) 는 키보드나 버튼으로 입력하세요")
        self.display.installEventFilter(self)
        lay.addWidget(self.display)

        hb = QtWidgets.QHBoxLayout()
        for label, tok in (("+", "+"), ("−", "-"), ("×", "*"), ("÷", "/"), ("(", "("), (")", ")")):
            b = QtWidgets.QPushButton(label)
            b.setFixedWidth(44)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=None, t=tok: self.push(t))
            hb.addWidget(b)
        hb.addSpacing(12)
        b = QtWidgets.QPushButton("⌫ 지우기")
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(self.pop)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("모두 지우기")
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(self.clear)
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)

        g = QtWidgets.QGroupBox("항목 넣기")
        gl = QtWidgets.QHBoxLayout(g)
        self.category = QtWidgets.QComboBox()
        for c, _ in self.terms:
            self.category.addItem(c)
        self.category.currentIndexChanged.connect(self.fill_terms)
        gl.addWidget(self.category)
        self.term = QtWidgets.QComboBox()
        self.term.setMinimumWidth(220)
        gl.addWidget(self.term, 1)
        b = QtWidgets.QPushButton("넣기")
        b.setDefault(False)
        b.setAutoDefault(False)
        b.clicked.connect(self.insert_term)
        gl.addWidget(b)
        lay.addWidget(g)

        self.normalize = QtWidgets.QCheckBox("기준 레인을 1로 정규화")
        self.normalize.setChecked(formula.get("normalize", "1") == "1")
        lay.addWidget(self.normalize)
        self.preview = QtWidgets.QLabel()
        self.preview.setWordWrap(True)
        lay.addWidget(self.preview)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        for btn in bb.buttons():
            btn.setAutoDefault(False)
        self.ok = bb.button(QtWidgets.QDialogButtonBox.Ok)
        lay.addWidget(bb)
        self.fill_terms()
        self.refresh()
        self.display.setFocus()

    def fill_terms(self, *_):
        self.term.clear()
        i = self.category.currentIndex()
        if 0 <= i < len(self.terms):
            for label, tok in self.terms[i][1]:
                self.term.addItem(label if label == tok else f"{label}   ({tok})", tok)

    def insert_term(self):
        tok = self.term.currentData()
        if tok:
            self.push(tok)

    def push(self, tok: str):
        last = self.tokens[-1] if self.tokens else None
        # a digit typed after a number extends it
        if last is not None and re.fullmatch(r"[\d.]+", tok) and re.fullmatch(r"\d*\.?\d*", last) and last:
            self.tokens[-1] = last + tok
        else:
            self.tokens.append(tok)
        self.refresh()

    def pop(self):
        if self.tokens:
            last = self.tokens.pop()
            if re.fullmatch(r"\d*\.?\d*", last) and len(last) > 1:
                self.tokens.append(last[:-1])
        self.refresh()

    def clear(self):
        self.tokens = []
        self.refresh()

    def eventFilter(self, obj, e):
        if obj is self.display and e.type() == QtCore.QEvent.KeyPress:
            key, text = e.key(), e.text()
            if key in _OP_KEYS or text in "+-*/()" and text:
                self.push(_OP_KEYS.get(key, text))
                return True
            if text and (text.isdigit() or text == "."):
                self.push(text)
                return True
            if key in (Qt.Key_Backspace, Qt.Key_Delete):
                self.pop()
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                if self.ok.isEnabled():
                    self.accept()
                return True
            return True  # letters are not typed: pick names from the list
        return super().eventFilter(obj, e)

    def expr(self) -> str:
        out = ""
        for t in self.tokens:
            if t in "+-*/" or t == "**":
                out += f" {t} "
            else:
                out += t
        return out.strip()

    def refresh(self):
        expr = self.expr()
        self.display.setText(expr)
        if not expr:
            self.preview.setText("")
            self.ok.setEnabled(False)
            return
        try:
            val = core.evaluate(expr, self.variables)
            vals = np.atleast_1d(np.asarray(val, float))
            shown = ", ".join("—" if not np.isfinite(v) else f"{v:.4g}" for v in vals[:10]) + (" …" if vals.size > 10 else "")
            self.preview.setText(f"미리보기: {shown}")
            self.preview.setStyleSheet("color: #2e7d32;")
            self.ok.setEnabled(True)
        except Exception as e:
            self.preview.setText(f"미완성/오류: {e}")
            self.preview.setStyleSheet("color: #c0392b;")
            self.ok.setEnabled(False)

    def result(self) -> dict:
        return {"name": self.name.text().strip() or self.expr(), "expr": self.expr(),
                "normalize": "1" if self.normalize.isChecked() else "0"}


RAW_NOTE = ("※ 정량은 항상 원본 TIF의 raw intensity로 합니다. 표시 대비와 검출용 LUT는 정량에 쓰지 않고, "
            "어두운 밴드 이미지는 선형 반전(최댓값 − 값)만, 그리고 선택한 배경 제거만 적용합니다.")


def raw_note() -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(RAW_NOTE)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: #1565c0; font-size: 11px;")
    return lab


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
        self.figure: Optional[dict] = None  # Figure 만들기 settings (saved with the analysis)
        self.marker: Optional[np.ndarray] = None  # ladder/marker image (not saved, path is)

    @property
    def name(self) -> str:
        return os.path.basename(self.image_path) if self.image_path else "(빈 이미지)"


def _doc_attr(name):
    return property(lambda self: getattr(self.doc, name), lambda self, v: setattr(self.doc, name, v))


KIND_PROTEIN = "protein"
KIND_LANE = "lane"
KIND_LABELS = {KIND_PROTEIN: "단백질", KIND_LANE: "레인"}
VALUE_LABELS = {"ratio": "정규화 비율 (Target ÷ Control)", "net": "Net intensity"}


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
        self.results: Dict[str, np.ndarray] = {}  # current image
        # several images: groups made of other images' groups, formulas across images
        self.big_groups: List[Dict[str, object]] = []  # {name, kind, members: [[image, group]], value}
        self.multi_formulas: List[Dict[str, str]] = []  # {name, expr, normalize}
        self.multi_ref: Optional[str] = None  # lane key scaled to 1
        self.multi_keys: List[str] = []
        self.multi_results: Dict[str, Dict[str, object]] = {}  # name -> {mean, sd, points}
        self._updating = False

        self.view = BlotView()
        self.view.changed.connect(self.on_boxes_changed)
        self.view.selectionChangedBands.connect(self.on_view_selection)
        self.view.contextRequested.connect(self.show_band_menu)
        self.view.addRequested.connect(self.add_band_at)
        self.view.filesDropped.connect(self.open_paths)
        self.view.resizeModeExited.connect(lambda: self.resize_mode.setChecked(False))

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_detect_tab(), "1. 검출")
        self.tabs.addTab(self._build_rows_tab(), "2. 단백질 / 레인")
        self.tabs.addTab(self._build_bands_tab(), "3. 밴드")
        self.tabs.addTab(self._build_results_tab(), "4. 정량 / 그래프")
        self.tabs.addTab(self._build_multi_tab(), "5. 여러 이미지")
        from .figure_window import FigureEditor

        self.figure_editor = FigureEditor(self)
        self.FIG_TAB = self.tabs.addTab(self.figure_editor, "6. Figure 만들기")
        self.tabs.tabBar().setTabTextColor(self.FIG_TAB, QtGui.QColor("#1565c0"))
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self._split_sizes = None
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
        self.doc_tabs.setToolTip("열린 이미지 (수식에서는 I1, I2 …로 부릅니다)")
        self.doc_tabs.currentChanged.connect(self.switch_doc)
        self.doc_tabs.tabCloseRequested.connect(self.close_doc)
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
        self.show_marker = QtWidgets.QCheckBox("마커 같이 보기")
        self.show_marker.setChecked(True)
        self.show_marker.toggled.connect(lambda _=None: (self.refresh_marker_view(), self.view.fit()))
        self.resize_mode = QtWidgets.QCheckBox("박스 크기 조절 모드")
        self.resize_mode.setToolTip(
            "켜면 박스 가장자리·모서리를 드래그해 크기를 바꿉니다. 같은 단백질 그룹(행)과 함께 선택한 밴드의 행이 함께 바뀝니다.\n"
            "우클릭 메뉴에서도 켜고 끌 수 있고, Esc로 끝냅니다."
        )
        self.resize_mode.toggled.connect(self.on_resize_mode)
        cl = QtWidgets.QHBoxLayout()
        cl.addWidget(QtWidgets.QLabel("표시 대비"))
        cl.addWidget(self.contrast_lo)
        cl.addWidget(self.contrast_hi)
        cl.addWidget(self.show_corrected)
        cl.addWidget(self.show_labels)
        cl.addWidget(self.show_marker)
        cl.addWidget(self.resize_mode)
        ll.addLayout(cl)
        hint = QtWidgets.QLabel(
            "밴드를 드래그로 선택 → 우클릭: 새 단백질(행) 그룹 / 새 레인(열) 그룹 만들기 · 더블클릭: 밴드 추가 · "
            "방향키: 미세 이동 · Delete: 삭제 · ⌘/Ctrl+휠: 확대"
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        ll.addWidget(hint)

        split = QtWidgets.QSplitter()
        self.split = split
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
            self.refresh_marker_view()
            self.update_display(reset_zoom=True)
        else:
            self.view.set_marker(None)
            self.marker_label.setText("없음")
            self.file_label.setText("-")
            self.setWindowTitle("Blot Profiler")
            if self.view.pix_item is not None:
                self.view.set_image(QtGui.QImage(), reset_zoom=False)
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
        for g in self.big_groups:
            g["members"] = [[i - 1 if i > k else i, n] for i, n in g["members"] if i != k]
        if not self.docs:
            self.docs.append(ImageDoc())
        self.cur = min(self.cur if self.cur < k else max(0, self.cur - 1), len(self.docs) - 1)
        self.refresh_doc_tabs()
        self.switch_doc(self.cur)

    def on_resize_mode(self, on):
        self.view.set_resize_mode(on)
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.statusBar().showMessage(
            "박스 크기 조절 모드: 박스 가장자리/모서리를 드래그하세요 (같은 단백질 그룹이 함께 바뀜). Esc 또는 우클릭 메뉴로 종료"
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
        m.addAction("밴드 ROI·그룹 저장…", self.save_roi)
        m.addAction("밴드 ROI·그룹 불러오기 (현재 이미지에 적용)…", self.load_roi)
        m.addSeparator()
        m.addAction("결과 CSV 내보내기…", self.export_csv)
        m.addAction("그래프 저장 (PNG/PDF/SVG)…", lambda: self.export_plot())
        m.addAction("박스 표시 이미지 저장…", self.export_overlay)
        fm = self.menuBar().addMenu("Figure")
        fm.addAction("Figure 만들기 탭으로 (논문용 그림)", self.open_figure_window).setShortcut("Ctrl+Shift+F")
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
        hb = QtWidgets.QHBoxLayout()
        self.marker_label = QtWidgets.QLabel("없음")
        self.marker_label.setWordWrap(True)
        hb.addWidget(self.marker_label, 1)
        b = QtWidgets.QPushButton("마커 열기…")
        b.setToolTip("현재 블롯과 같은 위치에서 찍은 마커(ladder, white-light) 이미지")
        b.clicked.connect(self.open_marker)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("빼기")
        b.clicked.connect(self.clear_marker)
        hb.addWidget(b)
        f.addRow("마커", hb)
        self.ask_marker = QtWidgets.QCheckBox("블롯 TIF를 열면 짝이 되는 마커 이미지도 받기")
        self.ask_marker.setToolTip("같은 폴더에 '…Marker…' 파일이 있으면 자동으로 붙이고, 없으면 마커 파일을 고르는 창을 띄웁니다 (취소 = 건너뛰기)")
        self.ask_marker.setChecked(True)
        f.addRow(self.ask_marker)
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
        self.lut_gamma = _spin(0.2, 1.0, 0.6, 0.05, decimals=2)
        self.lut_gamma.setToolTip("밴드를 찾을 때만 쓰는 대비 조정(LUT) 감마. 1보다 작을수록 흐린 밴드를 더 잘 잡습니다. 1 = 조정 안 함.\n"
                                  "정량에는 쓰이지 않습니다.")
        f.addRow("검출용 대비(LUT) 감마", self.lut_gamma)
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
        f.addRow(raw_note())
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

        g_roi = QtWidgets.QGroupBox("밴드 ROI · 그룹")
        gl = QtWidgets.QHBoxLayout(g_roi)
        b = QtWidgets.QPushButton("ROI·그룹 저장…")
        b.setToolTip("현재 이미지의 박스 위치·크기, 단백질/레인 그룹 이름과 역할을 파일로 저장합니다")
        b.clicked.connect(self.save_roi)
        gl.addWidget(b)
        b = QtWidgets.QPushButton("ROI·그룹 불러오기…")
        b.setToolTip("저장한 ROI·그룹을 현재 이미지에 적용합니다 (다른 이미지에도 쓸 수 있습니다)")
        b.clicked.connect(self.load_roi)
        gl.addWidget(b)
        lay.addWidget(g_roi)
        lay.addStretch(1)
        self.on_bg_changed(recompute=False)
        return w

    # ---------------------------------------------------------------- tab 2
    def _build_rows_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        info = QtWidgets.QLabel(
            "검출하면 단백질 그룹(P1, P2 …)과 레인 그룹(L1, L2 …)이 위치에 따라 자동으로 제안됩니다.\n"
            "정량 없이 바로 '6. Figure 만들기'로 가도 되고, 고치려면 밴드를 드래그로 선택한 뒤 우클릭 →\n"
            "  · '새 단백질(행) 그룹 만들기' — 이름이 곧 단백질 이름입니다 (예: Vinculin, pErk).\n"
            "  · '새 레인(열) 그룹 만들기' — 이름이 곧 레인 이름입니다 (예: DMSO, IgG 12h).\n"
            "    같은 단백질의 밴드를 여러 열에서 고르면 그 열들이 한 레인 그룹(반복)이 됩니다.\n"
            "밴드 ID는 '단백질_레인'으로 자동으로 붙고, 이름이 겹치면 _2, _3이 붙습니다.\n"
            "한 단백질 그룹의 박스는 모두 같은 크기이고, 같은 레인 안의 밴드는 합해서 정량합니다."
        )
        info.setStyleSheet("color: #555;")
        lay.addWidget(info)
        self.rows_table = QtWidgets.QTableWidget(0, 4)
        self.rows_table.setHorizontalHeaderLabels(["단백질 그룹", "역할", "박스 폭", "박스 높이"])
        self.rows_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.rows_table.verticalHeader().setVisible(False)
        self.rows_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.rows_table.itemSelectionChanged.connect(self.on_row_selected)
        lay.addWidget(self.rows_table, 2)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("선택한 단백질 그룹 삭제")
        b.clicked.connect(self.delete_row)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("박스 높이를 밴드에 맞춤")
        b.clicked.connect(self.autofit_rows)
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)

        self.auto_lanes = QtWidgets.QCheckBox("박스 위치·크기를 바꿀 때마다 위치(왼쪽→오른쪽) 순서로 레인 자동 갱신")
        self.auto_lanes.setChecked(True)
        self.auto_lanes.setToolTip("이름을 붙인 레인 그룹은 유지되고, 나머지 레인은 L1, L2 … 순서로 다시 붙습니다.\n"
                                   "레인은 언제든 '새 레인(열) 그룹 만들기'로 바꿀 수 있습니다.")
        self.auto_lanes.toggled.connect(lambda on: on and self.apply_auto_lanes())
        lay.addWidget(self.auto_lanes)
        self.lanes_table = QtWidgets.QTableWidget(0, 2)
        self.lanes_table.setHorizontalHeaderLabels(["레인 (왼쪽부터)", "레인 그룹"])
        self.lanes_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.lanes_table.verticalHeader().setVisible(False)
        self.lanes_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.lanes_table.itemSelectionChanged.connect(self.on_lane_selected)
        lay.addWidget(self.lanes_table, 2)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("선택한 레인 삭제")
        b.clicked.connect(self.delete_lane)
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)
        return w

    # ---------------------------------------------------------------- tab 3
    def _build_bands_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.bands_table = QtWidgets.QTableWidget(0, 7)
        self.bands_table.setHorizontalHeaderLabels(["밴드 ID", "단백질", "레인 그룹", "Raw", "배경", "Net (정량값)", "Mean/px"])
        self.bands_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.bands_table.verticalHeader().setVisible(False)
        self.bands_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.bands_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.bands_table.itemSelectionChanged.connect(self.on_table_selection)
        lay.addWidget(self.bands_table, 1)
        lay.addWidget(raw_note())
        self.profile = PlotCanvas()
        self.profile.toolbar.hide()
        self.profile.setMaximumHeight(170)
        lay.addWidget(QtWidgets.QLabel("선택한 밴드의 세로 intensity 프로파일 (박스 영역 음영)"))
        lay.addWidget(self.profile)
        return w

    def _scroll(self, w):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(w)
        return scroll

    def _formula_box(self, title, hint):
        g = QtWidgets.QGroupBox(title)
        gl = QtWidgets.QVBoxLayout(g)
        t = QtWidgets.QTableWidget(0, 3)
        t.setHorizontalHeaderLabels(["이름", "수식", "기준 레인 정규화"])
        t.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        t.setMinimumHeight(100)
        t.setMaximumHeight(150)
        gl.addWidget(t)
        hb = QtWidgets.QHBoxLayout()
        gl.addLayout(hb)
        msg = QtWidgets.QLabel(hint)
        msg.setStyleSheet("color: #888;")
        msg.setWordWrap(True)
        gl.addWidget(msg)
        return g, t, hb, msg

    # ---------------------------------------------------------------- tab 4
    def _build_results_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.results_title = QtWidgets.QLabel()
        self.results_title.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.results_title)
        lay.addWidget(raw_note())

        g = QtWidgets.QGroupBox("Internal control 정규화")
        f = QtWidgets.QFormLayout(g)
        self.auto_norm = QtWidgets.QCheckBox("측정(Target) 단백질 ÷ Control 단백질 (레인별) 자동 계산")
        self.auto_norm.setChecked(True)
        self.auto_norm.toggled.connect(self.recompute_results)
        f.addRow(self.auto_norm)
        self.ref_lane = QtWidgets.QComboBox()
        self.ref_lane.currentIndexChanged.connect(self.on_ref_lane)
        f.addRow("기준 레인 (=1.0)", self.ref_lane)
        lay.addWidget(g)

        g, self.formula_table, hb, self.formula_msg = self._formula_box(
            "사용자 수식 (단백질 그룹 / 밴드 간 더하기·빼기·곱하기·나누기)", FORMULA_HINT)
        self.formula_table.itemChanged.connect(self.on_formula_changed)
        self.formula_table.cellDoubleClicked.connect(lambda r, c: self.edit_formula(r) if c == 1 else None)
        for text, fn in (("수식 추가…", self.add_formula), ("수식 편집…", self.edit_formula), ("수식 삭제", self.delete_formula)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        hb.addStretch(1)
        lay.addWidget(g)

        hb = QtWidgets.QHBoxLayout()
        self.by_condition = QtWidgets.QCheckBox("같은 레인 그룹 평균 ± SD")
        self.by_condition.setToolTip("여러 열이 한 레인 그룹(반복)일 때 평균과 SD로 그립니다")
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
        b.clicked.connect(lambda: self.export_plot(self.plot))
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)
        return self._scroll(w)

    # ---------------------------------------------------------------- figure window
    def open_figure_window(self):
        self.tabs.setCurrentIndex(self.FIG_TAB)

    def on_tab_changed(self, i):
        """The figure tab gets most of the window width; the blot view shrinks."""
        if i == self.FIG_TAB:
            if self._split_sizes is None:
                self._split_sizes = self.split.sizes()
            total = sum(self.split.sizes())
            self.split.setSizes([int(total * 0.28), total - int(total * 0.28)])
        elif self._split_sizes is not None:
            self.split.setSizes(self._split_sizes)
            self._split_sizes = None
        else:
            return
        QtCore.QTimer.singleShot(0, self.view.fit)

    def _figure_changed(self, d):
        ed = self.figure_editor
        if self.tabs.currentIndex() == self.FIG_TAB and ed.isVisible():
            ed.set_doc(self.doc)
        else:
            ed.dirty = True

    # ---------------------------------------------------------------- tab 5
    def _build_multi_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        info = QtWidgets.QLabel(
            "열린 모든 이미지의 그룹입니다. 같은 종류(단백질 또는 레인)의 그룹을 체크하고 '큰 그룹 만들기'를 누르세요.\n"
            "  · 레인 큰 그룹: 이미지마다 이름이 다른 레인을 한 레인으로 맞춥니다 (예: I1의 'IgG 12h' + I2의 'IgG12').\n"
            "  · 단백질 큰 그룹: 여러 이미지의 단백질 값을 레인별로 평균 ± SD 합니다 (예: 반복 블롯의 Kras).\n"
            "이미지 사이에서 레인은 레인 그룹 이름(또는 레인 큰 그룹)으로 맞춰집니다."
        )
        info.setStyleSheet("color: #555;")
        lay.addWidget(info)
        self.group_tree = QtWidgets.QTreeWidget()
        self.group_tree.setHeaderLabels(["이미지 / 그룹", "종류"])
        self.group_tree.setMinimumHeight(150)
        self.group_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        lay.addWidget(self.group_tree, 1)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("체크한 그룹으로 큰 그룹 만들기…")
        b.setStyleSheet("font-weight: bold;")
        b.clicked.connect(self.make_big_group)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("선택한 큰 그룹 삭제")
        b.clicked.connect(self.delete_big_group)
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)
        self.big_table = QtWidgets.QTableWidget(0, 4)
        self.big_table.setHorizontalHeaderLabels(["큰 그룹", "종류", "구성", "값"])
        self.big_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.big_table.verticalHeader().setVisible(False)
        self.big_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.big_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.big_table.setMinimumHeight(110)
        self.big_table.setMaximumHeight(170)
        lay.addWidget(self.big_table)

        f = QtWidgets.QFormLayout()
        self.multi_ref_combo = QtWidgets.QComboBox()
        self.multi_ref_combo.setToolTip("단백질 큰 그룹은 이미지마다 이 레인을 1로 맞춘 뒤 평균합니다 (블롯 간 노출 차이 보정)")
        self.multi_ref_combo.currentIndexChanged.connect(self.on_multi_ref)
        f.addRow("기준 레인 (=1.0)", self.multi_ref_combo)
        lay.addLayout(f)

        g, self.multi_formula_table, hb, self.multi_msg = self._formula_box(
            "이미지 간 수식", MULTI_FORMULA_HINT)
        self.multi_formula_table.itemChanged.connect(self.on_multi_formula_changed)
        self.multi_formula_table.cellDoubleClicked.connect(lambda r, c: self.edit_multi_formula(r) if c == 1 else None)
        for text, fn in (("수식 추가…", self.add_multi_formula), ("수식 편집…", self.edit_multi_formula),
                         ("수식 삭제", self.delete_multi_formula)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        hb.addStretch(1)
        lay.addWidget(g)

        hb = QtWidgets.QHBoxLayout()
        self.multi_points = QtWidgets.QCheckBox("이미지별 값 점 표시")
        self.multi_points.setChecked(True)
        self.multi_points.toggled.connect(self.replot_multi)
        hb.addWidget(self.multi_points)
        hb.addWidget(QtWidgets.QLabel("그래프 항목:"))
        self.multi_series = QtWidgets.QListWidget()
        self.multi_series.setMaximumHeight(60)
        self.multi_series.setFlow(QtWidgets.QListView.LeftToRight)
        self.multi_series.setWrapping(True)
        self.multi_series.itemChanged.connect(self.replot_multi)
        hb.addWidget(self.multi_series, 1)
        lay.addLayout(hb)
        self.multi_plot = PlotCanvas(aspect=0.75)
        lay.addWidget(self.multi_plot)
        self.multi_table = QtWidgets.QTableWidget(0, 0)
        self.multi_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.multi_table.setMinimumHeight(200)
        lay.addWidget(self.multi_table, 1)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("CSV 내보내기…")
        b.clicked.connect(self.export_csv)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("그래프 저장…")
        b.clicked.connect(lambda: self.export_plot(self.multi_plot))
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)
        return self._scroll(w)

    # ---------------------------------------------------------------- image
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.open_paths([u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()])

    @staticmethod
    def is_marker_file(p: str) -> bool:
        return "marker" in os.path.basename(p).lower()

    def open_path(self, p: str):
        self.open_paths([p])

    def open_paths(self, paths):
        """Open analysis files and blots; files named '…marker…' become the marker of their blot
        (or of the current image when no blot is opened with them)."""
        paths = [p for p in paths if p]
        markers = [p for p in paths if self.is_marker_file(p) and not p.lower().endswith(".json")]
        for p in paths:
            if p.lower().endswith(".json"):
                self.load_project(p)
        blots = [p for p in paths if p not in markers and not p.lower().endswith(".json")]
        for p in blots:
            self.load_image(p, marker_candidates=markers)
        if markers and not blots:
            if self.doc.raw is None:
                for p in markers:  # no blot open: treat it as an image after all
                    self.load_image(p, ask_marker=False)
            else:
                self.set_marker(self.doc, markers[0])

    def open_image(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "TIF 파일 열기 (여러 개 선택 가능)", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        self.open_paths(paths)

    # ---------------------------------------------------------------- marker (ladder) image
    @staticmethod
    def find_marker_for(path: str, candidates=()) -> Optional[str]:
        """The marker file belonging to blot ``path``: among ``candidates`` (opened with it)
        or files in its folder whose name contains 'marker' and otherwise matches the blot's."""
        import difflib

        def stem(p):
            s_ = os.path.splitext(os.path.basename(p))[0].lower()
            return re.sub(r"^[0-9a-f]{6,}[-_]", "", s_)  # ignore a leading random id (copied/uploaded files)

        def base(p):
            return re.sub(r"[-_ .]?marker", "", stem(p), flags=re.I)

        folder = os.path.dirname(os.path.abspath(path))
        try:
            local = [os.path.join(folder, f) for f in os.listdir(folder)]
        except OSError:
            local = []
        exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        pool = [p for p in list(candidates) + local
                if p.lower().endswith(exts) and "marker" in os.path.basename(p).lower()
                and os.path.abspath(p) != os.path.abspath(path)]
        if not pool:
            return None
        b = stem(path)
        scored = sorted(((difflib.SequenceMatcher(None, base(p), b).ratio(), p) for p in pool), reverse=True)
        best_score, best = scored[0]
        if best in candidates or best_score >= 0.8:
            return best
        return None

    def ask_marker_for(self, d: ImageDoc, candidates=()):
        path = self.find_marker_for(d.image_path, candidates)
        if path:
            if self.set_marker(d, path):
                self.statusBar().showMessage(f"마커 이미지를 자동으로 붙였습니다: {os.path.basename(path)}")
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"'{os.path.basename(d.image_path)}'의 마커(ladder) 이미지를 고르세요 — 없으면 취소",
            os.path.dirname(d.image_path), "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if p:
            self.set_marker(d, p)

    def open_marker(self):
        if self.doc.raw is None:
            self.statusBar().showMessage("먼저 블롯 TIF를 여세요")
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "마커(ladder) 이미지 열기", os.path.dirname(self.doc.image_path or ""),
                                                     "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if p:
            self.set_marker(self.doc, p)

    def set_marker(self, d: ImageDoc, path: str, quiet: bool = False) -> bool:
        """Attach a marker image (same geometry as the blot) to image ``d``."""
        try:
            m = core.load_image(path)
            resized = m.shape != d.raw.shape
            if resized:
                m = figure.match_marker(m, d.raw.shape)
        except Exception as e:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "마커 이미지", f"쓸 수 없는 마커 이미지입니다:\n{e}")
            return False
        if resized and not quiet:
            QtWidgets.QMessageBox.information(self, "마커 이미지 크기", "마커 이미지의 크기가 블롯과 달라 같은 비율로 맞췄습니다. "
                                              "두 이미지가 같은 위치에서 찍힌 것인지 확인하세요.")
        d.marker = m
        if d.figure is None:
            d.figure = figure.default_settings()
        s = d.figure
        s["marker_path"] = path
        if not quiet:
            s["marker_invert"] = not core.needs_inversion(m)
            if d.analysis.bands:
                s["strip_x0"], s["strip_x1"] = figure.guess_ladder_x(m, d.analysis)
            s["show_strip"] = True
        if d is self.doc:
            self.refresh_marker_view()
            self.view.fit()
            self._figure_changed(d)
        return True

    def clear_marker(self):
        d = self.doc
        d.marker = None
        if d.figure:
            d.figure["marker_path"] = None
            d.figure["show_strip"] = False
        self.refresh_marker_view()
        self.view.fit()
        self._figure_changed(d)

    def refresh_marker_view(self):
        d = self.doc
        m = d.marker
        if m is None and d.figure and d.figure.get("marker_path") and d.raw is not None and os.path.exists(d.figure["marker_path"]):
            self.set_marker(d, d.figure["marker_path"], quiet=True)
            return
        self.marker_label.setText(os.path.basename(d.figure["marker_path"]) if m is not None and d.figure else "없음")
        if m is None or not self.show_marker.isChecked():
            self.view.set_marker(None)
            return
        x = figure._stretch(m, 1, 99.5, 1.0)
        if (d.figure or {}).get("marker_invert", True):
            x = 1.0 - x  # dark ladder on white, like the blot preview
        self.view.set_marker(to_qimage(x, 0.0, 1.0))

    def load_image(self, path: str, detect=True, doc: Optional[ImageDoc] = None, ask_marker: bool = True,
                   marker_candidates=()):
        """Open ``path`` in ``doc`` (default: a new tab, or the current one if it is empty).

        After detection the matching marker image is attached (found next to the
        blot or among ``marker_candidates``) or asked for.
        """
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
        d.marker = None
        d.figure = None
        self._make_signal(d)
        self.refresh_doc_tabs()
        self.switch_doc(self.docs.index(d))
        if detect:
            self.detect()
        if ask_marker and self.ask_marker.isChecked():
            self.ask_marker_for(d, marker_candidates)

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
            lut_gamma=self.lut_gamma.value(),
        )
        old = self.analysis
        names = [r.name for r in old.rows]
        a = core.detect_bands(corr, settings, names if names else None)
        # keep the user's protein/lane group names, roles and formulas when re-detecting
        for i, r in enumerate(a.rows):
            if i < len(old.rows):
                r.role = old.rows[i].role
        if old.lane_labels and len(old.lane_labels) == a.n_lanes:
            a.lane_labels = list(old.lane_labels)
        a.formulas = list(old.formulas)
        if a.rows and not any(r.role == core.ROLE_CONTROL for r in a.rows):
            a.rows[0].role = core.ROLE_CONTROL  # loading control is usually the top row; user can change
        self.set_analysis(a)
        self.statusBar().showMessage(
            f"자동 그룹 제안: 단백질 {len(a.rows)}개({', '.join(r.name for r in a.rows)}), 레인 {a.n_lanes}개, 박스 {len(a.bands)}개. "
            "바로 '6. Figure 만들기'로 가거나, 밴드를 드래그로 선택 → 우클릭으로 그룹 이름·구성을 고치세요."
        )

    def set_analysis(self, a: core.Analysis):
        self.analysis = a
        a.ensure_lane_labels()
        self.view.set_analysis(a)
        self.refresh_all()
        self._figure_changed(self.doc)

    def refresh_all(self):
        self.fill_rows_table()
        self.fill_lanes_table()
        self.fill_formulas()
        self.remeasure()

    def structure_changed(self):
        """Bands, protein or lane groups changed: redraw boxes and every table."""
        self.view.rebuild()
        self.fill_rows_table()
        self.fill_lanes_table()
        self.remeasure()
        self._figure_changed(self.doc)

    def apply_auto_lanes(self) -> bool:
        if not self.auto_lanes.isChecked() or not core.auto_assign_lanes(self.analysis):
            return False
        self.view.refresh_items()
        self.fill_lanes_table()
        return True

    def on_boxes_changed(self):
        # every position change (also while dragging) re-derives the lanes
        if self.apply_auto_lanes():
            self.statusBar().showMessage("박스 위치에 맞춰 레인을 다시 지정했습니다 (이름 붙인 레인 그룹은 유지)")
        # box sizes may have been changed by dragging in resize mode
        for i, r in enumerate(self.analysis.rows):
            for col, attr in ((2, "width"), (3, "height")):
                sp = self.rows_table.cellWidget(i, col)
                if sp is not None and sp.value() != getattr(r, attr):
                    sp.blockSignals(True)
                    sp.setValue(getattr(r, attr))
                    sp.blockSignals(False)
        self.remeasure()
        self._figure_changed(self.doc)

    def add_band_at(self, x, y):
        core.add_band(self.analysis, x, y)
        if self.auto_lanes.isChecked():
            core.auto_assign_lanes(self.analysis)
        self.structure_changed()

    def delete_selected(self):
        sel = self.view.selected_bands()
        if not sel:
            return
        ids = {id(b) for b in sel}
        self.analysis.bands = [b for b in self.analysis.bands if id(b) not in ids]
        core.remove_empty(self.analysis)
        self.structure_changed()

    def fill_missing(self):
        n = core.fill_missing_lanes(self.analysis)
        self.structure_changed()
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
        self.set_analysis(core.Analysis(formulas=list(self.analysis.formulas)))

    def show_band_menu(self, gpos, item):
        sel = self.view.selected_bands()
        menu = QtWidgets.QMenu(self)
        if sel:
            menu.addAction(f"새 단백질(행) 그룹 만들기… ({len(sel)}개 밴드)", self.make_protein_group)
            menu.addAction(f"새 레인(열) 그룹 만들기… ({len(sel)}개 밴드)", self.make_lane_group)
            rows = sorted({b.row for b in sel})
            if len(rows) == 1:
                ro = menu.addMenu(f"'{self.analysis.rows[rows[0]].name}' 역할")
                for role in core.ROLES:
                    ro.addAction(ROLE_LABELS[role], lambda role=role, r=rows[0]: self.set_row_role(r, role))
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

    # ---------------------------------------------------------------- protein / lane groups
    def _ask_name(self, title, label, default):
        name, ok = QtWidgets.QInputDialog.getText(self, title, label, text=default)
        return name.strip() if ok and name.strip() else None

    def make_protein_group(self):
        sel = self.view.selected_bands()
        if not sel:
            self.statusBar().showMessage("이미지에서 밴드를 먼저 드래그로 선택하세요")
            return
        rows = {b.row for b in sel}
        default = self.analysis.rows[min(rows)].name if len(rows) == 1 else ""
        name = self._ask_name("새 단백질(행) 그룹", "단백질 이름 (예: Vinculin, pErk, Kras G12D):", default)
        if not name:
            return
        final = core.make_row_group(self.analysis, sel, name)
        self.structure_changed()
        self.view.select_bands(sel)
        note = f" ('{name}'이(가) 이미 있어 '{final}'(으)로 붙였습니다)" if final != name else ""
        self.statusBar().showMessage(f"단백질 그룹 '{final}': 밴드 {len(sel)}개{note}")

    def make_lane_group(self):
        sel = self.view.selected_bands()
        if not sel:
            self.statusBar().showMessage("이미지에서 밴드를 먼저 드래그로 선택하세요")
            return
        a = self.analysis
        lanes = {b.lane for b in sel}
        default = a.lane_labels[min(lanes)] if len(lanes) == 1 else ""
        name = self._ask_name("새 레인(열) 그룹", "레인 이름 (예: DMSO, IgG 12h):", default)
        if not name:
            return
        final = core.make_lane_group(a, sel, name)
        self.structure_changed()
        self.view.select_bands(sel)
        n = a.lane_labels.count(final)
        note = f" ('{name}'이(가) 이미 있어 '{final}'(으)로 붙였습니다)" if final != name else ""
        self.statusBar().showMessage(f"레인 그룹 '{final}': {n}개 열{' (반복)' if n > 1 else ''}{note}")

    # ---------------------------------------------------------------- tables: proteins / lanes
    def fill_rows_table(self):
        t = self.rows_table
        t.blockSignals(True)
        t.setRowCount(len(self.analysis.rows))
        for i, r in enumerate(self.analysis.rows):
            it = QtWidgets.QTableWidgetItem(r.name)
            it.setForeground(row_color(i))
            t.setItem(i, 0, it)
            cb = QtWidgets.QComboBox()
            for role in core.ROLES:
                cb.addItem(ROLE_LABELS[role], role)
            cb.setCurrentIndex(core.ROLES.index(r.role))
            cb.currentIndexChanged.connect(lambda _=None, i=i, cb=cb: self.set_row_role(i, cb.currentData()))
            t.setCellWidget(i, 1, cb)
            for col, attr in ((2, "width"), (3, "height")):
                sp = _spin(2, 5000, getattr(r, attr), suffix=" px")
                sp.valueChanged.connect(lambda v, i=i, attr=attr: self.set_row_size(i, attr, v))
                t.setCellWidget(i, col, sp)
        t.blockSignals(False)
        t.resizeColumnToContents(1)

    def set_row_role(self, i, role):
        if i >= len(self.analysis.rows):
            return
        self.analysis.rows[i].role = role
        self.view.refresh_items()
        cb = self.rows_table.cellWidget(i, 1)
        if cb and cb.currentData() != role:
            cb.blockSignals(True)
            cb.setCurrentIndex(core.ROLES.index(role))
            cb.blockSignals(False)
        self.recompute_results()
        self.recompute_multi()

    def set_row_size(self, i, attr, v):
        setattr(self.analysis.rows[i], attr, int(v))
        self.view.refresh_items()
        self.apply_auto_lanes()
        self.remeasure()

    def on_row_selected(self):
        rows = {i.row() for i in self.rows_table.selectedIndexes()}
        if rows:
            self.view.select_bands([b for b in self.analysis.bands if b.row in rows])

    def on_lane_selected(self):
        lanes = {i.row() for i in self.lanes_table.selectedIndexes()}
        if lanes:
            self.view.select_bands([b for b in self.analysis.bands if b.lane in lanes])

    def delete_row(self):
        rows = sorted({i.row() for i in self.rows_table.selectedIndexes()})
        if not rows:
            rows = sorted({b.row for b in self.view.selected_bands()})
        for r in reversed(rows):
            self.analysis.remove_row(r)
        core.remove_empty(self.analysis)
        self.structure_changed()

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
        t.setRowCount(a.n_lanes)
        for i in range(a.n_lanes):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(a.lane_labels[i]))
        self.ref_lane.blockSignals(True)
        self.ref_lane.clear()
        self.ref_lane.addItem("정규화 안 함 (비율 그대로)", None)
        for i in range(a.n_lanes):
            self.ref_lane.addItem(f"{i + 1}: {a.lane_labels[i]}", i)
        idx = 0 if self.doc.ref_lane is None else self.ref_lane.findData(self.doc.ref_lane)
        self.ref_lane.setCurrentIndex(idx if idx >= 0 else min(1, self.ref_lane.count() - 1))
        self.ref_lane.blockSignals(False)
        self.doc.ref_lane = self.ref_lane.currentData()

    def on_ref_lane(self, *_):
        self.doc.ref_lane = self.ref_lane.currentData()
        self.recompute_results()
        self.recompute_multi()

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
        core.remove_empty(a)
        self.structure_changed()

    # ---------------------------------------------------------------- formulas (current image)
    @staticmethod
    def _fill_formula_table(t, formulas):
        t.blockSignals(True)
        t.setRowCount(len(formulas))
        for i, f in enumerate(formulas):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(str(f.get("name", ""))))
            it = QtWidgets.QTableWidgetItem(str(f.get("expr", "")))
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setToolTip("더블클릭: 수식 편집")
            t.setItem(i, 1, it)
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if f.get("normalize", "1") == "1" else Qt.Unchecked)
            t.setItem(i, 2, it)
        t.blockSignals(False)

    @staticmethod
    def _formula_edit(formulas, item):
        f = formulas[item.row()]
        if item.column() == 0:
            f["name"] = item.text()
        elif item.column() == 2:
            f["normalize"] = "1" if item.checkState() == Qt.Checked else "0"

    def fill_formulas(self):
        self._fill_formula_table(self.formula_table, self.analysis.formulas)

    def formula_terms(self):
        a = self.analysis
        ids = sorted({m.band_id for m in self.measurements})
        return [
            ("단백질 그룹 (레인별 값)", [(r.name, core._safe_name(r.name)) for r in a.rows]),
            ("밴드 (값 하나)", [(i, i) for i in ids]),
        ]

    def _edit_formula_dialog(self, title, terms, variables, formula):
        dlg = FormulaDialog(self, title, terms, variables, formula)
        return dlg.result() if dlg.exec() == QtWidgets.QDialog.Accepted else None

    def add_formula(self):
        a = self.analysis
        f = self._edit_formula_dialog("수식 추가", self.formula_terms(), core.formula_variables(a, self.measurements),
                                      {"name": f"수식{len(a.formulas) + 1}"})
        if f:
            a.formulas.append(f)
            self.fill_formulas()
            self.recompute_results()

    def edit_formula(self, row=None):
        a = self.analysis
        if row is None or row is False:
            rows = sorted({i.row() for i in self.formula_table.selectedIndexes()})
            row = rows[0] if rows else None
        if row is None or row >= len(a.formulas):
            self.statusBar().showMessage("편집할 수식을 표에서 고르세요")
            return
        f = self._edit_formula_dialog("수식 편집", self.formula_terms(), core.formula_variables(a, self.measurements), a.formulas[row])
        if f:
            a.formulas[row] = f
            self.fill_formulas()
            self.recompute_results()

    def delete_formula(self):
        for r in sorted({i.row() for i in self.formula_table.selectedIndexes()}, reverse=True):
            del self.analysis.formulas[r]
        self.fill_formulas()
        self.recompute_results()

    def on_formula_changed(self, item):
        self._formula_edit(self.analysis.formulas, item)
        self.recompute_results()

    # ---------------------------------------------------------------- measurement / results (current image)
    def measure_doc(self, d: ImageDoc):
        if d.signal is None or not d.analysis.rows:
            d.measurements = []
        else:
            d.measurements = core.measure(d.signal, d.analysis, d.bg, self.get_corrected(d))

    def remeasure(self):
        self.measure_doc(self.doc)
        self.fill_bands_table()
        self.recompute_results()
        self.refresh_multi()

    def fill_bands_table(self):
        a = self.analysis
        t = self.bands_table
        t.blockSignals(True)
        t.setRowCount(len(self.measurements))
        for i, m in enumerate(self.measurements):
            vals = [
                m.band_id, a.rows[m.row].name, a.lane_labels[m.lane] if m.lane < len(a.lane_labels) else str(m.lane + 1),
                f"{m.raw:.1f}", f"{m.background:.1f}", f"{m.net:.1f}", f"{m.mean:.2f}",
            ]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if c == 0:
                    it.setForeground(row_color(m.row))
                if c >= 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, c, it)
        t.blockSignals(False)

    def on_table_selection(self):
        sel = {self.bands_table.item(i.row(), 0).text() for i in self.bands_table.selectedIndexes() if self.bands_table.item(i.row(), 0)}
        ids = self.analysis.band_ids()
        bands = [b for b in self.analysis.bands if ids[id(b)] in sel]
        self.view.select_bands(bands)
        self.draw_profile(bands[:1])

    def on_view_selection(self):
        sel = self.view.selected_bands()
        self.draw_profile(sel[:1])
        if sel:
            ids = self.analysis.band_ids()
            self.statusBar().showMessage(f"선택: {', '.join(ids[id(b)] for b in sel[:12])}{' …' if len(sel) > 12 else ''}")

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
            bid = self.analysis.band_id(b)
            if self.doc.bg.method in ("local_border", "row_median"):
                m = next((m for m in self.measurements if m.band_id == bid), None)
                if m and m.area:
                    ax.axhline(m.background / m.area, color="r", ls="--", lw=0.8)
            ax.set_title(bid, fontsize=8)
            ax.tick_params(labelsize=7)
        self.profile.canvas.draw_idle()

    def loaded_docs(self) -> List[ImageDoc]:
        return [d for d in self.docs if d.raw is not None]

    def image_results(self, d: ImageDoc):
        """Results of one image: Target/Control ratios and its own formulas. Returns (results, messages)."""
        a = d.analysis
        res: Dict[str, np.ndarray] = {}
        msgs = []
        if not d.measurements:
            return res, msgs
        if self.auto_norm.isChecked():
            controls = [i for i, r in enumerate(a.rows) if r.role == core.ROLE_CONTROL]
            if controls:
                c = controls[0]
                for i, r in enumerate(a.rows):
                    if r.role == core.ROLE_TARGET:
                        res[f"{r.name}/{a.rows[c].name}"] = core.normalize_to_control(a, d.measurements, i, c, d.ref_lane)
            else:
                # no internal control: plain band intensities (raw Net per lane)
                for i, r in enumerate(a.rows):
                    if r.role != core.ROLE_IGNORE:
                        res[f"{r.name} (Net)"] = core.row_vector(a, d.measurements, i)
        variables = core.formula_variables(a, d.measurements)
        for f in a.formulas:
            name = f.get("name") or f.get("expr")
            expr = str(f.get("expr", "")).strip()
            if not expr:
                continue
            try:
                val = core.evaluate(expr, variables)
                if f.get("normalize", "1") == "1" and np.ndim(val) == 1:
                    val = core.scale_to_reference(val, d.ref_lane)
                res[name] = np.atleast_1d(np.asarray(val, float))
            except Exception as e:
                msgs.append(f"{name}: {e}")
        return res, msgs

    def recompute_results(self, *_):
        self.results_title.setText(f"현재 이미지: {self.doc_key(self.cur)} · {self.doc.name}")
        self.results, msgs = self.image_results(self.doc)
        self.formula_msg.setText("\n".join(msgs) if msgs else FORMULA_HINT)
        self.formula_msg.setStyleSheet("color: #c0392b;" if msgs else "color: #888;")
        self._fill_series_list(self.series_list, self.results)
        self.fill_results_table()
        self.replot()
        if hasattr(self, "figure_editor") and self.tabs.currentIndex() != self.FIG_TAB:
            self.figure_editor.dirty = True  # numbers in the figure may have changed

    @staticmethod
    def _fill_series_list(lst, names):
        prev = {lst.item(i).text(): lst.item(i).checkState() for i in range(lst.count())}
        lst.blockSignals(True)
        lst.clear()
        for name in names:
            it = QtWidgets.QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(prev.get(name, Qt.Checked))
            lst.addItem(it)
        lst.blockSignals(False)

    @staticmethod
    def _checked(lst):
        return [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def fill_results_table(self):
        a = self.analysis
        names = list(self.results)
        t = self.results_table
        n = a.n_lanes
        t.clear()
        t.setColumnCount(len(names) + 1)
        t.setHorizontalHeaderLabels(["레인 그룹"] + names)
        t.setRowCount(max(n, 1 if names else 0))
        for i in range(n):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(a.lane_labels[i] if i < len(a.lane_labels) else str(i + 1)))
        for c, name in enumerate(names):
            vals = self.results[name]
            if vals.size == 1:
                t.setItem(0, c + 1, QtWidgets.QTableWidgetItem(f"{vals[0]:.4g}"))
                continue
            for i in range(min(n, vals.size)):
                it = QtWidgets.QTableWidgetItem("" if not np.isfinite(vals[i]) else f"{vals[i]:.4g}")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, c + 1, it)
        t.resizeColumnsToContents()

    def replot(self, *_):
        a = self.analysis
        series = {k: self.results[k] for k in self._checked(self.series_list) if k in self.results}
        # don't mix scalars and lane vectors in one chart
        if any(np.size(v) > 1 for v in series.values()):
            series = {k: v for k, v in series.items() if np.size(v) > 1}
        if series and all(k.endswith(" (Net)") for k in series):
            ylabel = "Net intensity (raw)"
        else:
            ylabel = "상대 intensity" if self.doc.ref_lane is not None else "정량값"
        labels = list(a.lane_labels)
        self.plot.plot(series, labels, labels, self.by_condition.isChecked(), ylabel, self.show_points.isChecked())

    # ---------------------------------------------------------------- several images
    def lane_key_map(self) -> Dict[tuple, str]:
        """(image index, lane group) -> common lane name, from the lane big groups."""
        m = {}
        for g in self.big_groups:
            if g["kind"] == KIND_LANE:
                for k, name in g["members"]:
                    m[(int(k), name)] = g["name"]
        return m

    def image_lane_keys(self) -> Dict[int, List[str]]:
        m = self.lane_key_map()
        return {k: [m.get((k, lab), lab) for lab in d.analysis.lane_labels]
                for k, d in enumerate(self.docs) if d.raw is not None}

    def refresh_multi(self):
        self.fill_group_tree()
        self.fill_big_table()
        self.recompute_multi()

    def fill_group_tree(self):
        t = self.group_tree
        checked = set()
        for i in range(t.topLevelItemCount()):
            top = t.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.checkState(0) == Qt.Checked:
                    checked.add(tuple(c.data(0, Qt.UserRole)))
        t.blockSignals(True)
        t.clear()
        for k, d in enumerate(self.docs):
            if d.raw is None:
                continue
            top = QtWidgets.QTreeWidgetItem([f"{self.doc_key(k)}: {d.name}", ""])
            t.addTopLevelItem(top)
            for kind, names in ((KIND_PROTEIN, [r.name for r in d.analysis.rows]), (KIND_LANE, d.analysis.lane_groups())):
                for name in names:
                    c = QtWidgets.QTreeWidgetItem([name, KIND_LABELS[kind]])
                    key = (k, kind, name)
                    c.setData(0, Qt.UserRole, list(key))
                    c.setFlags(c.flags() | Qt.ItemIsUserCheckable)
                    c.setCheckState(0, Qt.Checked if key in checked else Qt.Unchecked)
                    top.addChild(c)
            top.setExpanded(True)
        t.blockSignals(False)

    def make_big_group(self):
        picked = []
        t = self.group_tree
        for i in range(t.topLevelItemCount()):
            top = t.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.checkState(0) == Qt.Checked:
                    picked.append(c.data(0, Qt.UserRole))
        if not picked:
            self.statusBar().showMessage("위 목록에서 묶을 그룹을 체크하세요")
            return
        kinds = {p[1] for p in picked}
        if len(kinds) > 1:
            QtWidgets.QMessageBox.information(self, "큰 그룹", "단백질 그룹과 레인 그룹은 따로 묶어 주세요.")
            return
        kind = kinds.pop()
        default = picked[0][2]
        name = self._ask_name("큰 그룹 만들기", f"{KIND_LABELS[kind]} 큰 그룹 이름:", default)
        if not name:
            return
        final = core.unique_name(name, [g["name"] for g in self.big_groups])
        self.big_groups.append({"name": final, "kind": kind, "members": [[int(p[0]), p[2]] for p in picked],
                                "value": "ratio" if kind == KIND_PROTEIN else ""})
        for i in range(t.topLevelItemCount()):
            top = t.topLevelItem(i)
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, Qt.Unchecked)
        self.refresh_multi()
        self.statusBar().showMessage(f"큰 그룹 '{final}'을(를) 만들었습니다" + (f" ('{name}'이(가) 이미 있어 이름을 바꿈)" if final != name else ""))

    def delete_big_group(self):
        for r in sorted({i.row() for i in self.big_table.selectedIndexes()}, reverse=True):
            del self.big_groups[r]
        self.refresh_multi()

    def fill_big_table(self):
        t = self.big_table
        t.setRowCount(len(self.big_groups))
        for i, g in enumerate(self.big_groups):
            members = ", ".join(f"{self.doc_key(k)}:{n}" for k, n in g["members"])
            for c, v in enumerate((g["name"], KIND_LABELS[g["kind"]], members)):
                t.setItem(i, c, QtWidgets.QTableWidgetItem(v))
            if g["kind"] == KIND_PROTEIN:
                cb = QtWidgets.QComboBox()
                for key, label in VALUE_LABELS.items():
                    cb.addItem(label, key)
                cb.setCurrentIndex(max(0, cb.findData(g.get("value", "ratio"))))
                cb.currentIndexChanged.connect(lambda _=None, g=g, cb=cb: self.set_big_value(g, cb.currentData()))
                t.setCellWidget(i, 3, cb)
            else:
                t.setItem(i, 3, QtWidgets.QTableWidgetItem("—"))
        t.resizeColumnToContents(0)

    def set_big_value(self, g, value):
        g["value"] = value
        self.recompute_multi()

    def on_multi_ref(self, *_):
        self.multi_ref = self.multi_ref_combo.currentData()
        self.recompute_multi()

    def fill_multi_formulas(self):
        self._fill_formula_table(self.multi_formula_table, self.multi_formulas)

    def multi_terms(self):
        terms = []
        for k, d in enumerate(self.docs):
            if d.raw is None:
                continue
            key = self.doc_key(k)
            terms.append((f"{key} 단백질 (레인별 값)", [(f"{key}: {r.name}", f"{key}.{core._safe_name(r.name)}") for r in d.analysis.rows]))
            ids = sorted({m.band_id for m in d.measurements})
            terms.append((f"{key} 밴드 (값 하나)", [(f"{key}: {i}", f"{key}.{i}") for i in ids]))
        prot = [g["name"] for g in self.big_groups if g["kind"] == KIND_PROTEIN and g["name"] in self.multi_results]
        terms.append(("단백질 큰 그룹 (이미지 평균)", [(n, core._safe_name(n)) for n in prot]))
        return terms

    def add_multi_formula(self):
        _, _, variables, _ = self.compute_multi()
        f = self._edit_formula_dialog("이미지 간 수식 추가", self.multi_terms(), variables,
                                      {"name": f"이미지수식{len(self.multi_formulas) + 1}"})
        if f:
            self.multi_formulas.append(f)
            self.fill_multi_formulas()
            self.recompute_multi()

    def edit_multi_formula(self, row=None):
        if row is None or row is False:
            rows = sorted({i.row() for i in self.multi_formula_table.selectedIndexes()})
            row = rows[0] if rows else None
        if row is None or row >= len(self.multi_formulas):
            self.statusBar().showMessage("편집할 수식을 표에서 고르세요")
            return
        _, _, variables, _ = self.compute_multi()
        f = self._edit_formula_dialog("이미지 간 수식 편집", self.multi_terms(), variables, self.multi_formulas[row])
        if f:
            self.multi_formulas[row] = f
            self.fill_multi_formulas()
            self.recompute_multi()

    def delete_multi_formula(self):
        for r in sorted({i.row() for i in self.multi_formula_table.selectedIndexes()}, reverse=True):
            del self.multi_formulas[r]
        self.fill_multi_formulas()
        self.recompute_multi()

    def on_multi_formula_changed(self, item):
        self._formula_edit(self.multi_formulas, item)
        self.recompute_multi()

    def compute_multi(self):
        """Values of all images on common lanes. Returns (lane keys, results, variables, messages)."""
        keys = self.image_lane_keys()
        common = core.common_lane_keys([keys[k] for k in sorted(keys)])
        ref = common.index(self.multi_ref) if self.multi_ref in common else None
        image_vars = []
        for k, d in enumerate(self.docs):
            v: Dict[str, object] = {}
            if k in keys and d.measurements:
                a = d.analysis
                for ri, r in enumerate(a.rows):
                    v[core._safe_name(r.name)] = core.to_common(core.row_vector(a, d.measurements, ri), keys[k], common)
                for m in d.measurements:
                    v.setdefault(m.band_id, m.net)
            image_vars.append(v)
        variables = core.project_variables(image_vars, base=-1)
        results: Dict[str, Dict[str, object]] = {}
        msgs = []
        for g in self.big_groups:
            if g["kind"] != KIND_PROTEIN:
                continue
            vecs, labels = [], []
            for k, name in g["members"]:
                d = self.docs[k] if k < len(self.docs) else None
                if d is None or k not in keys or not d.measurements:
                    continue
                a = d.analysis
                ri = next((i for i, r in enumerate(a.rows) if r.name == name), None)
                if ri is None:
                    msgs.append(f"{g['name']}: {self.doc_key(k)}에 '{name}' 단백질 그룹이 없습니다")
                    continue
                if g.get("value") == "ratio":
                    ctrl = next((i for i, r in enumerate(a.rows) if r.role == core.ROLE_CONTROL), None)
                    if ctrl is None or ctrl == ri:
                        msgs.append(f"{g['name']}: {self.doc_key(k)}에 Control 단백질이 없어 비율을 계산할 수 없습니다")
                        continue
                    vec = core.normalize_to_control(a, d.measurements, ri, ctrl, None)
                else:
                    vec = core.row_vector(a, d.measurements, ri)
                vecs.append(core.to_common(vec, keys[k], common))
                labels.append(f"{self.doc_key(k)}:{name}")
            if not vecs:
                continue
            mean, sd, n, scaled = core.combine_members(vecs, ref)
            results[g["name"]] = {"mean": mean, "sd": sd, "n": n, "points": scaled, "members": labels}
            variables.setdefault(core._safe_name(g["name"]), mean)
        for f in self.multi_formulas:
            name = f.get("name") or f.get("expr")
            expr = str(f.get("expr", "")).strip()
            if not expr:
                continue
            try:
                val = np.atleast_1d(np.asarray(core.evaluate(expr, variables), float))
                if f.get("normalize", "1") == "1" and val.size == len(common) > 1:
                    val = core.scale_to_reference(val, ref)
                results[name] = {"mean": val, "sd": None, "n": None, "points": None, "members": []}
            except Exception as e:
                msgs.append(f"{name}: {e}")
        return common, results, variables, msgs

    def recompute_multi(self, *_):
        common, self.multi_results, _, msgs = self.compute_multi()
        self.multi_keys = common
        self.multi_msg.setText("\n".join(msgs) if msgs else MULTI_FORMULA_HINT)
        self.multi_msg.setStyleSheet("color: #c0392b;" if msgs else "color: #888;")
        cb = self.multi_ref_combo
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("정규화 안 함", None)
        for key in common:
            cb.addItem(key, key)
        idx = cb.findData(self.multi_ref) if self.multi_ref is not None else 0
        cb.setCurrentIndex(max(0, idx))
        cb.blockSignals(False)
        self._fill_series_list(self.multi_series, self.multi_results)
        self.fill_multi_table()
        self.replot_multi()

    def fill_multi_table(self):
        t = self.multi_table
        cols = ["레인"]
        for name, r in self.multi_results.items():
            cols += [name] if r["sd"] is None else [f"{name} 평균", f"{name} SD", f"{name} n"]
        t.clear()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(self.multi_keys))
        for i, key in enumerate(self.multi_keys):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(key))
        c = 1
        for r in self.multi_results.values():
            arrays = [r["mean"]] if r["sd"] is None else [r["mean"], r["sd"], r["n"]]
            for arr in arrays:
                arr = np.atleast_1d(arr)
                for i in range(min(arr.size, max(len(self.multi_keys), 1))):
                    v = arr[i]
                    txt = "" if not np.isfinite(v) else (f"{int(v)}" if arr.dtype.kind == "i" else f"{v:.4g}")
                    it = QtWidgets.QTableWidgetItem(txt)
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if t.rowCount() <= i:
                        t.setRowCount(i + 1)
                    t.setItem(i, c, it)
                c += 1
        t.resizeColumnsToContents()

    def replot_multi(self, *_):
        series = {k: self.multi_results[k] for k in self._checked(self.multi_series) if k in self.multi_results}
        if any(np.size(s["mean"]) > 1 for s in series.values()):
            series = {k: s for k, s in series.items() if np.size(s["mean"]) > 1}
        ylabel = "상대 intensity" if self.multi_ref is not None else "정량값"
        self.multi_plot.plot_bars(self.multi_keys, series, ylabel, self.multi_points.isChecked())

    # ---------------------------------------------------------------- save / export
    def project_dict(self):
        return {
            "app": "Blot Profiler",
            "version": __version__,
            "format": 3,
            "detect": {
                "sensitivity": self.sens.value(), "smooth": self.smooth.value(), "min_area": self.min_area.value(),
                "pad": self.pad.value(), "edge": self.edge.value(), "lut_gamma": self.lut_gamma.value(),
            },
            "images": [
                {
                    "image_path": d.image_path, "invert": d.invert, "background": vars(d.bg),
                    "reference_lane": d.ref_lane, "analysis": d.analysis.to_dict(), "figure": d.figure,
                }
                for d in self.docs if d.raw is not None
            ],
            "big_groups": self.big_groups,
            "multi_formulas": self.multi_formulas,
            "multi_reference": self.multi_ref,
        }

    ROI_APP = "Blot Profiler ROI"

    def save_roi(self):
        a = self.analysis
        if not a.bands:
            self.statusBar().showMessage("저장할 밴드 ROI가 없습니다")
            return
        base = os.path.splitext(self.image_path)[0] + "_roi.json" if self.image_path else "roi.json"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "밴드 ROI·그룹 저장", base, "ROI (*.json)")
        if not p:
            return
        d = a.to_dict()
        d.pop("formulas", None)
        d = {"app": self.ROI_APP, "version": __version__,
             "image_size": list(self.raw.shape[::-1]) if self.raw is not None else None, **d}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        self.statusBar().showMessage(f"ROI·그룹 저장: {p} (단백질 {len(a.rows)}개, 레인 {a.n_lanes}개, 밴드 {len(a.bands)}개)")

    def load_roi(self, p=None):
        if self.raw is None:
            self.statusBar().showMessage("먼저 ROI를 적용할 이미지를 여세요")
            return
        if not p:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "밴드 ROI·그룹 불러오기", "", "ROI / 분석 (*.json)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            size = d.get("image_size")
            if "images" in d:  # a full analysis file: take its first image
                im = d["images"][0] if d["images"] else {}
                d = im.get("analysis") or {}
                size = None
            elif "analysis" in d:  # analysis file format 1
                d = d["analysis"]
            if not d.get("bands") or not d.get("rows"):
                raise ValueError("밴드 ROI가 들어 있지 않은 파일입니다")
            a = core.Analysis.from_dict(d)
            a.formulas = list(self.analysis.formulas)
            H, W = self.raw.shape
            if size and tuple(size) != (W, H):
                r = QtWidgets.QMessageBox.question(
                    self, "이미지 크기가 다릅니다",
                    f"ROI를 저장한 이미지({size[0]}×{size[1]})와 현재 이미지({W}×{H})의 크기가 다릅니다.\n"
                    "현재 이미지 크기에 맞춰 박스 위치와 크기를 비율대로 조정할까요?\n(아니요: 픽셀 좌표 그대로 적용)")
                if r == QtWidgets.QMessageBox.Yes:
                    sx, sy = W / size[0], H / size[1]
                    for b in a.bands:
                        b.cx *= sx
                        b.cy *= sy
                    for row in a.rows:
                        row.width = max(2, int(round(row.width * sx)))
                        row.height = max(2, int(round(row.height * sy)))
            self.set_analysis(a)
            self.statusBar().showMessage(f"ROI·그룹을 적용했습니다: {os.path.basename(p)} (밴드 {len(a.bands)}개)")
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "불러오기 실패", str(e))

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
            else:  # format 1: a single image
                images = [{k: d.get(k) for k in ("image_path", "invert", "background", "reference_lane", "analysis")}]
            det = d.get("detect", {})
            self.sens.setValue(det.get("sensitivity", 1.0))
            self.smooth.setValue(det.get("smooth", 1.5))
            self.min_area.setValue(det.get("min_area", 30))
            self.pad.setValue(det.get("pad", 3))
            self.edge.setValue(det.get("edge", 0))
            self.lut_gamma.setValue(det.get("lut_gamma", 0.6))
            docs, index = [], {}
            for i, im in enumerate(images):
                path = self._find_image(im.get("image_path"), p)
                if not path:
                    continue
                doc = ImageDoc(invert=int(im.get("invert") or 0), bg=core.BackgroundSettings(**(im.get("background") or {})))
                doc.raw = core.load_image(path)
                doc.image_path = path
                doc.ref_lane = im.get("reference_lane", 0)
                doc.analysis = core.Analysis.from_dict(im.get("analysis") or {})
                doc.analysis.ensure_lane_labels()
                doc.figure = im.get("figure")
                self._make_signal(doc)
                self.measure_doc(doc)
                index[i] = len(docs)
                docs.append(doc)
            if not docs:
                return
            # format 2 kept formulas for all images in one list, each with a base image
            for fml in d.get("formulas", []):
                k = index.get(int(fml.get("image", 0)))
                if k is not None:
                    docs[k].analysis.formulas.append({x: fml[x] for x in ("name", "expr", "normalize") if x in fml})
            self.docs = docs
            self.big_groups = [
                dict(g, members=[[index[int(k)], n] for k, n in g.get("members", []) if int(k) in index])
                for g in d.get("big_groups", [])
            ]
            self.multi_formulas = [dict(f) for f in d.get("multi_formulas", [])]
            self.multi_ref = d.get("multi_reference")
            self.cur = 0
            self.refresh_doc_tabs()
            self.fill_multi_formulas()
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
                w.writerow([f"# image {k}", d.image_path or "", "background", BG_LABELS[d.bg.method],
                            "radius", d.bg.radius, "border", d.bg.border])
            w.writerow([])
            w.writerow(["[Bands]"])
            w.writerow(["image", "band_id", "protein", "role", "lane", "lane_group", "x", "y", "width", "height",
                        "raw_intensity", "background", "net_intensity", "mean_per_px"])
            for d in docs:
                a = d.analysis
                k = self.doc_key(self.docs.index(d))
                ids = a.band_ids()
                by_id = {ids[id(b)]: b for b in a.bands}
                for m in d.measurements:
                    x, y, wd, ht = by_id[m.band_id].box(a.rows)
                    r = a.rows[m.row]
                    w.writerow([k, m.band_id, r.name, r.role, m.lane + 1, a.lane_labels[m.lane],
                                x, y, wd, ht, f"{m.raw:.3f}", f"{m.background:.3f}", f"{m.net:.3f}", f"{m.mean:.4f}"])
            w.writerow([])
            w.writerow(["[Results per image]"])
            w.writerow(["image", "series", "reference_lane", "lane", "lane_group", "value"])
            cond_rows = []
            for d in docs:
                a = d.analysis
                k = self.doc_key(self.docs.index(d))
                ref = "" if d.ref_lane is None else d.ref_lane + 1
                res, _ = self.image_results(d)
                for n, v in res.items():
                    if v.size == 1:
                        w.writerow([k, n, ref, "", "", f"{v[0]:.6g}"])
                        continue
                    for i in range(v.size):
                        w.writerow([k, n, ref, i + 1, a.lane_labels[i] if i < len(a.lane_labels) else "",
                                    f"{v[i]:.6g}" if np.isfinite(v[i]) else ""])
                    if len(set(a.lane_labels)) < len(a.lane_labels):
                        cn, means, sds, ns = core.condition_stats(v, a.lane_labels, a.lane_labels)
                        cond_rows += [[k, n, c_, f"{m_:.6g}", f"{s_:.6g}", k_] for c_, m_, s_, k_ in zip(cn, means, sds, ns)]
            if cond_rows:
                w.writerow([])
                w.writerow(["[By lane group (replicate lanes)]"])
                w.writerow(["image", "series", "lane_group", "mean", "sd", "n"])
                w.writerows(cond_rows)
            if self.big_groups or self.multi_formulas:
                common, results, _, _ = self.compute_multi()
                w.writerow([])
                w.writerow(["[Several images]", "reference lane", self.multi_ref or ""])
                for g in self.big_groups:
                    w.writerow(["big group", g["name"], KIND_LABELS[g["kind"]], VALUE_LABELS.get(g.get("value"), ""),
                                "; ".join(f"{self.doc_key(k)}:{n}" for k, n in g["members"])])
                w.writerow(["series", "lane", "mean", "sd", "n", "members"])
                for n, r in results.items():
                    for i, key in enumerate(common):
                        if i >= np.size(r["mean"]):
                            break
                        mean = np.atleast_1d(r["mean"])[i]
                        sd = "" if r["sd"] is None else f"{r['sd'][i]:.6g}"
                        cnt = "" if r["n"] is None else int(r["n"][i])
                        pts = "" if r["points"] is None else "; ".join(
                            f"{lab}={p[i]:.6g}" for lab, p in zip(r["members"], r["points"]) if np.isfinite(p[i]))
                        w.writerow([n, key, f"{mean:.6g}" if np.isfinite(mean) else "", sd, cnt, pts])
                w.writerow([])
                w.writerow(["[Formulas across images]"])
                for fml in self.multi_formulas:
                    w.writerow([fml.get("name"), fml.get("expr")])
            w.writerow([])
            w.writerow(["[Formulas per image]"])
            for d in docs:
                for fml in d.analysis.formulas:
                    w.writerow([self.doc_key(self.docs.index(d)), fml.get("name"), fml.get("expr")])
        self.statusBar().showMessage(f"CSV 저장: {p}")

    def export_plot(self, canvas=None):
        if canvas is None:
            canvas = self.multi_plot if self.tabs.currentIndex() == 4 else self.plot
        base = os.path.splitext(self.image_path)[0] + "_plot.png" if self.image_path else "plot.png"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "그래프 저장", base, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if p:
            canvas.fig.savefig(p, dpi=300, bbox_inches="tight")
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


FORMULA_HINT = "'수식 추가…'에서 단백질·밴드를 목록에서 고르고 + − × ÷ ( ) 로 잇습니다. 예: pErk / Erk · Kras_DOX / Kras_DMSO"
MULTI_FORMULA_HINT = "'수식 추가…'에서 이미지별 단백질·밴드·큰 그룹을 목록에서 고릅니다. 예: I2.Kras / I1.Kras"

HELP_TEXT = """Blot Profiler 사용 순서

1. 검출 탭: TIF 파일을 열면 밴드를 자동으로 찾습니다. 여러 TIF를 열면 위쪽 탭(I1, I2 …)으로 전환합니다.
   · 배경 제거 방법을 고르고(기본: Rolling ball) 민감도를 조절해 다시 검출할 수 있습니다.

2. 그룹 만들기: 이미지에서 밴드를 드래그로 선택한 뒤 우클릭
   · 새 단백질(행) 그룹 만들기 → 이름 = 단백질 이름 (예: Vinculin, pErk)
   · 새 레인(열) 그룹 만들기 → 이름 = 레인 이름 (예: DMSO, IgG 12h)
     같은 단백질의 밴드를 여러 열에서 고르면 그 열들이 반복(평균±SD)이 됩니다.
   · 밴드 ID = '단백질_레인' (이름이 겹치면 _2, _3 자동)
   · 한 단백질 그룹의 박스는 같은 크기. 우클릭 → '박스 크기 조절 모드'에서 가장자리를 드래그 (Esc 종료).

   · 밴드 ROI·그룹 저장/불러오기(검출 탭, 파일 메뉴): 박스와 그룹만 따로 저장해 다른 이미지에 적용합니다.

3. 단백질 / 레인 탭: 단백질의 역할(측정 Target / Control / 제외)과 박스 크기를 정합니다.

4. 정량 / 그래프 탭 (현재 이미지): Target ÷ Control 을 레인별로 계산하고 기준 레인을 1로 맞춥니다.
   수식 추가…: 단백질 그룹·밴드는 목록에서 골라 넣고, + − × ÷ ( ) 와 숫자만 직접 입력합니다.
   수식을 더블클릭하면 다시 편집합니다.

5. 여러 이미지 탭: 이미지들의 그룹을 체크해 큰 그룹을 만듭니다.
   · 레인 큰 그룹: 이미지마다 다른 레인 이름을 하나로 맞춤
   · 단백질 큰 그룹: 여러 블롯의 값을 레인별 평균 ± SD
   · 이미지 간 수식: 수식 창에서 이미지별 항목을 골라 만듭니다 (예: I2.Kras / I1.Kras)

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
