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
from . import project as projfile  # noqa: E402
from . import lanes  # noqa: E402
from . import graph, prefs  # noqa: E402
from .ui import controls, theme  # noqa: E402


def _tok(name: str) -> str:
    """A colour token of the design system for a widget style sheet (light or dark)."""
    from .ui import hig_style  # noqa: PLC0415

    return hig_style.rgba(name, theme.scheme())
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
        self.setToolTip("측정 묶음: " + ", ".join(f"{k} › {v}" for k, v in self.band.tags.items()) if self.band.tags else "")
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
            self.band.user = True
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


def _seg_dist(p, a, b) -> float:
    """Distance (px) of point p from segment a–b."""
    ax, ay, bx, by, px, py = a.x(), a.y(), b.x(), b.y(), p.x(), p.y()
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5


class BlotView(QtWidgets.QGraphicsView):
    filesDropped = QtCore.Signal(list)  # image/analysis files dropped on the view
    drawn = QtCore.Signal(str, float, float, float, float)  # "line"/"crop", x0, y0, x1, y1 (image px)
    editRequested = QtCore.Signal(object)  # BandItem double-clicked: resize it
    linesChanged = QtCore.Signal(str)  # lane / protein lines edited (message)
    markerEdited = QtCore.Signal(str)  # marker strip or size marks edited
    tickSelected = QtCore.Signal(int)
    proteinDrawn = QtCore.Signal(float, float, float, float)
    laneMenu = QtCore.Signal(QtCore.QPoint, float, float)
    proteinMenu = QtCore.Signal(QtCore.QPoint, int)
    toolHint = QtCore.Signal(str)
    zoomChanged = QtCore.Signal(float)
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
        self.setBackgroundBrush(theme.region_color("canvas"))
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
        self.draw_mode: Optional[str] = None  # "line" (straighten) / "crop": the next drag draws
        self.extra_items = []  # measurement groups
        self.line_items = []  # lane / protein lines, marker strip (1.0 canvas tools)
        self.tool = "select"
        self.sel: Optional[dict] = None
        self._drag: Optional[dict] = None
        self.marker_s: Optional[dict] = None  # figure settings of the image: marker strip and size marks
        self._space = False
        self._pan = None
        self.show_set: Optional[str] = None  # grouping set drawn on the image
        self.auto_fit = True  # keep the image fitted to the view until the user zooms
        self.show_lanes = True
        self._draw = None
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

    def set_marker(self, qimg: Optional[QtGui.QImage], region=None, on_region=None):
        """Show the marker (ladder) image left of the blot, same height, for reference.

        ``region`` = (x0, x1, dy) of the figure's marker strip, editable with the mouse.
        """
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
            if region is not None and on_region is not None:
                from .figure_window import StripRegionItem

                x0, x1, dy = region
                StripRegionItem(x0, x1, dy, pm.height(), pm.width(), on_region, parent=self.marker_item)
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
            self.fitInView(r.adjusted(-10, -26, 10, 10), Qt.KeepAspectRatio)
        self.auto_fit = True
        self.zoomChanged.emit(self.transform().m11())

    def zoom(self, factor: float):
        self.auto_fit = False
        self.scale(factor, factor)
        self.zoomChanged.emit(self.transform().m11())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.auto_fit:  # the image follows the size of the window until the user zooms
            self.fit()

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
        self.rebuild_extras()

    def rebuild_extras(self):
        """The (slanted) lane centre lines of lane-first detection and the groups of the
        measurement grouping set being shown (coloured frames joined by a line, with the name)."""
        sc = self.scene()
        for it in self.extra_items:
            sc.removeItem(it)
        self.extra_items = []
        a = self.analysis
        g = core.grouping(a, self.show_set) if self.show_set else None
        for gi, gname in enumerate(g["groups"] if g else []):
            mem = core.group_members(a, self.show_set, gname)
            if not mem:
                continue
            color = QtGui.QColor(GROUP_COLORS[gi % len(GROUP_COLORS)])
            pen = QtGui.QPen(color, 3)
            pen.setCosmetic(True)
            for b in mem:
                x, y, w_, h_ = b.box(a.rows)
                it = sc.addRect(QtCore.QRectF(x - 3, y - 3, w_ + 6, h_ + 6), pen)
                it.setZValue(4)
                self.extra_items.append(it)
            pts = sorted(((b.cx, b.cy) for b in mem), key=lambda p: (p[1], p[0]))
            if len(pts) > 1:
                lp = QtGui.QPen(color, 2, Qt.DashLine)
                lp.setCosmetic(True)
                path = QtGui.QPainterPath(QtCore.QPointF(*pts[0]))
                for x, y in pts[1:]:
                    path.lineTo(x, y)
                it = sc.addPath(path, lp)
                it.setZValue(4)
                self.extra_items.append(it)
            t = sc.addSimpleText(f"{gname}")
            f = t.font()
            f.setBold(True)
            f.setPointSizeF(11)
            t.setFont(f)
            t.setBrush(color)
            t.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            top = min(mem, key=lambda b: b.cy)
            x, y, w_, h_ = top.box(a.rows)
            t.setPos(x + w_ + 4, y - 4)  # right of the top box, clear of the box ID
            t.setZValue(5)
            self.extra_items.append(t)
        self.rebuild_lines()


    # ---------------------------------------------------------------- 1.0 canvas tools
    # 선택 (boxes) · 레인 선 (blue polylines, point editing) · 단백질 선 (red dashed lines)
    # · 마커 영역 (strip on the marker image, size marks). Hit tests in screen pixels.
    LANE_BLUE = "#1f7ae0"
    PROT_RED = "#d70015"
    MARK_ORANGE = "#ff9f0a"
    HIT = 7  # px

    def refresh_theme(self):
        self.setBackgroundBrush(theme.region_color("canvas"))

    def set_tool(self, tool: str):
        self.tool = tool
        self.sel = None
        for it in self.items_:
            it.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, tool == "select")
            it.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, tool == "select")
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag if tool == "select" else QtWidgets.QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.ArrowCursor if tool == "select" else Qt.CrossCursor)
        self.rebuild_lines()

    def _px(self, x, y) -> QtCore.QPointF:
        return QtCore.QPointF(self.mapFromScene(QtCore.QPointF(x, y)))

    def _img_h(self) -> float:
        return float(self.pix_item.pixmap().height()) if self.pix_item is not None else 0.0

    def _lane_hit(self, pos):
        """(line index, point index or None) under the mouse."""
        best = None
        for i, ln in enumerate(self.analysis.lane_lines):
            pts = sorted(ln, key=lambda p: p[1])
            for j, (x, y) in enumerate(pts):
                d_ = (self._px(x, y) - pos).manhattanLength()
                if d_ <= self.HIT + 2 and (best is None or d_ < best[0]):
                    best = (d_, i, j)
        if best:
            return best[1], best[2]
        for i, ln in enumerate(self.analysis.lane_lines):
            pts = sorted(ln, key=lambda p: p[1])
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                a, b = self._px(x0, y0), self._px(x1, y1)
                if _seg_dist(pos, a, b) <= self.HIT:
                    return i, None
        return None

    def _pline_hit(self, pos):
        """(line index, 'a' | 'b' | None) under the mouse."""
        for i, p in enumerate(self.analysis.protein_lines):
            for end, (x, y) in (("a", (p["x0"], p["y0"])), ("b", (p["x1"], p["y1"]))):
                if (self._px(x, y) - pos).manhattanLength() <= self.HIT + 2:
                    return i, end
        for i, p in enumerate(self.analysis.protein_lines):
            if _seg_dist(pos, self._px(p["x0"], p["y0"]), self._px(p["x1"], p["y1"])) <= self.HIT:
                return i, None
        return None

    def _marker_x(self, x) -> float:
        return (self.marker_item.x() if self.marker_item is not None else 0.0) + x

    def _tick_hit(self, pos):
        s = self.marker_s
        if not s or self.marker_item is None:
            return None
        x0, x1 = self._marker_x(s.get("strip_x0", 0)), self._marker_x(s.get("strip_x1", 0))
        for i, k in enumerate(s.get("kda", [])):
            a, b = self._px(x0, k["y"]), self._px(x1, k["y"])
            if _seg_dist(pos, a, b) <= self.HIT:
                return i
        return None

    def rebuild_lines(self):
        """Lane lines, protein lines and the marker strip as drawn items (rebuilt after each edit)."""
        sc = self.scene()
        for it in self.line_items:
            sc.removeItem(it)
        self.line_items = []
        a = self.analysis
        H = self._img_h()
        if self.pix_item is None:
            return
        tool = self.tool
        sel = self.sel or {}
        show_lanes = self.show_lanes or tool in ("lane", "protein")
        if show_lanes:
            for i, ln in enumerate(a.lane_lines):
                pts = sorted(ln, key=lambda p: p[1])
                on = sel.get("kind") == "lane" and sel.get("i") == i
                pen = QtGui.QPen(QtGui.QColor(self.LANE_BLUE), 2.6 if on else 1.6)
                pen.setCosmetic(True)
                if tool not in ("lane",):
                    pen.setColor(QtGui.QColor(31, 122, 224, 140))
                    pen.setStyle(Qt.DashLine)
                path = QtGui.QPainterPath(QtCore.QPointF(*pts[0]))
                for x, y in pts[1:]:
                    path.lineTo(x, y)
                it = sc.addPath(path, pen)
                it.setZValue(6)
                self.line_items.append(it)
                if tool == "lane":
                    for j, (x, y) in enumerate(pts):
                        filled = on and sel.get("j") == j
                        self._handle(x, y, "circle", self.LANE_BLUE, filled, 4.5 if on else 3.5)
                lab = a.lane_labels[i] if i < len(a.lane_labels) else f"L{i + 1}"
                self._tag(core.poly_x(pts, H - 1) if H else pts[-1][0], H - 1, lab, self.LANE_BLUE, on, below=False)
        if tool == "protein" or (self.show_lanes and a.protein_lines):
            for i, p in enumerate(a.protein_lines):
                on = sel.get("kind") == "pline" and sel.get("i") == i
                pen = QtGui.QPen(QtGui.QColor(self.PROT_RED), 2.4 if on else 1.6, Qt.DashLine)
                pen.setCosmetic(True)
                it = sc.addLine(p["x0"], p["y0"], p["x1"], p["y1"], pen)
                it.setZValue(6)
                self.line_items.append(it)
                if tool == "protein":
                    for x, y in ((p["x0"], p["y0"]), (p["x1"], p["y1"])):
                        self._handle(x, y, "square", self.PROT_RED, on, 4.0)
                self._tag(p["x1"] + 6, p["y1"], str(p.get("row", "")), self.PROT_RED, True, below=None)
        s = self.marker_s
        if self.marker_item is not None and s and (tool == "marker" or s.get("kda")):
            x0, x1 = self._marker_x(s.get("strip_x0", 0)), self._marker_x(s.get("strip_x1", 0))
            if x1 > x0:
                pen = QtGui.QPen(QtGui.QColor(self.MARK_ORANGE), 1.5, Qt.DashLine)
                pen.setCosmetic(True)
                it = sc.addRect(QtCore.QRectF(x0, 0, x1 - x0, H), pen)
                it.setZValue(6)
                self.line_items.append(it)
                for i, k in enumerate(s.get("kda", [])):
                    on = sel.get("kind") == "tick" and sel.get("i") == i
                    color = "#0a84ff" if on else self.MARK_ORANGE
                    pen = QtGui.QPen(QtGui.QColor(color), 3 if on else 1.6)
                    pen.setCosmetic(True)
                    it = sc.addLine(x0 + 2, k["y"], x1 - 2, k["y"], pen)
                    it.setZValue(7)
                    self.line_items.append(it)
                    v = k.get("kda")
                    txt = "?" if v is None else (f"{v:g}")
                    self._tag(x1 + 3, k["y"], txt, color, on, below=None, plain=not on)

    def _handle(self, x, y, shape, color, filled, r):
        sc = self.scene()
        pen = QtGui.QPen(QtGui.QColor(color) if not filled else QtGui.QColor("white"), 1.5)
        pen.setCosmetic(True)
        brush = QtGui.QColor(color) if filled else QtGui.QColor("white")
        rect = QtCore.QRectF(-r, -r, 2 * r, 2 * r)
        it = sc.addEllipse(rect, pen, brush) if shape == "circle" else sc.addRect(rect, pen, brush)
        it.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        it.setPos(x, y)
        it.setZValue(8)
        self.line_items.append(it)

    def _tag(self, x, y, text, color, filled, below=False, plain=False):
        if not text:
            return
        sc = self.scene()
        t = QtWidgets.QGraphicsSimpleTextItem(text)
        f = t.font()
        f.setPointSizeF(9)
        f.setBold(True)
        t.setFont(f)
        t.setBrush(QtGui.QColor("white") if filled and not plain else QtGui.QColor(color))
        br = t.boundingRect().adjusted(-4, -1, 4, 1)
        bg = QtWidgets.QGraphicsPathItem()
        path = QtGui.QPainterPath()
        path.addRoundedRect(br, 4, 4)
        bg.setPath(path)
        bg.setBrush(QtGui.QColor(color) if filled and not plain else QtGui.QColor(255, 255, 255, 230))
        pen = QtGui.QPen(QtGui.QColor(color), 1)
        pen.setCosmetic(True)
        bg.setPen(pen)
        grp = sc.createItemGroup([bg, t])
        grp.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        off = -br.height() / 2 if below is None else (2 if below else -br.height() - 2)
        grp.setPos(x, y)
        grp.setTransform(QtGui.QTransform().translate(-br.width() / 2 if below is not None else 0, off))
        grp.setZValue(9)
        self.line_items.append(grp)

    def _tool_press(self, e) -> bool:
        """Mouse press with a line / marker tool; True when handled."""
        pos = e.position()
        p = self.mapToScene(e.position().toPoint())
        a = self.analysis
        H = self._img_h()
        if self.tool == "lane":
            hit = self._lane_hit(pos)
            if e.button() == Qt.RightButton:
                self.sel = {"kind": "lane", "i": hit[0], "j": hit[1]} if hit else None
                self.rebuild_lines()
                self.laneMenu.emit(e.globalPosition().toPoint(), p.x(), p.y())
                return True
            if hit:
                i, j = hit
                self.sel = {"kind": "lane", "i": i, "j": j}
                pts = a.lane_lines[i]
                self._drag = {"kind": "lane_pt" if j is not None else "lane", "i": i, "j": j, "p0": p,
                              "orig": [list(q) for q in sorted(pts, key=lambda q: q[1])]}
            elif self.pix_item is not None and self.pix_item.sceneBoundingRect().contains(p):
                a.lane_lines.append([[p.x(), 0.0], [p.x(), H - 1.0]])
                a.lane_lines.sort(key=lambda ln: float(np.mean([q[0] for q in ln])))
                i = next(k for k, ln in enumerate(a.lane_lines) if abs(ln[0][0] - p.x()) < 1e-6)
                self.sel = {"kind": "lane", "i": i, "j": None}
                self.linesChanged.emit("새 레인 선을 넣었습니다")
            else:
                self.sel = None
            self.rebuild_lines()
            return True
        if self.tool == "protein":
            hit = self._pline_hit(pos)
            if e.button() == Qt.RightButton:
                self.sel = {"kind": "pline", "i": hit[0]} if hit else None
                self.rebuild_lines()
                self.proteinMenu.emit(e.globalPosition().toPoint(), hit[0] if hit else -1)
                return True
            if hit:
                i, end = hit
                self.sel = {"kind": "pline", "i": i}
                self._drag = {"kind": "pline_end" if end else "pline", "i": i, "end": end, "p0": p,
                              "orig": dict(a.protein_lines[i])}
            else:  # drag sideways to draw a new red line
                self.sel = None
                self._drag = {"kind": "pline_new", "p0": p}
            self.rebuild_lines()
            return True
        if self.tool == "marker":
            if self.marker_item is None or not self.marker_s:
                self.toolHint.emit("마커 이미지가 없습니다 — 오른쪽 '마커' 카드에서 마커 이미지를 넣으세요")
                return True
            t = self._tick_hit(pos)
            s = self.marker_s
            if t is not None:
                self.sel = {"kind": "tick", "i": t}
                self._drag = {"kind": "tick", "i": t, "p0": p}
                self.tickSelected.emit(t)
            elif self.marker_item.sceneBoundingRect().contains(p):
                x0, x1 = self._marker_x(s.get("strip_x0", 0)), self._marker_x(s.get("strip_x1", 0))
                if x1 > x0 and x0 <= p.x() <= x1:  # inside the strip: a new size mark
                    s.setdefault("kda", []).append({"kda": None, "y": float(p.y())})
                    s["kda"].sort(key=lambda k: k["y"])
                    i = next(k for k, v in enumerate(s["kda"]) if abs(v["y"] - p.y()) < 1e-6)
                    self.sel = {"kind": "tick", "i": i}
                    self.markerEdited.emit("마커 밴드 위치를 넣었습니다 — 크기를 넣으면 나머지는 래더 순서로 채워집니다")
                    self.tickSelected.emit(i)
                else:  # outside: draw the strip
                    self._drag = {"kind": "strip", "p0": p}
            self.rebuild_lines()
            return True
        return False

    def _tool_move(self, e) -> bool:
        st = self._drag
        if not st:
            return False
        p = self.mapToScene(e.position().toPoint())
        a = self.analysis
        k = st["kind"]
        if k == "lane_pt":
            pts = st["orig"]
            j = st["j"]
            new = [list(q) for q in pts]
            new[j] = [p.x(), min(max(p.y(), (pts[j - 1][1] + 1) if j > 0 else -1e9), (pts[j + 1][1] - 1) if j + 1 < len(pts) else 1e9)]
            a.lane_lines[st["i"]] = new
        elif k == "lane":
            dx = p.x() - st["p0"].x()
            a.lane_lines[st["i"]] = [[q[0] + dx, q[1]] for q in st["orig"]]
        elif k in ("pline", "pline_end"):
            o = st["orig"]
            q = dict(o)
            if k == "pline":
                dy = p.y() - st["p0"].y()
                q["y0"], q["y1"] = o["y0"] + dy, o["y1"] + dy
            elif st["end"] == "a":
                q["x0"], q["y0"] = p.x(), p.y()
            else:
                q["x1"], q["y1"] = p.x(), p.y()
            a.protein_lines[st["i"]] = q
        elif k == "pline_new":
            st["p1"] = p
            if "item" not in st:
                pen = QtGui.QPen(QtGui.QColor(self.PROT_RED), 1.6, Qt.DashLine)
                pen.setCosmetic(True)
                st["item"] = self.scene().addLine(st["p0"].x(), st["p0"].y(), p.x(), p.y(), pen)
            st["item"].setLine(st["p0"].x(), st["p0"].y(), p.x(), p.y())
            return True
        elif k == "tick":
            self.marker_s["kda"][st["i"]]["y"] = float(p.y())
        elif k == "strip":
            st["p1"] = p
            if "item" not in st:
                pen = QtGui.QPen(QtGui.QColor(self.MARK_ORANGE), 1.5, Qt.DashLine)
                pen.setCosmetic(True)
                st["item"] = self.scene().addRect(QtCore.QRectF(st["p0"], p).normalized(), pen)
            st["item"].setRect(QtCore.QRectF(st["p0"], p).normalized())
            return True
        self.rebuild_lines()
        return True

    def _tool_release(self, e) -> bool:
        st, self._drag = self._drag, None
        if not st:
            return False
        p = self.mapToScene(e.position().toPoint())
        k = st["kind"]
        for key in ("item",):
            if key in st:
                self.scene().removeItem(st[key])
        moved = (p - st["p0"]).manhattanLength() > 2
        if k == "pline_new" and moved and abs(p.x() - st["p0"].x()) > 10:
            x0, y0, x1, y1 = st["p0"].x(), st["p0"].y(), p.x(), p.y()
            if x1 < x0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            self.proteinDrawn.emit(x0, y0, x1, y1)
        elif k == "strip" and moved and self.marker_item is not None:
            mx = self.marker_item.x()
            w = self.marker_item.pixmap().width()
            xa, xb = sorted((p.x() - mx, st["p0"].x() - mx))
            self.marker_s["strip_x0"], self.marker_s["strip_x1"] = int(max(0, xa)), int(min(w, xb))
            self.marker_s["show_strip"] = True
            self.markerEdited.emit("마커 영역을 정했습니다 — 영역 안을 클릭해 마커 밴드 위치를 넣으세요")
        elif k == "tick" and moved:
            self.marker_s["kda"].sort(key=lambda t: t["y"])
            self.markerEdited.emit("마커 밴드 위치를 옮겼습니다")
        elif k in ("lane_pt", "lane") and moved:
            self.analysis.lane_lines.sort(key=lambda ln: float(np.mean([q[0] for q in ln])))
            self.linesChanged.emit("레인 선을 고쳤습니다 — '선 기준으로 다시 검출'로 박스에 반영합니다")
        elif k in ("pline", "pline_end") and moved:
            self.linesChanged.emit("단백질 선을 고쳤습니다 — '선 기준으로 다시 검출'로 박스에 반영합니다")
        self.rebuild_lines()
        return True

    def add_lane_point(self, i, y):
        pts = sorted(self.analysis.lane_lines[i], key=lambda q: q[1])
        x = float(core.poly_x(pts, y))
        pts.append([x, float(y)])
        pts.sort(key=lambda q: q[1])
        self.analysis.lane_lines[i] = pts
        self.sel = {"kind": "lane", "i": i, "j": next(j for j, q in enumerate(pts) if q[1] == float(y))}
        self.rebuild_lines()
        self.linesChanged.emit("레인 선에 점을 넣었습니다 — 점을 끌어 선을 휘게 합니다")

    def delete_selection(self, whole: bool = False) -> bool:
        """⌫ on a line tool: the selected point, line or size mark."""
        sel = self.sel or {}
        a = self.analysis
        if sel.get("kind") == "lane" and sel.get("i") is not None and sel["i"] < len(a.lane_lines):
            pts = sorted(a.lane_lines[sel["i"]], key=lambda q: q[1])
            if sel.get("j") is not None and not whole and len(pts) > 2:
                del pts[sel["j"]]
                a.lane_lines[sel["i"]] = pts
                self.sel = {"kind": "lane", "i": sel["i"], "j": None}  # the line stays selected
                self.rebuild_lines()
                self.linesChanged.emit("점을 지웠습니다")
                return True
            else:
                del a.lane_lines[sel["i"]]
                msg = "레인 선을 지웠습니다"
        elif sel.get("kind") == "pline" and sel.get("i") is not None and sel["i"] < len(a.protein_lines):
            del a.protein_lines[sel["i"]]
            msg = "단백질 선을 지웠습니다"
        elif sel.get("kind") == "tick" and self.marker_s and sel.get("i") is not None:
            del self.marker_s["kda"][sel["i"]]
            self.sel = None
            self.rebuild_lines()
            self.markerEdited.emit("마커 밴드 표시를 지웠습니다")
            return True
        else:
            return False
        self.sel = None
        self.rebuild_lines()
        self.linesChanged.emit(msg)
        return True

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
        exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".json", ".blotproj")
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
            self.zoom(1.0015 ** e.angleDelta().y())
        else:
            super().wheelEvent(e)

    def set_draw_mode(self, mode: Optional[str]):
        self.draw_mode = mode
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag if mode else QtWidgets.QGraphicsView.RubberBandDrag)
        self.viewport().setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    def mousePressEvent(self, e):
        if self._space and e.button() == Qt.LeftButton:  # Space + drag: move the view
            self._pan = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if self.tool != "select" and not self.draw_mode and self._tool_press(e):
            return
        if self.draw_mode and e.button() == Qt.LeftButton and self.pix_item is not None:
            p = self.mapToScene(e.position().toPoint())
            pen = QtGui.QPen(QtGui.QColor("#ff6d00"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            item = (self.scene().addLine(p.x(), p.y(), p.x(), p.y(), pen) if self.draw_mode == "line"
                    else self.scene().addRect(p.x(), p.y(), 0, 0, pen))
            item.setZValue(20)
            self._draw = (p, item)
            return
        if self.resize_mode and e.button() == Qt.LeftButton:
            item = self.itemAt(e.position().toPoint())
            if isinstance(item, QtWidgets.QGraphicsSimpleTextItem):
                item = item.parentItem()
            if not isinstance(item, BandItem):  # a click outside the boxes ends the size editing
                self.resizeModeExited.emit()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d_ = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d_.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d_.y()))
            self.auto_fit = False
            return
        if self._tool_move(e):
            return
        if self._draw is not None:
            p0, item = self._draw
            p = self.mapToScene(e.position().toPoint())
            if self.draw_mode == "line":
                item.setLine(p0.x(), p0.y(), p.x(), p.y())
            else:
                item.setRect(QtCore.QRectF(p0, p).normalized())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.viewport().setCursor(Qt.OpenHandCursor if self._space else (Qt.ArrowCursor if self.tool == "select" else Qt.CrossCursor))
            return
        if self._tool_release(e):
            return
        if self._draw is not None:
            p0, item = self._draw
            self._draw = None
            p = self.mapToScene(e.position().toPoint())
            self.scene().removeItem(item)
            kind = self.draw_mode
            if (p - p0).manhattanLength() > 5:
                if kind == "crop":  # clip to the image
                    r = self.pix_item.sceneBoundingRect()
                    p0 = QtCore.QPointF(min(max(p0.x(), r.left()), r.right()), min(max(p0.y(), r.top()), r.bottom()))
                    p = QtCore.QPointF(min(max(p.x(), r.left()), r.right()), min(max(p.y(), r.top()), r.bottom()))
                self.drawn.emit(kind, p0.x(), p0.y(), p.x(), p.y())
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        p = self.mapToScene(e.position().toPoint())
        if self.tool == "lane":
            hit = self._lane_hit(e.position())
            if hit:
                self.add_lane_point(hit[0], p.y())
            return
        if self.tool != "select":
            return
        item = self.itemAt(e.position().toPoint())
        if isinstance(item, QtWidgets.QGraphicsSimpleTextItem):
            item = item.parentItem()
        if isinstance(item, BandItem):
            self.editRequested.emit(item)
            return
        if self.pix_item is not None and self.pix_item.sceneBoundingRect().contains(p):  # not on the marker beside the blot
            self.addRequested.emit(p.x(), p.y())

    def contextMenuEvent(self, e):
        item = self.itemAt(e.pos())
        if isinstance(item, QtWidgets.QGraphicsSimpleTextItem):
            item = item.parentItem()
        if isinstance(item, BandItem) and not item.isSelected():
            self.select_bands([item.band])
        self.contextRequested.emit(e.globalPos(), item if isinstance(item, BandItem) else None)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.viewport().setCursor(Qt.ArrowCursor if self.tool == "select" else Qt.CrossCursor)
            return
        super().keyReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            return
        if e.key() == Qt.Key_BracketLeft:  # [ zoom in, ] zoom out (spec 1.0)
            self.zoom(1.25)
            return
        if e.key() == Qt.Key_BracketRight:
            self.zoom(0.8)
            return
        if e.key() in (Qt.Key_Backspace, Qt.Key_Delete) and self.tool != "select":
            if self.delete_selection(whole=bool(e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))):
                return
        if e.key() == Qt.Key_Escape and self.draw_mode:
            self.drawn.emit("cancel", 0, 0, 0, 0)
            return
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
        self.fig = Figure(figsize=(4, 3) if aspect else (5, 3.5))
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

    style_fn = None  # () -> graph style (File → 설정 defaults + the sidebar choices)

    def _style(self):
        return self.style_fn() if self.style_fn else graph.style()

    def plot(self, series: Dict[str, np.ndarray], labels: List[str], conditions: List[str], by_condition: bool,
             ylabel: str, show_points: bool = True, title: str = ""):
        st = self._style()
        if not show_points and st.get("kind") == "bar_points":
            st["kind"] = "bars"
        graph.plot_series(self.fig, series, labels, conditions, by_condition, ylabel, st, title)
        self.canvas.draw_idle()

    def plot_bars(self, keys: List[str], series: Dict[str, Dict[str, object]], ylabel: str, show_points: bool = True):
        """Bars over named lanes; ``series[name]`` has mean, sd (or None) and points (per-image values or None)."""
        st = self._style()
        if not show_points and st.get("kind") == "bar_points":
            st["kind"] = "bars"
        graph.plot_summary(self.fig, keys, series, ylabel, st)
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


class FormulaEditor(QtWidgets.QWidget):
    """Edit a formula without typing names (used in the graph sidebar and in a dialog).

    ``terms`` is ``[(category, [(label, token), ...]), ...]``; ``variables`` is
    used for the live preview. ``validChanged`` tells whether the formula can be
    used, ``submitted`` is Enter on a valid formula.
    """

    validChanged = QtCore.Signal(bool)
    submitted = QtCore.Signal()
    QUICK = 16  # names of the first category (proteins) also as buttons

    def __init__(self, parent, terms, variables, formula: Optional[dict] = None, compact: bool = False):
        super().__init__(parent)
        self.terms = [(c, items) for c, items in terms if items]
        self.variables = variables
        formula = formula or {}
        self.tokens: List[str] = tokenize(str(formula.get("expr", "")))

        self.prefix_fn = lambda: ""  # quick buttons: "I2." to take a protein from another image
        self.preview_fn = None  # values -> preview text (grouped means in the graph sidebar)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.top = QtWidgets.QVBoxLayout()  # room for extra pickers above the protein buttons
        f = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(str(formula.get("name", "")))
        f.addRow("이름", self.name)
        lay.addLayout(f)

        self.display = QtWidgets.QLineEdit()
        self.display.setReadOnly(True)
        font = self.display.font()
        font.setPointSizeF(font.pointSizeF() * 1.3)
        self.display.setFont(font)
        self.display.setPlaceholderText("항목을 누르고, + - * / ( ) 와 숫자는 키보드나 버튼으로")
        self.display.installEventFilter(self)
        lay.addWidget(self.display)
        lay.addLayout(self.top)

        if self.terms:  # proteins as buttons: one click puts the name in
            cap = QtWidgets.QLabel(f"{self.terms[0][0]} — 눌러서 넣기")
            cap.setStyleSheet(f"color: {_tok('label-secondary')}; font-size: 11px;")
            lay.addWidget(cap)
            grid = QtWidgets.QGridLayout()
            cols = 3 if compact else 6
            grid.setSpacing(4)
            for i, (label, tok) in enumerate(self.terms[0][1][: self.QUICK]):
                b = QtWidgets.QPushButton(label)
                b.setFocusPolicy(Qt.NoFocus)
                b.setToolTip(tok)
                b.clicked.connect(lambda _=None, t=tok: self.push(self.prefix_fn() + t))
                grid.addWidget(b, i // cols, i % cols)
            lay.addLayout(grid)
        hb = QtWidgets.QHBoxLayout()
        for label, tok in (("+", "+"), ("−", "-"), ("×", "*"), ("÷", "/"), ("(", "("), (")", ")")):
            b = QtWidgets.QPushButton(label)
            b.setFixedWidth(30 if compact else 44)
            b.setStyleSheet("padding: 0; font-weight: 600;")
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=None, t=tok: self.push(t))
            hb.addWidget(b)
        b = QtWidgets.QPushButton("⌫")
        b.setToolTip("하나 지우기")
        b.setFixedWidth(30 if compact else 44)
        b.setStyleSheet("padding: 0;")
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(self.pop)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("C" if compact else "모두 지우기")
        b.setToolTip("모두 지우기")
        if compact:
            b.setFixedWidth(30)
            b.setStyleSheet("padding: 0;")
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(self.clear)
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)

        g = QtWidgets.QGroupBox("다른 항목 넣기 (밴드 값 등)")
        gl = QtWidgets.QVBoxLayout(g) if compact else QtWidgets.QHBoxLayout(g)
        self.category = QtWidgets.QComboBox()
        for c, _ in self.terms:
            self.category.addItem(c)
        self.category.currentIndexChanged.connect(self.fill_terms)
        gl.addWidget(self.category)
        self.term = QtWidgets.QComboBox()
        self.term.setMinimumWidth(80 if compact else 220)
        if compact:
            self.category.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self.term.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
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
        self.valid = False
        self.fill_terms()
        self.refresh()

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
                if self.valid:
                    self.submitted.emit()
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
            self._set_valid(False)
            return
        try:
            val = core.evaluate(expr, self.variables)
            vals = np.atleast_1d(np.asarray(val, float))
            shown = ", ".join("—" if not np.isfinite(v) else f"{v:.4g}" for v in vals[:10]) + (" …" if vals.size > 10 else "")
            self.preview.setText(self.preview_fn(vals) if self.preview_fn else f"미리보기: {shown}")
            self.preview.setStyleSheet(f"color: {_tok('green')};")
            self._set_valid(True)
        except Exception as e:
            self.preview.setText(f"미완성/오류: {e}")
            self.preview.setStyleSheet(f"color: {_tok('red')};")
            self._set_valid(False)

    def _set_valid(self, ok: bool):
        self.valid = ok
        self.validChanged.emit(ok)

    def result(self) -> dict:
        return {"name": self.name.text().strip() or self.expr(), "expr": self.expr(),
                "normalize": "1" if self.normalize.isChecked() else "0"}


class FormulaDialog(QtWidgets.QDialog):
    """The formula editor in its own window (tab 5)."""

    def __init__(self, parent, title: str, terms, variables, formula: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(620)
        lay = QtWidgets.QVBoxLayout(self)
        self.editor = FormulaEditor(self, terms, variables, formula)
        lay.addWidget(self.editor)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        for btn in bb.buttons():
            btn.setAutoDefault(False)
        self.ok = bb.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok.setEnabled(self.editor.valid)
        self.editor.validChanged.connect(self.ok.setEnabled)
        self.editor.submitted.connect(self.accept)
        lay.addWidget(bb)
        self.editor.display.setFocus()

    def result(self) -> dict:
        return self.editor.result()


RAW_NOTE = ("※ 정량은 항상 원본 TIF의 raw intensity로 합니다. 표시 대비와 검출용 LUT는 정량에 쓰지 않고, "
            "어두운 밴드 이미지는 선형 반전(최댓값 − 값)만, 그리고 선택한 배경 제거만 적용합니다.")


def raw_note() -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(RAW_NOTE)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color: {_tok('blue')}; font-size: 11px;")
    return lab


GROUP_COLORS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#c0ca33", "#6d4c41", "#d81b60", "#3949ab"]
LANES_BASIS = ""  # results per lane (no grouping set)
ROLE_SHORT = {core.ROLE_CONTROL: "Control", core.ROLE_TARGET: "Target", core.ROLE_IGNORE: "제외"}


class GroupAssignDialog(QtWidgets.QDialog):
    """Put the selected boxes in a group of a grouping set (one dialog, sensible defaults)."""

    def __init__(self, parent, a: core.Analysis, bands, default_set: Optional[str]):
        super().__init__(parent)
        self.setWindowTitle("측정 묶음에 넣기")
        self.a = a
        lay = QtWidgets.QVBoxLayout(self)
        roles = [a.rows[b.row].role for b in bands]
        info = QtWidgets.QLabel(
            f"선택한 박스 {len(bands)}개 (Control {roles.count(core.ROLE_CONTROL)} · Target {roles.count(core.ROLE_TARGET)})를 "
            "한 그룹으로 묶습니다.<br>레인은 그대로 두고, 정량 탭에서 이 묶음 세트를 고르면 그룹마다 Target ÷ Control 을 계산합니다.<br>"
            "<span style='color:#666'>한 박스는 세트마다 한 그룹에만 들어가고, 여러 세트에 동시에 들어갈 수 있습니다.</span>")
        info.setWordWrap(True)
        lay.addWidget(info)
        f = QtWidgets.QFormLayout()
        self.set_combo = QtWidgets.QComboBox()
        self.set_combo.setEditable(True)
        names = [g["name"] for g in a.groupings]
        self.set_combo.addItems(names or ["묶음 1"])
        if default_set in names:
            self.set_combo.setCurrentText(default_set)
        self.set_combo.setToolTip("새 이름을 쓰면 새 묶음 세트가 됩니다 (예: '샘플 짝', '시간별')")
        f.addRow("묶음 세트", self.set_combo)
        self.group_combo = QtWidgets.QComboBox()
        self.group_combo.setEditable(True)
        self.group_combo.setToolTip("새 이름을 쓰면 새 그룹, 있는 그룹을 고르면 그 그룹에 더합니다")
        f.addRow("그룹", self.group_combo)
        lay.addLayout(f)
        self.set_combo.currentTextChanged.connect(self._fill_groups)
        self._fill_groups()
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _fill_groups(self, *_):
        sname = self.set_combo.currentText().strip()
        g = core.grouping(self.a, sname)
        self.group_combo.clear()
        self.group_combo.addItem(core.next_group_name(self.a, sname))  # a new group first
        for x in (g["groups"] if g else []):
            self.group_combo.addItem(x)
        self.group_combo.setCurrentIndex(0)

    def values(self):
        return self.set_combo.currentText().strip(), self.group_combo.currentText().strip()


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


class MarkerStepDialog(QtWidgets.QDialog):
    """Step 2 after opening a blot: explain and add the matching marker (ladder) image."""

    THUMB_H = 260

    def __init__(self, parent, doc, candidate: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("2단계: 마커 이미지 추가")
        self.setAcceptDrops(True)
        self.doc = doc
        self.path: Optional[str] = None
        lay = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("2단계: 마커(ladder) 이미지 추가")
        f = title.font()
        f.setPointSizeF(f.pointSizeF() * 1.4)
        f.setBold(True)
        title.setFont(f)
        lay.addWidget(title)
        steps = QtWidgets.QLabel("1단계 블롯 열기 ✓  →  <b>2단계 마커 이미지</b>  →  3단계 밴드 확인 · 그룹 이름  →  정량 / Figure")
        steps.setStyleSheet(f"color: {_tok('label-secondary')};")
        lay.addWidget(steps)
        info = QtWidgets.QLabel(
            "방금 연 블롯과 <b>같은 막(membrane)을 같은 자리에서 찍은 마커 사진</b>(white-light / marker 채널)을 넣어 주세요.<br>"
            "보통 이미저가 블롯과 함께 저장하는 '…Marker…' 파일입니다. 크기(픽셀 수)가 달라도 <b>가로세로 비율이 비슷하면</b> 블롯 크기에 맞춰 불러옵니다.<br>"
            "마커는 밴드 박스를 잡을 때 옆에 참고용으로 보이고, Figure에서 ladder 띠와 kDa 눈금에 쓰입니다. 정량에는 쓰이지 않습니다.")
        info.setWordWrap(True)
        lay.addWidget(info)
        row = QtWidgets.QHBoxLayout()
        self.blot_thumb = QtWidgets.QLabel()
        self.marker_thumb = QtWidgets.QLabel("마커 TIF를 여기로 끌어다 놓거나\n아래 '마커 이미지 추가하기'를 누르세요")
        for lab in (self.blot_thumb, self.marker_thumb):
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumSize(300, self.THUMB_H)
        self.marker_thumb.setStyleSheet(f"border: 2px dashed {_tok('gray-2')}; color: {_tok('label-secondary')}; "
                                        f"background: {_tok('fill-quaternary')}; border-radius: 8px;")
        for caption, lab in ((f"블롯 (1단계): {doc.name}", self.blot_thumb), ("마커 (2단계)", self.marker_thumb)):
            col = QtWidgets.QVBoxLayout()
            c = QtWidgets.QLabel(caption)
            c.setStyleSheet("font-weight: bold;")
            col.addWidget(c)
            col.addWidget(lab, 1)
            row.addLayout(col, 1)
        lay.addLayout(row, 1)
        self.state = QtWidgets.QLabel()
        self.state.setWordWrap(True)
        self.state.setMinimumHeight(48)
        self.state.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(self.state)
        # orientation of the marker photo: one button per orientation that fits the blot
        self.orient_box = QtWidgets.QGroupBox("마커 방향 (블롯과 같은 방향을 고르세요)")
        self.orient_row = QtWidgets.QHBoxLayout(self.orient_box)
        self.orient_group = QtWidgets.QButtonGroup(self)
        self.orient_group.setExclusive(True)
        self.orient_box.setVisible(False)
        lay.addWidget(self.orient_box)
        self.marker_tf = {"rot90": 0, "flip": False}
        self._marker_raw = None
        self.add_btn = QtWidgets.QPushButton("마커 이미지 추가하기…")
        self.add_btn.setStyleSheet("font-size: 15px; font-weight: bold; padding: 10px;")
        self.add_btn.clicked.connect(self.choose)
        lay.addWidget(self.add_btn)
        bb = QtWidgets.QHBoxLayout()
        skip = QtWidgets.QPushButton("건너뛰기 (마커 없이 계속)")
        skip.clicked.connect(self.reject)
        bb.addWidget(skip)
        bb.addStretch(1)
        self.ok = QtWidgets.QPushButton("이 마커 사용")
        self.ok.setDefault(True)
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self.accept)
        bb.addWidget(self.ok)
        lay.addLayout(bb)
        self.blot_thumb.setPixmap(self._thumb(1.0 - figure._stretch(doc.signal, 0.5, 99.8, 0.6)))
        if candidate:
            self.set_path(candidate, found=True)
        else:
            self.state.setText("같은 폴더에서 짝이 되는 마커 파일을 찾지 못했습니다. 마커 사진이 없으면 '건너뛰기'를 누르세요. "
                               "(나중에 1번 탭의 '마커 열기…'로 넣을 수 있습니다)")
        self.resize(800, 640)

    def _thumb(self, a01) -> QtGui.QPixmap:
        qimg = to_qimage(a01, 0.0, 1.0)
        return QtGui.QPixmap.fromImage(qimg).scaledToHeight(self.THUMB_H, Qt.SmoothTransformation)

    def choose(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "마커(ladder) 이미지 선택", os.path.dirname(self.doc.image_path or ""),
                                                     "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if p:
            self.set_path(p)

    def _marker_display(self, m):
        x = figure._stretch(m, 1, 99.5, 1.0)
        return 1.0 - x if not core.needs_inversion(m) else x

    def _fill_orientations(self, cands):
        while self.orient_row.count():
            it = self.orient_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for b in self.orient_group.buttons():
            self.orient_group.removeButton(b)
        disp = self._marker_display(self._marker_raw)
        for i, c in enumerate(cands):
            btn = QtWidgets.QToolButton()
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            pm = QtGui.QPixmap.fromImage(to_qimage(core.orient(disp, c), 0.0, 1.0)).scaledToHeight(80, Qt.SmoothTransformation)
            btn.setIcon(QtGui.QIcon(pm))
            btn.setIconSize(pm.size())
            btn.setText(c["label"])
            btn.clicked.connect(lambda _=None, c=c: self.choose_orientation(c))
            self.orient_group.addButton(btn, i)
            self.orient_row.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        self.orient_row.addStretch(1)
        self.orient_box.setVisible(len(cands) > 1)

    def choose_orientation(self, c):
        self.marker_tf = {"rot90": c["rot90"], "flip": c["flip"]}
        self.marker_thumb.setPixmap(self._thumb(core.orient(self._marker_display(self._marker_raw), self.marker_tf)))

    def set_path(self, p: str, found: bool = False):
        try:
            m = core.load_image(p)
        except Exception as e:
            self.state.setText(f"<span style='color:#c62828'>열 수 없는 이미지입니다: {e}</span>")
            return
        base = self.doc.raw_orig if self.doc.raw_orig is not None else self.doc.raw  # before rotation/crop
        H, W = base.shape
        self._marker_raw = m
        cands = core.marker_orientations(m.shape, (H, W))
        self.marker_thumb.setStyleSheet(f"border: 2px solid {_tok('blue')}; background: white; border-radius: 8px;")
        name = os.path.basename(p)
        turned = ""
        if cands:
            self.choose_orientation(cands[0])
            self._fill_orientations(cands)
            if cands[0]["rot90"] % 2:
                turned = (f"<br><span style='color:#1565c0'>⟳ 마커의 가로·세로가 블롯과 바뀌어 있어 '{cands[0]['label']}'로 돌려 맞췄습니다. "
                          "아래 '마커 방향'에서 블롯과 같은 방향인지 확인하고, 아니면 다른 방향을 고르세요.</span>")
            m = core.orient(m, self.marker_tf)
        else:
            self.marker_thumb.setPixmap(self._thumb(self._marker_display(m)))
            self.orient_box.setVisible(False)
        mh, mw = m.shape
        if (mh, mw) == (H, W):
            msg = f"<span style='color:#2e7d32'>✓ {'자동으로 찾은 마커' if found else '선택한 마커'}: <b>{name}</b> — 크기가 블롯과 같습니다 ({mw}×{mh}).</span>"
        elif core._aspect_ok((mh, mw), (H, W)):
            msg = (f"<span style='color:#ef6c00'>⚠ {name}: 크기({mw}×{mh})가 블롯({W}×{H})과 달라 블롯 크기에 맞춰 불러옵니다 "
                   f"(비율 차이 {100 * core.aspect_error((mh, mw), (H, W)):.1f}%). 같은 막을 찍은 사진인지 확인하세요.</span>")
        else:
            self.state.setText(f"<span style='color:#c62828'>✗ {name}: 가로세로 비율이 블롯과 너무 달라 쓸 수 없습니다 "
                               f"({mw}×{mh} vs {W}×{H}, 차이 {100 * core.aspect_error((mh, mw), (H, W)):.0f}% — "
                               f"{100 * core.MARKER_ASPECT_TOL:.0f}%까지 허용). 같은 막을 찍은 마커 사진을 고르세요.</span>")
            self.path = None
            self.ok.setEnabled(False)
            return
        self.state.setText(msg + turned + ("<br>맞으면 '이 마커 사용', 다른 파일이면 '마커 이미지 추가하기…'로 바꾸세요." if found else ""))
        self.path = p
        self.ok.setEnabled(True)
        self.add_btn.setText("다른 마커 이미지 고르기…")

    @staticmethod
    def _image_urls(e):
        exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        return [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile().lower().endswith(exts)]

    def dragEnterEvent(self, e):
        if self._image_urls(e):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if self._image_urls(e):
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = self._image_urls(e)
        if urls:
            e.acceptProposedAction()
            # handle it after the drop has finished (no file dialogs inside the drag session)
            QtCore.QTimer.singleShot(0, lambda: self.set_path(urls[0]))


class ExportDialog(QtWidgets.QDialog):
    """파일 › 내보내기… (spec §9): pick what to save (data and pictures) and save it at once."""

    DATA = [("graph_csv", "정량 그래프에 쓰인 값", "켜 둔 항목·이미지·레인만 · 레인별 값 + 그룹 평균·SD·n", "graph_data.csv"),
            ("multi_csv", "여러 이미지 그래프에 쓰인 값", "수식 결과 포함", "multi_graph_data.csv"),
            ("results_csv", "모든 밴드 값", "모든 박스의 Raw·배경·Net, 그래프와 상관없이 전체", "all_bands.csv"),
            ("figure_csv", "Figure에 쓰인 데이터", "레인 이름, kDa 위치, Figure 안 정량 숫자", "figure_data.csv")]
    PICS = [("graph_img", "정량 그래프", "", "graph", ("png", "pdf", "svg")),
            ("multi_img", "여러 이미지 그래프", "", "multi_graph", ("png", "pdf", "svg")),
            ("figure_img", "Figure", "", "figure", ("tif", "png", "pdf", "svg")),
            ("overlay_img", "박스 표시한 블롯 이미지", "검출 확인용", "boxes", ("png",))]
    # (kept for older callers)
    ITEMS = [(k, n, d, f) for k, n, d, f in DATA]

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("내보내기")
        self.resize(1080, 680)
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        t = QtWidgets.QLabel("내보내기")
        t.setStyleSheet("font-size: 18px; font-weight: 700;")
        left.addWidget(t)
        left.addWidget(controls.hint("필요한 것만 골라 한 번에 저장합니다. 각 창 위의 '내보내기' 버튼은 그 창에 보이는 것 하나만 바로 저장합니다."))
        self.checks, self.fmts = {}, {}
        g = QtWidgets.QGroupBox("데이터 (CSV)")
        gl = QtWidgets.QVBoxLayout(g)
        for key, name, sub, _ in self.DATA:
            gl.addLayout(self._row(key, name, sub, None))
        left.addWidget(g)
        g = QtWidgets.QGroupBox("그림")
        gl = QtWidgets.QVBoxLayout(g)
        w_, h_ = main.graph_opts.get("size_mm") or [90, 60]
        for key, name, sub, _, fmts in self.PICS:
            if key == "graph_img":
                sub = f"{w_:g} × {h_:g} mm"
            gl.addLayout(self._row(key, name, sub, fmts))
        left.addWidget(g)
        for k in ("graph_csv", "graph_img"):
            self.checks[k].setChecked(True)
        g, f = controls.card()
        self.scope = QtWidgets.QComboBox()
        self.scope.addItem("정량 창에서 고른 이미지", "current")
        self.scope.addItem("열린 모든 이미지", "all")
        self.scope.currentIndexChanged.connect(self.refresh)
        f.addRow("범위", self.scope)
        self.dpi = _spin(72, 2400, int(main.prefs["figure"].get("export_dpi", 600)), 50, suffix=" dpi")
        f.addRow("해상도 (PNG·TIFF)", self.dpi)
        hb = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(prefs.export_folder(main.prefs, main.image_path or ""))
        hb.addWidget(self.folder, 1)
        b = QtWidgets.QPushButton("바꾸기…")
        b.clicked.connect(self.choose_folder)
        hb.addWidget(b)
        f.addRow("저장 위치", hb)
        self.prefix = QtWidgets.QLineEdit(main._base_name())
        self.prefix.textChanged.connect(self.refresh)
        f.addRow("파일 이름 앞부분", self.prefix)
        left.addWidget(g)
        left.addStretch(1)
        bb = QtWidgets.QHBoxLayout()
        self.count = QtWidgets.QLabel()
        theme.hint(self.count)
        bb.addWidget(self.count, 1)
        b = QtWidgets.QPushButton("취소")
        b.clicked.connect(self.reject)
        bb.addWidget(b)
        self.go = controls.prominent(QtWidgets.QPushButton("내보내기"))
        self.go.setDefault(True)
        self.go.clicked.connect(self.run)
        bb.addWidget(self.go)
        left.addLayout(bb)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        theme.unify(lw)
        root.addWidget(lw, 5)
        right = QtWidgets.QVBoxLayout()
        self.preview_title = QtWidgets.QLabel()
        self.preview_title.setStyleSheet("font-weight: 600;")
        right.addWidget(self.preview_title)
        self.table = QtWidgets.QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        right.addWidget(self.table, 1)
        self.note = controls.hint("")
        right.addWidget(self.note)
        root.addLayout(right, 6)
        self.refresh()

    def _row(self, key, name, sub, fmts):
        hb = QtWidgets.QHBoxLayout()
        cb = QtWidgets.QCheckBox()
        cb.toggled.connect(self.refresh)
        self.checks[key] = cb
        hb.addWidget(cb)
        v = QtWidgets.QVBoxLayout()
        v.setSpacing(0)
        lab = QtWidgets.QLabel(name)
        lab.setStyleSheet("font-weight: 600;")
        v.addWidget(lab)
        if sub:
            v.addWidget(controls.hint(sub))
        hb.addLayout(v, 1)
        if fmts:
            seg = controls.Segmented([(f, {"tif": "TIFF"}.get(f, f.upper())) for f in fmts], small=True)
            pref = self.main.prefs["export"].get("image_format", "png")
            seg.set_value(pref if pref in fmts else fmts[0])
            seg.changed.connect(lambda _=None: self.refresh())
            self.fmts[key] = seg
            hb.addWidget(seg)
        else:
            hb.addWidget(QtWidgets.QLabel("한 파일"))
        return hb

    def choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "저장 폴더", self.folder.text())
        if d:
            self.folder.setText(d)

    def files(self):
        pre = self.prefix.text().strip() or "blot"
        out = [(key, f"{pre}_{fn}") for key, _, _, fn in self.DATA if self.checks[key].isChecked()]
        out += [(key, f"{pre}_{fn}.{self.fmts[key].value()}") for key, _, _, fn, _ in self.PICS if self.checks[key].isChecked()]
        return out

    def refresh(self, *_):
        if not hasattr(self, "go"):  # still being built
            return
        files = self.files()
        self.count.setText(f"파일 {len(files)}개를 저장합니다" if files else "고른 항목이 없습니다")
        self.go.setEnabled(bool(files))
        m = self.main
        pre = self.prefix.text().strip() or "blot"
        if self.checks["figure_csv"].isChecked() and not self.checks["graph_csv"].isChecked():
            header, rows = m.figure_editor.numbers_table() if m._figure_ready() else (["-"], [])
            self.preview_title.setText(f"CSV 미리보기 · {pre}_figure_data.csv")
            self.note.setText("Figure의 레인 이름·kDa 위치·숫자 줄입니다.")
        elif self.checks["results_csv"].isChecked() and not self.checks["graph_csv"].isChecked():
            header, rows = ["(모든 밴드 값은 저장할 때 만듭니다)"], []
            self.preview_title.setText(f"CSV 미리보기 · {pre}_all_bands.csv")
            self.note.setText("")
        else:
            header, rows, notes = m.graph_data(self.scope.currentData())
            self.preview_title.setText(f"CSV 미리보기 · {pre}_graph_data.csv")
            self.note.setText("맨 위 # 주석 줄: " + " / ".join(n.lstrip("# ") for n in notes))
        t = self.table
        t.clear()
        t.setColumnCount(len(header))
        t.setHorizontalHeaderLabels(header)
        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            summary = row and row[0] == ""
            for c, x in enumerate(row):
                it = QtWidgets.QTableWidgetItem(str(x))
                if summary:
                    it.setBackground(theme._hex(theme.TOKENS[theme.scheme()]["status-success-tint"]))
                t.setItem(r, c, it)
        t.resizeColumnsToContents()

    def run(self):
        m = self.main
        folder = os.path.expanduser(self.folder.text().strip())
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "내보내기", f"폴더를 만들 수 없습니다: {e}")
            return
        files = [(k, os.path.join(folder, n)) for k, n in self.files()]
        exist = [os.path.basename(p) for _, p in files if os.path.exists(p)]
        if exist and QtWidgets.QMessageBox.question(self, "내보내기", "이미 있는 파일을 덮어쓸까요?\n" + "\n".join(exist)) \
                != QtWidgets.QMessageBox.Yes:
            return
        scope, dpi, done, errors = self.scope.currentData(), self.dpi.value(), [], []
        for key, p in files:
            try:
                if key == "graph_csv":
                    m.write_graph_csv(p, scope)
                elif key == "multi_csv":
                    m.write_graph_csv(p, "multi")
                elif key == "results_csv":
                    m.write_results_csv(p)
                elif key in ("figure_csv", "figure_img"):
                    if not m._figure_ready():
                        raise ValueError("Figure를 만들 수 없습니다")
                    if key == "figure_csv":
                        m.figure_editor.write_numbers_csv(p)
                    else:
                        m.figure_editor.write_figure(p, dpi)
                elif key == "graph_img":
                    m.write_graph_image(p, m.plot, dpi)
                elif key == "multi_img":
                    m.write_graph_image(p, m.multi_plot, dpi)
                elif key == "overlay_img":
                    m.write_overlay(p)
                done.append(os.path.basename(p))
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        m.remember_folder(folder)
        if errors:
            QtWidgets.QMessageBox.warning(self, "내보내기", "일부를 저장하지 못했습니다:\n" + "\n".join(errors))
        m.statusBar().showMessage(f"{len(done)}개 파일을 저장했습니다 ({folder}): " + ", ".join(done))
        self.accept()


class SettingsDialog(QtWidgets.QDialog):
    """File → 설정: the default look of new graphs and figures, and export defaults."""

    def __init__(self, main, current: dict):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("설정 — 그래프·Figure 기본 모양")
        self.resize(1000, 640)
        self.p = json.loads(json.dumps(current))
        self.apply_figure = False
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        left.addWidget(self.tabs, 1)
        info = QtWidgets.QLabel("여기 값은 새로 만드는 그래프·Figure의 초기값입니다(이 컴퓨터에 저장). 이미 만든 Figure는 그대로이고, "
                                "'지금 열린 그림에도 적용'으로 열린 그림도 바꿀 수 있습니다.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {_tok('label-secondary')};")
        left.addWidget(info)
        bb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("PRISM 기본값으로 되돌리기")
        b.clicked.connect(self.reset)
        bb.addWidget(b)
        bb.addStretch(1)
        b = QtWidgets.QPushButton("지금 열린 그림에도 적용")
        b.setToolTip("저장하고, 지금 열린 그래프와 Figure도 이 값으로 바꿉니다")
        b.clicked.connect(self.save_and_apply)
        bb.addWidget(b)
        b = QtWidgets.QPushButton("취소")
        b.clicked.connect(self.reject)
        bb.addWidget(b)
        b = QtWidgets.QPushButton("저장")
        b.setDefault(True)
        b.setStyleSheet("font-weight: bold; padding: 5px 14px;")
        b.clicked.connect(self.accept)
        bb.addWidget(b)
        left.addLayout(bb)
        root.addLayout(left, 5)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>그래프 미리보기</b> (바꾸면 바로 반영)"))
        self.canvas = PlotCanvas()
        self.canvas.toolbar.setVisible(False)
        self.canvas.style_fn = lambda: graph.style(self.p["graph"])
        right.addWidget(self.canvas, 1)
        root.addLayout(right, 4)
        self.build()

    def build(self):
        self.tabs.clear()
        g, fg, ex = self.p["graph"], self.p["figure"], self.p["export"]
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        f.addRow("글꼴", self._font(g, "font"))
        for lab, key in (("축 이름 크기", "label_size"), ("눈금 숫자 크기", "tick_size"), ("범례 글자 크기", "legend_size"),
                         ("제목 크기", "title_size")):
            f.addRow(lab, self._num(g, key, 4, 36, 0.5, 1, " pt"))
        f.addRow(self._check(g, "bold", "글자 굵게"))
        for lab, key in (("축 선 굵기", "axis_lw"), ("막대 테두리 굵기", "bar_lw"), ("오차 막대 굵기", "err_lw")):
            f.addRow(lab, self._num(g, key, 0.2, 6, 0.1, 1, " pt"))
        f.addRow("눈금 길이", self._num(g, "tick_len", 0, 20, 0.5, 1, " pt"))
        f.addRow("눈금 방향", self._combo(g, "tick_dir", [("out", "바깥"), ("in", "안쪽")]))
        f.addRow("점 크기", self._num(g, "point_size", 2, 200, 2, 0, ""))
        f.addRow("오차 막대 캡 길이", self._num(g, "cap", 0, 20, 1, 0, ""))
        f.addRow("기본 형태", self._combo(g, "kind", prefs.GRAPH_KINDS))
        f.addRow("오차 막대", self._combo(g, "err", prefs.ERRORS))
        f.addRow("색", self._combo(g, "colors", prefs.COLOR_SCHEMES))
        f.addRow(self._check(g, "y0", "Y축 0부터"))
        self.tabs.addTab(self._scrolled(w), "그래프")
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        f.addRow("글꼴", self._font(fg, "font"))
        for key, (label, _) in figure.FONT_CATEGORIES.items():
            f.addRow(f"{label} 글자", self._num(fg["fonts"], key, 3, 36, 0.5, 1, " pt"))
        f.addRow(self._check(fg, "label_bold", "단백질 이름(IB) 굵게"))
        f.addRow("패널 테두리 굵기", self._num(fg, "frame_lw", 0, 3, 0.1, 1, " pt"))
        f.addRow("그룹 막대 굵기", self._num(fg, "group_bar_lw", 0.2, 6, 0.1, 1, " pt"))
        mm = _spin(25, 300, round(float(fg.get("width_in", 3.2)) * 25.4), 1, suffix=" mm")
        mm.valueChanged.connect(lambda v: (fg.__setitem__("width_in", round(v / 25.4, 3)), self.preview()))
        f.addRow("기본 너비", mm)
        f.addRow("패널 간격", self._num(fg, "panel_gap_in", 0, 1, 0.01, 2, " in"))
        f.addRow("저장 해상도", self._num(fg, "export_dpi", 72, 2400, 50, 0, " dpi"))
        self.tabs.addTab(self._scrolled(w), "Figure")
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        f.addRow("이미지 기본 형식", self._combo(ex, "image_format", prefs.IMAGE_FORMATS))
        f.addRow("기본 저장 위치", self._combo(ex, "folder", prefs.FOLDERS))
        self.tabs.addTab(self._scrolled(w), "내보내기")
        self.preview()

    @staticmethod
    def _scrolled(w):
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QtWidgets.QFrame.NoFrame)
        sc.setWidget(w)
        return sc

    def _num(self, d, key, lo, hi, step, dec, suffix):
        sp = _spin(lo, hi, d.get(key, lo), step, dec if dec else None, suffix)
        sp.valueChanged.connect(lambda v: (d.__setitem__(key, int(v) if not dec else float(v)), self.preview()))
        return sp

    def _check(self, d, key, text):
        cb = QtWidgets.QCheckBox(text)
        cb.setChecked(bool(d.get(key)))
        cb.toggled.connect(lambda on: (d.__setitem__(key, on), self.preview()))
        return cb

    def _combo(self, d, key, items):
        cb = QtWidgets.QComboBox()
        for k, text in items:
            cb.addItem(text, k)
        cb.setCurrentIndex(max(0, cb.findData(d.get(key))))
        cb.currentIndexChanged.connect(lambda _=None: (d.__setitem__(key, cb.currentData()), self.preview()))
        return cb

    def _font(self, d, key):
        cb = QtWidgets.QComboBox()
        cb.setEditable(True)
        cb.addItems(["Arial", "Helvetica", "Helvetica Neue", "Times New Roman", "Calibri"])
        cb.setCurrentText(str(d.get(key, "Arial")))
        cb.currentTextChanged.connect(lambda t: (d.__setitem__(key, t), self.preview()))
        return cb

    def preview(self, *_):
        m = self.main
        series = {k: v for k, v in m.results.items() if m.series_on.get(k, True) and np.size(v) > 1}
        if series:
            a, _ = m.basis_of(m.doc)
            labels = list(a.lane_labels)
        else:  # nothing measured yet: example numbers
            labels = ["Vehicle", "Vehicle", "Drug A", "Drug A", "Drug A", "Drug B", "Drug B", "Drug B"]
            series = {"Target A / Control": np.array([1.0, 0.8, 1.5, 1.3, 1.2, 2.1, 1.8, 2.4]),
                      "Target B / Control": np.array([1.0, 1.1, 0.7, 0.6, 0.9, 0.5, 0.4, 0.6])}
        self.canvas.plot(series, labels, labels, True, "Relative intensity")

    def reset(self):
        self.p = prefs.defaults()
        self.build()

    def save_and_apply(self):
        self.apply_figure = True
        self.accept()

    def result_prefs(self) -> dict:
        return self.p


class ImageDoc:
    """One opened blot image with its own boxes, settings and measurements."""

    def __init__(self, invert: int = 0, bg: Optional[core.BackgroundSettings] = None):
        self.image_path: Optional[str] = None
        self.project: str = ""  # name of the project file the image came from (several projects open together)
        self.raw: Optional[np.ndarray] = None
        self.signal: Optional[np.ndarray] = None
        self.corrected: Optional[np.ndarray] = None
        self._corr_key = None
        self.analysis = core.Analysis()
        self.measurements: List[core.Measurement] = []
        self.invert = invert  # index of the "밴드 색" combo
        self.bg = core.BackgroundSettings(**vars(bg)) if bg else core.BackgroundSettings()
        self.ref_lane: Optional[int] = 0
        self.basis: str = LANES_BASIS  # grouping set used to pair control and target ("" = lanes)
        self.basis_ref: Dict[str, Optional[int]] = {}  # reference group (=1.0) per grouping set
        self.figure: Optional[dict] = None  # Figure 만들기 settings (saved with the analysis)
        self.marker: Optional[np.ndarray] = None  # ladder/marker image (not saved, path is)
        self.mode = core.MODE_WESTERN  # or core.MODE_GEL
        self.transform = core.default_transform()  # rotation / crop applied to the file's pixels
        self.marker_tf = {"rot90": 0, "flip": False}  # how the marker photo is turned to match the blot
        self.raw_orig: Optional[np.ndarray] = None  # the file as loaded (before rotation/crop)
        self.marker_orig: Optional[np.ndarray] = None

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
        self.prefs = prefs.load()  # File → 설정: default look of graphs/figures, export defaults
        figure.set_user_defaults(self.prefs["figure"])
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
        self._tilt_done = False
        self._tagged = ([], [])  # measurement groups of the boxes, carried over when detecting again
        self.ref_name: Optional[str] = None  # 1.0 기준 (lane / group name)
        self._ref_default = True  # first image: its first lane becomes the reference
        self.ctrl_name: Optional[str] = None  # Control protein (None = the one with the Control role)
        self.formulas_all: List[Dict[str, str]] = []  # formulas of the graph sidebar (every image)
        self._pooled = []
        self.project_path: Optional[str] = None  # where the project was last saved/opened
        self._saved_fp: Optional[str] = None  # state at that moment (unsaved-change mark)

        self.view = BlotView()
        self.view.changed.connect(self.on_boxes_changed)
        self.view.selectionChangedBands.connect(self.on_view_selection)
        self.view.contextRequested.connect(self.show_band_menu)
        self.view.addRequested.connect(self.add_band_at)
        # dropped files are opened once the drop has finished: the marker step window runs its own
        # event loop and, inside a drag session (macOS), its file dialog and drops do not work
        self.view.filesDropped.connect(lambda ps: QtCore.QTimer.singleShot(0, lambda: self.open_paths(ps)))
        self.view.editRequested.connect(self.edit_band_size)
        self.view.drawn.connect(lambda k, *xy: self.set_draw_mode(None) if k == "cancel" else self.on_drawn(k, *xy))
        self.view.resizeModeExited.connect(lambda: self.resize_mode.setChecked(False))

        # ---- 1.0 layout: sidebar | canvas | inspector (spec §1–4); results and figure in their own windows
        self.auto_lanes = QtWidgets.QCheckBox()  # lanes follow the boxes (always on; no control any more)
        self.auto_lanes.setChecked(True)
        self._groupings_widget = self._build_groupings()  # 측정 묶음 세트, edited from the quantification window
        from .figure_window import FigureEditor, FigureWindow
        from .quant_window import QuantWindow

        self.figure_editor = FigureEditor(self)
        self.fig_win = FigureWindow(self, self.figure_editor)
        self.quant_win = QuantWindow(self)
        self.step = 0

        self.contrast_lo = QtWidgets.QSlider(Qt.Horizontal)
        self.contrast_hi = QtWidgets.QSlider(Qt.Horizontal)
        for s, v in ((self.contrast_lo, 0), (self.contrast_hi, 1000)):
            s.setRange(0, 1000)
            s.setValue(v)
            s.valueChanged.connect(lambda _=None: self.update_display(reset_zoom=False))
        self.show_corrected = QtWidgets.QCheckBox("배경 제거 이미지 보기")
        self.show_corrected.toggled.connect(lambda _=None: self.update_display(reset_zoom=False))
        self.show_labels = QtWidgets.QCheckBox("박스 ID 보기")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(self.toggle_labels)
        self.show_marker = QtWidgets.QCheckBox("마커 같이 보기")
        self.show_marker.setChecked(True)
        self.show_marker.toggled.connect(lambda _=None: (self.refresh_marker_view(), self.view.fit()))
        self.show_lane_lines = QtWidgets.QCheckBox("레인 선 · 단백질 선 보기")
        self.show_lane_lines.setChecked(True)
        self.show_lane_lines.toggled.connect(lambda on: (setattr(self.view, "show_lanes", on), self.view.rebuild_lines()))
        self.resize_mode = QtWidgets.QCheckBox("박스 크기 수정")
        self.resize_mode.toggled.connect(self.on_resize_mode)

        self.sidebar = self._build_sidebar()
        canvas = QtWidgets.QWidget()
        canvas.setObjectName("hig_content")
        canvas.setAttribute(Qt.WA_StyledBackground, True)
        cl = QtWidgets.QVBoxLayout(canvas)
        cl.setContentsMargins(20, 14, 20, 8)
        cl.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        self.canvas_title = QtWidgets.QLabel("검출")
        self.canvas_title.setStyleSheet("font-size: 20px; font-weight: 700;")
        head.addWidget(self.canvas_title)
        head.addStretch(1)
        self.canvas_note = QtWidgets.QLabel()
        theme.hint(self.canvas_note)
        head.addWidget(self.canvas_note)
        self.tool_seg = controls.Segmented([("select", "선택"), ("lane", "레인 선"), ("protein", "단백질 선"), ("marker", "마커 영역")])
        self.tool_seg.changed.connect(self.set_tool)
        head.addWidget(self.tool_seg)
        cl.addLayout(head)
        cl.addWidget(self.view, 1)
        st = QtWidgets.QHBoxLayout()
        self.key_hint = QtWidgets.QLabel()
        theme.hint(self.key_hint)
        st.addWidget(self.key_hint, 1)
        self.img_info = QtWidgets.QLabel()
        theme.hint(self.img_info)
        st.addWidget(self.img_info)
        cl.addLayout(st)

        self.inspector = QtWidgets.QStackedWidget()
        self.inspector.addWidget(self._inspector_page(self._build_detect_inspector()))
        self.inspector.addWidget(self._inspector_page(self._build_lanes_inspector()))
        insp = QtWidgets.QWidget()
        insp.setObjectName("hig_inspector")
        insp.setAttribute(Qt.WA_StyledBackground, True)
        il = QtWidgets.QVBoxLayout(insp)
        il.setContentsMargins(*theme.INSPECTOR_BOX_MARGINS)  # cards keep 12 px from the box edge
        il.addWidget(self.inspector)
        insp.setFixedWidth(theme.INSPECTOR_W)

        from .ui.hig_widgets import PaneSplitter

        split = PaneSplitter()  # sidebar = the window's base; canvas and inspector float as rounded boxes
        split.setChildrenCollapsible(False)
        self.split = split
        split.addWidget(self.sidebar)
        split.addWidget(canvas)
        split.addWidget(insp)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setMaximumWidth(280)
        self.setCentralWidget(split)
        self.view.linesChanged.connect(self.on_lines_changed)
        self.view.markerEdited.connect(self.on_marker_edited)
        self.view.tickSelected.connect(self.on_tick_selected)
        self.view.proteinDrawn.connect(self.add_protein_line)
        self.view.laneMenu.connect(self.show_lane_menu)
        self.view.proteinMenu.connect(self.show_protein_menu)
        self.view.toolHint.connect(lambda m: self.statusBar().showMessage(m))
        self.zoom_chip = controls.ZoomChip(self.view, lambda: self.view.zoom(0.8), lambda: self.view.zoom(1.25), self.view.fit)
        self.view.zoomChanged.connect(self.zoom_chip.set_zoom)
        self.set_step(0)
        self._build_menu()
        self.statusBar().showMessage("TIF 파일을 열어 시작하세요 (파일 → 열기 또는 창에 끌어다 놓기)")
        self.setAcceptDrops(True)

        del_sc = QtGui.QShortcut(QtGui.QKeySequence.Delete, self.view)
        del_sc.activated.connect(self.delete_selected)
        bs_sc = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Backspace), self.view)
        bs_sc.activated.connect(self.delete_selected)
        self.refresh_doc_tabs()
        self._title_timer = QtCore.QTimer(self)
        self._title_timer.setInterval(1500)
        self._title_timer.timeout.connect(self.update_title)
        self._title_timer.start()

    # ---------------------------------------------------------------- 1.0 sidebar / steps / inspector
    def _inspector_page(self, page):
        theme.unify(page)
        return self._scroll(page)

    def _build_sidebar(self):
        side = QtWidgets.QWidget()
        side.setObjectName("hig_sidebar")
        side.setAttribute(Qt.WA_StyledBackground, True)
        lay = QtWidgets.QVBoxLayout(side)
        lay.setContentsMargins(10, 12, 10, 12)
        lay.setSpacing(2)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(controls.section("이미지"))
        hb.addStretch(1)
        b = QtWidgets.QToolButton()
        b.setText("＋")
        b.setToolTip("이미지 열기… (여러 개, 끌어다 놓아도 됩니다)")
        b.clicked.connect(self.open_image)
        hb.addWidget(b)
        lay.addLayout(hb)
        self.img_list = controls.SideList()
        self.img_list.currentRowChanged.connect(self.switch_doc)
        self.img_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.img_list.customContextMenuRequested.connect(self._img_menu)
        lay.addWidget(self.img_list)
        lay.addWidget(controls.section("단계"))
        self.step_list = controls.SideList()
        self.step_list.currentRowChanged.connect(self.set_step)
        lay.addWidget(self.step_list)
        lay.addStretch(1)  # 정량·그래프 and Figure open from the menus next to 파일 (mockup 1.0.1)
        lay.addWidget(controls.hint("저장·불러오기는 모두 위쪽 메뉴 막대의 파일 메뉴에 있습니다."))
        return side

    def _img_menu(self, pos):
        row = self.img_list.indexAt(pos).row()
        if row < 0:
            return
        m = QtWidgets.QMenu(self)
        m.addAction("이 이미지 닫기", lambda: self.close_doc(row))
        m.exec(self.img_list.mapToGlobal(pos))

    def refresh_steps(self):
        a = self.analysis
        rows = [("1  검출", f"레인 {a.n_lanes} · 줄 {len(a.rows)}" if a.bands else ""), ("2  레인 / 단백질", "")]
        self.step_list.set_rows(rows, self.step)

    def set_step(self, i: int):
        if i < 0:
            return
        self.step = i
        self.inspector.setCurrentIndex(i)
        self.canvas_title.setText("검출" if i == 0 else "레인 / 단백질")
        self.tool_seg.setVisible(i == 0)
        self.canvas_note.setVisible(i == 1)
        self.canvas_note.setText("박스를 클릭·드래그로 선택, 더블클릭으로 크기 수정, 끌어서 이동")
        if i == 1 and self.view.tool != "select":
            self.tool_seg.set_value("select")
            self.set_tool("select")
        self.refresh_steps()
        self.update_key_hint()

    def set_tool(self, tool: str):
        self.view.set_tool(tool)
        self.update_key_hint()
        msg = {"lane": "레인 선: 빈 곳 클릭 = 새 선 · 점 끌기 = 휘기 · 선 더블클릭 = 점 추가 · ⌫ = 지우기 · 우클릭 = 메뉴",
               "protein": "단백질 선: 가로로 끌어 새 빨간 선 · 양끝 네모로 기울이기 · 선을 끌어 옮기기 · ⌫ = 지우기",
               "marker": "마커 영역: 마커 이미지에서 끌어 영역을 정하고, 영역 안을 클릭해 마커 밴드 위치를 넣으세요",
               "select": "선택: 박스를 클릭·드래그로 고르고 끌어서 옮깁니다"}[tool]
        self.statusBar().showMessage(msg)

    def update_key_hint(self):
        tail = ("드래그: 선택 · ROI 수정 · 영역 설정" if self.step == 0
                else "더블클릭: 크기 수정 · ⌫ 지우기 · 우클릭: 그룹·병합")
        self.key_hint.setText(f"[ 확대   ] 축소   Space + 드래그: 화면 이동   {tail}")

    # ---- inspector: 검출
    def _take_row(self, widget):
        """Take ``widget`` out of the old settings form (the row's label goes with it)."""
        par = widget.parentWidget()
        lay = par.layout() if par is not None else None
        if isinstance(lay, QtWidgets.QFormLayout):
            res = lay.takeRow(widget)
            if res.labelItem is not None and res.labelItem.widget() is not None:
                res.labelItem.widget().deleteLater()
        return widget

    def _build_detect_inspector(self):
        old = self._build_detect_tab()  # every former setting; the ones below move into the cards
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        g, f = controls.card("검출")
        self.kind_seg = controls.Segmented([(core.MODE_WESTERN, "Western"), (core.MODE_GEL, "DNA gel")], small=True)
        self.kind_seg.changed.connect(lambda m: self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(m))))
        self.mode_combo.currentIndexChanged.connect(lambda _: self.kind_seg.set_value(self.mode_combo.currentData()))
        f.addRow("종류", self.kind_seg)
        f.addRow("민감도", controls.slider_row(self._take_row(self.sens)))
        self._take_row(self.auto_tilt)
        self.auto_tilt.setText("")
        theme.switch(self.auto_tilt)
        f.addRow(self._row_title("기울기 자동 보정", "블롯과 마커를 같은 각도로 통째로 돌립니다"), self.auto_tilt)
        hb = QtWidgets.QHBoxLayout()
        self.angle_main = _spin(-45.0, 45.0, 0.0, 0.1, 1, "°")
        self.angle_main.setProperty("hig_plain", True)  # a small value field, as in the mockup
        self.angle_main.setFixedWidth(84)
        self.angle_main.setKeyboardTracking(False)
        self.angle_main.valueChanged.connect(lambda v: abs(self.angle_spin.value() - v) > 1e-6 and self.angle_spin.setValue(v))
        self.angle_spin.valueChanged.connect(lambda v: (self.angle_main.blockSignals(True), self.angle_main.setValue(v),
                                                        self.angle_main.blockSignals(False)))
        hb.addWidget(self.angle_main)
        lab = QtWidgets.QLabel("자동")
        theme.hint(lab)
        hb.addWidget(lab)
        self.angle_auto = controls.switch("", True)
        self.angle_auto.toggled.connect(lambda on: self.auto_tilt.isChecked() != on and self.auto_tilt.setChecked(on))
        self.auto_tilt.toggled.connect(lambda on: self.angle_auto.isChecked() != on and self.angle_auto.setChecked(on))
        hb.addWidget(self.angle_auto)
        f.addRow("회전 각도", hb)
        lay.addWidget(g)

        g, f = controls.card()
        self.lines_title = QtWidgets.QLabel()
        self.lines_title.setStyleSheet("font-weight: 600;")
        f.addRow(self.lines_title)
        f.addRow(controls.hint("선을 고친 뒤 다시 검출하면 그 선을 기준으로 박스를 찾습니다."))
        b = QtWidgets.QPushButton("레인 선 자동으로 찾기")
        b.clicked.connect(self.find_lane_lines)
        b2 = controls.prominent(QtWidgets.QPushButton("선 기준으로 다시 검출"))
        b2.clicked.connect(lambda: self.detect(use_lines=True))
        hb = QtWidgets.QHBoxLayout()
        hb.setSpacing(6)
        for x in (b, b2):  # two compact capsules side by side (mockup)
            x.setStyleSheet("padding: 0 9px;")
            x.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        hb.addWidget(b)
        hb.addWidget(b2)
        f.addRow(hb)
        lay.addWidget(g)

        g, f = controls.card("단백질 선")
        self.only_on_lines = controls.switch("", False)
        self.only_on_lines.setToolTip("켜면 다시 검출할 때 빨간 선과 레인 선이 만나는 자리에만 박스를 놓습니다")
        f.addRow("빨간 선 위에서만 찾기", self.only_on_lines)
        self.pline_box = QtWidgets.QVBoxLayout()
        f.addRow(self.pline_box)
        lay.addWidget(g)

        g, f = controls.card("마커")
        hb = QtWidgets.QHBoxLayout()
        self.marker_name = QtWidgets.QLabel("없음")
        self.marker_name.setWordWrap(True)
        self.marker_name.setMinimumWidth(40)
        theme.hint(self.marker_name)
        hb.addWidget(self.marker_name, 1)
        b = QtWidgets.QPushButton("바꾸기…")
        b.clicked.connect(self.open_marker)
        hb.addWidget(b)
        f.addRow("마커 이미지", hb)
        self.ladder_combo = QtWidgets.QComboBox()
        self.ladder_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ladder_combo.setMinimumContentsLength(10)
        self.ladder_combo.currentIndexChanged.connect(lambda _: self.apply_ladder())
        f.addRow("래더", self.ladder_combo)
        hb = QtWidgets.QHBoxLayout()
        self.first_kda = QtWidgets.QDoubleSpinBox()
        self.first_kda.setRange(0, 100000)
        self.first_kda.setDecimals(0)
        self.first_kda.setSpecialValueText("—")
        self.first_kda.setKeyboardTracking(False)
        self.first_kda.valueChanged.connect(lambda _: self.apply_ladder())
        hb.addWidget(self.first_kda)
        self.kda_unit = QtWidgets.QLabel("kDa")
        hb.addWidget(self.kda_unit)
        f.addRow("첫 마커 크기", hb)
        self.kda_chips = controls.ChipRow()
        self.kda_chips.clicked.connect(self.edit_kda_chip)
        f.addRow(controls.hint("마커 영역 도구로 영역을 그리고 밴드를 클릭하세요. 나머지는 래더 순서로 자동 · 칩을 눌러 고침"))
        f.addRow(self.kda_chips)
        lay.addWidget(g)

        disp = QtWidgets.QWidget()
        dl = QtWidgets.QFormLayout(disp)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.contrast_lo)
        hb.addWidget(self.contrast_hi)
        dl.addRow("표시 대비", hb)
        for w in (self.show_corrected, self.show_labels, self.show_marker, self.show_lane_lines):
            dl.addRow(w)
        body = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(old)
        gd = QtWidgets.QGroupBox("보기")
        QtWidgets.QVBoxLayout(gd).addWidget(disp)
        bl.addWidget(gd)
        col = controls.Collapsible("세부 설정", body)
        col.btn.setToolTip("배경 제거 · 회전·자르기 · 검출 값 · 보기")
        lay.addWidget(col)
        lay.addStretch(1)
        self.fill_ladders()
        return page

    def _row_title(self, title, desc):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.addWidget(QtWidgets.QLabel(title))
        d = QtWidgets.QLabel(desc)
        d.setWordWrap(True)
        theme.hint(d)
        v.addWidget(d)
        w.setMinimumHeight(52)  # title + two lines of description; a form row would squeeze them together
        return w

    def refresh_lines_card(self):
        a = self.analysis
        self.lines_title.setText(f"레인 선 {len(a.lane_lines)}개 · 단백질 선 {len(a.protein_lines)}개")
        box = self.pline_box
        while box.count():
            it = box.takeAt(0)
            if it.layout():
                while it.layout().count():
                    w = it.layout().takeAt(0).widget()
                    if w:
                        w.hide()
                        w.deleteLater()
            elif it.widget():
                it.widget().hide()
                it.widget().deleteLater()
        if not a.protein_lines:
            box.addWidget(controls.hint("단백질 선 도구로 가로로 끌어 빨간 선을 그리세요"))
        names = [r.name for r in a.rows]
        for i, p in enumerate(a.protein_lines):
            hb = QtWidgets.QHBoxLayout()
            hb.addWidget(controls.color_dot("#d70015", 12))
            cb = QtWidgets.QComboBox()
            cb.addItems(names + (["(새 단백질)"] if p.get("row") not in names else []))
            cb.setCurrentText(p.get("row") if p.get("row") in names else "(새 단백질)")
            cb.setToolTip("이 선이 잡을 단백질 행")
            cb.currentTextChanged.connect(lambda t, i=i: self.set_pline_row(i, t))
            hb.addWidget(cb, 1)
            y = (p["y0"] + p["y1"]) / 2.0
            lab = QtWidgets.QLabel(f"y {y:.0f}")
            theme.hint(lab)
            hb.addWidget(lab)
            box.addLayout(hb)

    # ---- inspector: 레인 / 단백질
    def _build_lanes_inspector(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        g = QtWidgets.QGroupBox("레인")
        self.lane_grid = QtWidgets.QGridLayout(g)
        lay.addWidget(g)
        lay.addWidget(controls.hint("역할: 시료 · 마커(ladder) · 제외. 이름이 같은 레인은 반복으로 평균 ± SD."))
        g = QtWidgets.QGroupBox("단백질")
        self.prot_grid = QtWidgets.QGridLayout(g)
        lay.addWidget(g)
        hb = QtWidgets.QHBoxLayout()
        hb.addStretch(1)
        b = QtWidgets.QPushButton("＋ 새 단백질")
        b.setToolTip("이미지에서 새 단백질의 박스 하나를 드래그로 그립니다")
        b.clicked.connect(self.new_protein)
        hb.addWidget(b)
        b = controls.prominent(QtWidgets.QPushButton("ROI 박스 채우기"))
        b.setToolTip("고른 단백질 박스와 같은 크기의 박스를 모든 레인의 같은 높이에 채웁니다")
        b.clicked.connect(self.fill_roi_boxes)
        hb.addWidget(b)
        lay.addLayout(hb)
        self.fill_hint = controls.hint("")
        lay.addWidget(self.fill_hint)
        g, f = controls.card("선택한 박스")
        self.box_w = _spin(2, 5000, 20, suffix=" px")
        self.box_h = _spin(2, 5000, 10, suffix=" px")
        self.box_w.valueChanged.connect(lambda v: self._set_sel_size("width", v))
        self.box_h.valueChanged.connect(lambda v: self._set_sel_size("height", v))
        f.addRow("폭", self.box_w)
        f.addRow("높이", self.box_h)
        f.addRow(controls.hint("같은 단백질의 박스는 모두 같은 크기입니다. 박스를 더블클릭해 끌어도 됩니다."))
        self.sel_card = g
        g.setVisible(False)
        lay.addWidget(g)
        lay.addStretch(1)
        return page

    @staticmethod
    def _clear_grid(grid):
        while grid.count():
            it = grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def fill_lanes_table(self):
        a = self.analysis
        a.ensure_lane_labels()
        core.sync_lane_roles(a)
        g = self.lane_grid
        self._clear_grid(g)
        for c, t in enumerate(("#", "이름", "역할")):
            lab = QtWidgets.QLabel(t)
            theme.hint(lab)
            g.addWidget(lab, 0, c)
        for i in range(a.n_lanes):
            lab = QtWidgets.QLabel(f"L{i + 1}")
            theme.hint(lab)
            g.addWidget(lab, i + 1, 0)
            e = QtWidgets.QLineEdit(a.lane_labels[i])
            e.editingFinished.connect(lambda i=i, e=e: self.rename_lane(i, e.text()))
            g.addWidget(e, i + 1, 1)
            cb = QtWidgets.QComboBox()
            for r in core.LANE_ROLES:
                cb.addItem(core.LANE_ROLE_LABELS[r], r)
            cb.setCurrentIndex(max(0, cb.findData(a.lane_roles[i] if i < len(a.lane_roles) else core.LANE_SAMPLE)))
            cb.currentIndexChanged.connect(lambda _=None, i=i, cb=cb: self.set_lane_role(i, cb.currentData()))
            g.addWidget(cb, i + 1, 2)
        g.setColumnStretch(1, 1)
        self.fill_groupings()
        self.fill_basis_combo()
        self.refresh_steps()
        self.refresh_lines_card()
        self.view.rebuild_lines()

    def fill_rows_table(self):
        a = self.analysis
        g = self.prot_grid
        self._clear_grid(g)
        for i, r in enumerate(a.rows):
            g.addWidget(controls.color_dot(row_color(i).name(), 10), i, 0)
            e = QtWidgets.QLineEdit(r.name)
            e.editingFinished.connect(lambda i=i, e=e: self.rename_protein(i, e.text()))
            g.addWidget(e, i, 1)
            cb = QtWidgets.QComboBox()
            roles = [core.ROLE_CONTROL, core.ROLE_TARGET] + ([core.ROLE_IGNORE] if r.role == core.ROLE_IGNORE else [])
            for role in roles:
                cb.addItem(ROLE_SHORT[role], role)
            cb.setCurrentIndex(max(0, cb.findData(r.role)))
            cb.currentIndexChanged.connect(lambda _=None, i=i, cb=cb: self.set_row_role(i, cb.currentData()))
            g.addWidget(cb, i, 2)
        g.setColumnStretch(1, 1)
        self.update_sel_card()

    def rename_lane(self, i, text):
        a = self.analysis
        text = text.strip()
        if not text or i >= len(a.lane_labels) or a.lane_labels[i] == text:
            return
        a.lane_labels[i] = text
        if text not in a.named_lanes:
            a.named_lanes.append(text)
        self.view.refresh_items()
        self.view.rebuild_lines()
        self.remeasure()
        self._figure_changed(self.doc)

    def rename_protein(self, i, text):
        a = self.analysis
        text = text.strip()
        if not text or i >= len(a.rows) or a.rows[i].name == text:
            return
        old = a.rows[i].name
        new = core.unique_name(text, [r.name for k, r in enumerate(a.rows) if k != i])
        a.rows[i].name = new
        for p in a.protein_lines:
            if p.get("row") == old:
                p["row"] = new
        self.structure_changed()

    def set_lane_role(self, i, role):
        a = self.analysis
        core.sync_lane_roles(a)
        a.lane_roles[i] = role
        if self.doc.mode == core.MODE_GEL:
            core.set_ladder_lanes(a, [k for k, r in enumerate(a.lane_roles) if r == core.LANE_MARKER])
            core.sync_lane_roles(a)
        self.structure_changed()

    def update_sel_card(self):
        sel = self.view.selected_bands()
        self.sel_card.setVisible(bool(sel))
        if sel:
            r = self.analysis.rows[sel[0].row]
            for sp, v in ((self.box_w, r.width), (self.box_h, r.height)):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
            self.sel_card.setTitle(f"선택한 박스 · {r.name}")
            self.fill_hint.setText(f"'{r.name}' 박스와 같은 크기로 비어 있는 레인의 같은 높이에 박스를 채웁니다.")
        else:
            self.fill_hint.setText("박스를 하나 고르거나 '＋ 새 단백질'로 그린 뒤 'ROI 박스 채우기'를 누르세요.")

    def _set_sel_size(self, attr, v):
        rows = {b.row for b in self.view.selected_bands()}
        for ri in rows:
            setattr(self.analysis.rows[ri], attr, int(v))
        if rows:
            self.view.refresh_items()
            self.remeasure()

    def new_protein(self):
        if self.doc.raw is None:
            return
        self.set_step(1)
        self.set_draw_mode("box")
        self.statusBar().showMessage("새 단백질: 이미지에서 박스 하나를 드래그로 그리세요 · Esc 취소")

    def add_protein_box(self, x0, y0, x1, y1):
        a = self.analysis
        w, h = max(4, int(abs(x1 - x0))), max(3, int(abs(y1 - y0)))
        name = core.unique_name(f"P{len(a.rows) + 1}", [r.name for r in a.rows])
        a.rows.append(core.Row(name=name, width=w, height=h, role=core.ROLE_TARGET))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        lane = 0
        if a.lane_lines:
            lane = int(np.argmin([abs(core.poly_x(ln, cy) - cx) for ln in a.lane_lines]))
        b = core.Band(row=len(a.rows) - 1, lane=lane, cx=cx, cy=cy, user=True)
        a.bands.append(b)
        a.bands.sort(key=lambda b_: (b_.row, b_.lane))
        self.structure_changed()
        self.view.select_bands([b])
        self.update_sel_card()
        self.statusBar().showMessage(f"새 단백질 '{name}'을 만들었습니다 — 'ROI 박스 채우기'로 모든 레인에 채우세요")

    def fill_roi_boxes(self):
        sel = self.view.selected_bands()
        a = self.analysis
        if not sel:
            self.statusBar().showMessage("채울 단백질의 박스를 하나 고르세요")
            return
        ri = sel[0].row
        n = core.fill_row(a, ri)
        self.structure_changed()
        self.view.select_bands([b for b in a.bands if b.row == ri])
        self.statusBar().showMessage(f"'{a.rows[ri].name}' 박스 {n}개를 채웠습니다")

    # ---- lines
    def on_lines_changed(self, msg):
        self.refresh_lines_card()
        self.statusBar().showMessage(msg)

    def find_lane_lines(self):
        corr = self.get_corrected()
        if corr is None:
            return
        settings = core.DetectSettings(sensitivity=self.sens.value(), lut_gamma=self.lut_gamma.value())
        work = core.detection_image(corr, settings)
        H = work.shape[0]
        a = lanes.detect(corr, settings, self.doc.mode)
        self.analysis.lane_lines = [[[float(lanes.lane_line_x(g, 0.0)), 0.0], [float(lanes.lane_line_x(g, H - 1.0)), H - 1.0]]
                                    for g in a.lane_geom]
        self.view.rebuild_lines()
        self.refresh_lines_card()
        self.statusBar().showMessage(f"레인 선 {len(self.analysis.lane_lines)}개를 찾았습니다 — 고친 뒤 '선 기준으로 다시 검출'")

    def show_lane_menu(self, gpos, x, y):
        v = self.view
        sel = v.sel or {}
        a = self.analysis
        m = QtWidgets.QMenu(self)
        i = sel.get("i")
        if i is not None:
            m.addAction("여기에 점 추가\t더블클릭", lambda: v.add_lane_point(i, y))
            if sel.get("j") is not None:
                m.addAction("이 점 지우기\t⌫", lambda: v.delete_selection())
            m.addSeparator()
        H = v._img_h()
        pitch = float(np.median(np.diff(sorted(np.mean([q[0] for q in ln]) for ln in a.lane_lines)))) if len(a.lane_lines) > 1 else 60.0

        def add(dx):
            base = a.lane_lines[i] if i is not None else [[x, 0.0], [x, H - 1]]
            a.lane_lines.append([[q[0] + dx, q[1]] for q in base])
            a.lane_lines.sort(key=lambda ln: float(np.mean([q[0] for q in ln])))
            v.rebuild_lines()
            self.on_lines_changed("레인 선을 넣었습니다")
        m.addAction("왼쪽에 레인 선 추가", lambda: add(-pitch))
        m.addAction("오른쪽에 레인 선 추가", lambda: add(pitch))
        if i is not None:
            m.addSeparator()
            lab = a.lane_labels[i] if i < len(a.lane_labels) else f"L{i + 1}"
            act = m.addAction(f"레인 선 {lab} 지우기\t⌘⌫", lambda: v.delete_selection(whole=True))
            act.setProperty("hig_destructive", True)
        m.exec(gpos)

    def show_protein_menu(self, gpos, i):
        if i < 0:
            return
        m = QtWidgets.QMenu(self)
        sub = m.addMenu("잡을 단백질")
        for r in self.analysis.rows:
            sub.addAction(r.name, lambda n=r.name: self.set_pline_row(i, n))
        m.addSeparator()
        m.addAction("이 단백질 선 지우기\t⌫", lambda: self.view.delete_selection())
        m.exec(gpos)

    def add_protein_line(self, x0, y0, x1, y1):
        a = self.analysis
        used = {p.get("row") for p in a.protein_lines}
        row = next((r.name for r in a.rows if r.name not in used), None)
        if row is None:
            row = core.unique_name(f"P{len(a.rows) + 1}", [r.name for r in a.rows] + list(used))
        a.protein_lines.append({"row": row, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        a.protein_lines.sort(key=lambda p: (p["y0"] + p["y1"]) / 2.0)
        self.view.rebuild_lines()
        self.refresh_lines_card()
        self.statusBar().showMessage(f"단백질 선 '{row}'을 그렸습니다 — 양끝 네모로 기울이고, '선 기준으로 다시 검출'로 반영합니다")

    def set_pline_row(self, i, name):
        a = self.analysis
        if 0 <= i < len(a.protein_lines) and name and name != "(새 단백질)":
            a.protein_lines[i]["row"] = name
            self.view.rebuild_lines()
            self.refresh_lines_card()

    # ---- marker: strip, size marks, ladder
    def _marker_s(self):
        d = self.doc
        if d.figure is None:
            d.figure = figure.default_settings()
        return d.figure

    def fill_ladders(self):
        cb = self.ladder_combo
        cb.blockSignals(True)
        cb.clear()
        gel = self.doc.mode == core.MODE_GEL if hasattr(self, "docs") else False
        presets = core.LADDER_PRESETS if gel else core.LADDER_PRESETS_KDA
        for name in presets:
            cb.addItem(name, name)
        cur = (self.doc.figure or {}).get("ladder") if hasattr(self, "docs") else None
        cb.setCurrentIndex(max(0, cb.findData(cur)))
        cb.blockSignals(False)
        self.kda_unit.setText("bp" if gel else "kDa")

    def on_marker_edited(self, msg):
        s = self._marker_s()
        s["show_kda"] = bool(s.get("kda"))
        self.refresh_marker_card()
        self._figure_changed(self.doc)
        self.statusBar().showMessage(msg)

    def on_tick_selected(self, i):
        s = self._marker_s()
        k = s.get("kda", [])
        self.first_kda.blockSignals(True)
        self.first_kda.setValue(float(k[i]["kda"]) if 0 <= i < len(k) and k[i].get("kda") else 0)
        self.first_kda.blockSignals(False)
        self.refresh_marker_card()

    def _anchor(self):
        sel = self.view.sel or {}
        k = self._marker_s().get("kda", [])
        if sel.get("kind") == "tick" and 0 <= sel.get("i", -1) < len(k):
            return sel["i"]
        return 0 if k else None

    def apply_ladder(self):
        """The size of one marker band (첫 마커 크기) fills the others in the ladder's order."""
        if not hasattr(self, "docs"):
            return
        s = self._marker_s()
        name = self.ladder_combo.currentData()
        s["ladder"] = name
        marks = s.get("kda", [])
        i = self._anchor()
        v = self.first_kda.value()
        if i is None or not v:
            self.refresh_marker_card()
            return
        presets = core.LADDER_PRESETS if self.doc.mode == core.MODE_GEL else core.LADDER_PRESETS_KDA
        sizes = core.fill_ladder(len(marks), i, v, presets.get(name, core.DEFAULT_LADDER))
        for m_, sz in zip(marks, sizes):
            m_["kda"] = sz
        s["unit"] = "bp" if self.doc.mode == core.MODE_GEL else "kDa"
        self.on_marker_edited("마커 크기를 래더 순서로 채웠습니다 — 칩을 눌러 하나씩 고칠 수 있습니다")
        self.view.rebuild_lines()

    def edit_kda_chip(self, i):
        s = self._marker_s()
        k = s.get("kda", [])
        if not 0 <= i < len(k):
            return
        self.view.sel = {"kind": "tick", "i": i}
        self.view.rebuild_lines()
        v, ok = QtWidgets.QInputDialog.getDouble(self, "마커 크기", f"{i + 1}번째 밴드 크기 ({self.kda_unit.text()}):",
                                                 float(k[i].get("kda") or 0), 0, 1e6, 1)
        if ok:
            k[i]["kda"] = float(v) if v else None
            self.on_marker_edited("마커 크기를 고쳤습니다")
            self.view.rebuild_lines()

    def refresh_marker_card(self):
        d = self.doc
        m = (d.figure or {}).get("marker_path") if d.marker is not None else None
        self.marker_name.setText(os.path.basename(m) if m else "없음")
        k = (d.figure or {}).get("kda", []) if d.figure else []
        sel = self.view.sel or {}
        cur = sel.get("i", -1) if sel.get("kind") == "tick" else -1
        self.kda_chips.set_chips([("?" if t.get("kda") is None else f"{t['kda']:g}") for t in k], cur)

    # ---------------------------------------------------------------- images (documents)
    @property
    def doc(self) -> ImageDoc:
        return self.docs[self.cur]

    def _n_projects(self) -> int:
        return len({d.project for d in self.docs if d.raw is not None})

    def doc_label(self, k: int) -> str:
        d = self.docs[k]
        return f"{self.doc_key(k)}: {d.name}" + (f" [{d.project}]" if d.project and self._n_projects() > 1 else "")

    def doc_key(self, k: int) -> str:
        return f"I{k + 1}"

    def refresh_doc_tabs(self):
        """The image list of the sidebar (I1, I2 … with their project)."""
        rows = [(f"{self.doc_key(k)} · {d.name}", d.project if d.project and self._n_projects() > 1 else "")
                for k, d in enumerate(self.docs) if d.raw is not None]
        self.img_list.set_rows(rows or [("(이미지 없음)", "")], self.cur)

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
            self.update_title()
            self.refresh_marker_view()
            self.sync_orient_widgets()
            self.update_display(reset_zoom=True)
        else:
            self.view.set_marker(None)
            self.marker_label.setText("없음")
            self.file_label.setText("-")
            self.setWindowTitle("Blot Profiler")
            if self.view.pix_item is not None:
                self.view.set_image(QtGui.QImage(), reset_zoom=False)
        self.set_analysis(d.analysis)
        if self.img_list.currentRow() != k:
            self.img_list.blockSignals(True)
            self.img_list.setCurrentRow(k)
            self.img_list.blockSignals(False)
        self.fill_ladders()
        self.refresh_marker_card()
        self.img_info.setText(f"{d.raw.shape[1]} × {d.raw.shape[0]} · 측정은 원본(raw) 값" if d.raw is not None else "")

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

    def showEvent(self, e):
        super().showEvent(e)
        QtCore.QTimer.singleShot(0, self.view.fit)

    def edit_band_size(self, item):
        """Double-click on a box: select it and drag its edges to resize (its whole row follows)."""
        if not item.isSelected():
            self.view.select_bands([item.band])
        self.resize_mode.setChecked(True)

    def export_figure(self):
        if self._figure_ready():
            self.figure_editor.export()

    def merge_selected(self):
        sel = self.view.selected_bands()
        if len(sel) < 2:
            return
        n = core.merge_bands(self.analysis, sel)
        if self.auto_lanes.isChecked():
            core.auto_assign_lanes(self.analysis)
        self.structure_changed()
        self.statusBar().showMessage(f"선택한 박스 {len(sel)}개를 {n}개로 합쳤습니다 (같은 행의 박스 크기가 함께 맞춰집니다)")

    def on_resize_mode(self, on):
        self.view.set_resize_mode(on)
        self.view.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.statusBar().showMessage(
            "크기 수정: 박스 가장자리/모서리를 드래그하세요 (같은 단백질 행이 함께 바뀜). Esc 또는 빈 곳 클릭으로 끝"
            if on else "크기 수정을 끝냈습니다"
        )

    # ---------------------------------------------------------------- menus
    def _build_menu(self):
        """파일 메뉴 (spec §11): every save is here too."""
        m = self.menuBar().addMenu("파일")
        m.addAction("새 프로젝트", self.new_project).setShortcut(QtGui.QKeySequence.New)
        m.addAction("열기… (이미지 · 프로젝트, 여러 개 고르면 함께)", self.open_any).setShortcut(QtGui.QKeySequence.Open)
        m.addAction("프로젝트 추가로 불러오기…", self.add_project_dialog)
        m.addSeparator()
        m.addAction("저장", lambda: self.save_project()).setShortcut(QtGui.QKeySequence.Save)
        m.addAction("다른 이름으로 저장…", lambda: self.save_project(save_as=True)).setShortcut("Ctrl+Shift+S")
        m.addSeparator()
        m.addAction("ROI·그룹 저장…", self.save_roi)
        m.addAction("ROI·그룹 불러오기…", self.load_roi)
        m.addSeparator()
        m.addAction("내보내기…", self.open_export_dialog).setShortcut("Ctrl+E")
        em = m.addMenu("바로 내보내기")
        em.addAction("정량 그래프의 데이터 (CSV)…", self.export_graph_csv)
        em.addAction("정량 그래프 이미지…", lambda: self.export_plot())
        em.addAction("Figure 이미지…", self.export_figure)
        em.addAction("Figure에 쓰인 데이터 (CSV)…", self.export_figure_numbers)
        em.addAction("모든 밴드 값 (CSV)…", self.export_csv)
        em.addAction("박스 표시한 블롯 이미지…", self.export_overlay)
        m.addSeparator()
        act = m.addAction("설정…", self.open_settings)
        act.setShortcut(QtGui.QKeySequence.Preferences)
        act.setMenuRole(QtGui.QAction.NoRole)  # stays in the File menu on macOS too
        qm = self.menuBar().addMenu("정량·그래프")
        qm.addAction("정량·그래프 창 열기", self.open_quant_window).setShortcut("Ctrl+G")
        qm.addSeparator()
        for key, text in (("bands", "밴드 값"), ("quant", "정량"), ("multi", "여러 이미지")):
            qm.addAction(text, lambda k=key: self.open_quant_window(k))
        qm.addSeparator()
        qm.addAction("측정 다시 하기", lambda: self.quant_win.remeasure())
        qm.addAction("이 그래프의 데이터 (CSV)…", self.export_graph_csv)
        qm.addAction("그래프 이미지…", lambda: self.export_plot())
        fg = self.menuBar().addMenu("Figure")
        fg.addAction("Figure 창 열기", self.open_figure_window).setShortcut("Ctrl+Shift+F")
        fg.addSeparator()
        fg.addAction("레인별 자르기", lambda: (self.open_figure_window(), self.fig_win.layout_seg.set_value("rows", emit=True)))
        fg.addAction("Full blot", lambda: (self.open_figure_window(), self.fig_win.layout_seg.set_value("full", emit=True)))
        fg.addSeparator()
        fg.addAction("Figure 이미지…", self.export_figure)
        fg.addAction("Figure에 쓰인 데이터 (CSV)…", self.export_figure_numbers)
        v = self.menuBar().addMenu("보기")
        v.addAction("창에 맞추기", self.view.fit).setShortcut("Ctrl+0")
        v.addAction("확대", lambda: self.view.zoom(1.25)).setShortcut(QtGui.QKeySequence.ZoomIn)
        v.addAction("축소", lambda: self.view.zoom(0.8)).setShortcut(QtGui.QKeySequence.ZoomOut)
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
        hint = QtWidgets.QLabel("왼쪽 화면의 마커 위 파란 상자 = 그림에 쓰는 마커 띠. 끌어서 옮기고 가장자리로 폭을 바꿉니다.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_tok('blue')}; font-size: 11px;")
        f.addRow(hint)
        hb = QtWidgets.QHBoxLayout()
        self.marker_orient = QtWidgets.QComboBox()
        for k, flip, label in core.MARKER_ORIENTATIONS:
            self.marker_orient.addItem(label, f"{k},{int(flip)}")
        self.marker_orient.setToolTip("마커 사진을 돌리거나 뒤집어 블롯과 같은 방향으로 맞춥니다")
        self.marker_orient.currentIndexChanged.connect(self.on_marker_orient)
        hb.addWidget(self.marker_orient, 1)
        self.auto_orient_marker = QtWidgets.QCheckBox("가로·세로가 바뀐 마커는 자동으로 90° 돌리기")
        self.auto_orient_marker.setChecked(True)
        hb.addWidget(self.auto_orient_marker)
        f.addRow("마커 방향", hb)
        self.ask_marker = QtWidgets.QCheckBox("블롯 TIF를 열면 짝이 되는 마커 이미지도 받기")
        self.ask_marker.setToolTip("같은 폴더에 '…Marker…' 파일이 있으면 자동으로 붙이고, 없으면 마커 파일을 고르는 창을 띄웁니다 (취소 = 건너뛰기)")
        self.ask_marker.setChecked(True)
        f.addRow(self.ask_marker)
        lay.addWidget(g_img)
        lay.addWidget(self._build_orient())

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
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem("레인 먼저 찾기 (새 방식: 레인 → 레인 안 밴드 → 줄 맞추기)", "lane")
        self.method_combo.addItem("이전 검출 방식", "legacy")
        self.method_combo.setToolTip("새 방식은 레인을 먼저 찾고(기울어진 레인 ±10°, 좌우 ±10% 어긋남 허용), 레인마다 밴드를 찾은 뒤\n"
                                     "같은 높이의 밴드를 줄로 묶습니다.")
        f.addRow("검출 방식", self.method_combo)
        self.auto_tilt = QtWidgets.QCheckBox("검출 후 기울기 자동 보정 (이미지 전체 회전 · 마커도 같이)")
        self.auto_tilt.setChecked(True)
        self.auto_tilt.setToolTip("밴드 줄의 기울기를 재서 0.3° 이상이면 이미지 전체를 그만큼 돌리고 다시 검출합니다.\n"
                                  "1번 탭 '미세' 회전에서 확인·되돌릴 수 있습니다")
        f.addRow(self.auto_tilt)
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

        lay.addStretch(1)
        self.on_bg_changed(recompute=False)
        return w

    # ---------------------------------------------------------------- tab 2
    def _build_groupings(self):
        g = QtWidgets.QGroupBox("측정 묶음 (Control · Target 짝짓기, 레인은 그대로)")
        l = QtWidgets.QVBoxLayout(g)
        info = QtWidgets.QLabel(
            "떨어져 있는 박스도 그룹으로 묶어 Control 과 짝지을 수 있습니다. 묶음 세트를 여러 개 만들어 두고(예: '샘플 짝', '시간별'), "
            "4번 정량 탭의 'Control 짝짓기 기준'에서 어떤 세트로 계산할지 고릅니다. 한 박스는 여러 세트에 동시에 들어갈 수 있습니다.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {_tok('label-secondary')};")
        l.addWidget(info)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(QtWidgets.QLabel("세트"))
        self.set_list = QtWidgets.QComboBox()
        self.set_list.setToolTip("고른 세트의 그룹이 이미지에 색깔 테두리·이름으로 표시됩니다")
        self.set_list.currentIndexChanged.connect(self.on_set_chosen)
        hb.addWidget(self.set_list, 1)
        for text, fn in (("새 세트…", self.new_set), ("이름 바꾸기…", self.rename_set), ("세트 삭제", self.delete_set)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        l.addLayout(hb)
        self.group_table = QtWidgets.QTableWidget(0, 3)
        self.group_table.setHorizontalHeaderLabels(["그룹", "박스", "상태"])
        self.group_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.group_table.setMinimumHeight(140)
        self.group_table.itemSelectionChanged.connect(self.on_group_row_selected)
        self.group_table.itemChanged.connect(self.on_group_renamed)
        l.addWidget(self.group_table)
        hb = QtWidgets.QHBoxLayout()
        for text, fn in (("선택한 박스를 묶음에 넣기…", self.assign_group), ("▲", lambda: self.move_group(-1)),
                         ("▼", lambda: self.move_group(1)), ("선택한 그룹 지우기", self.delete_groups)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        hb.addStretch(1)
        l.addLayout(hb)
        self.group_hint = QtWidgets.QLabel()
        self.group_hint.setWordWrap(True)
        self.group_hint.setStyleSheet(f"color: {_tok('label-secondary')}; font-size: 11px;")
        l.addWidget(self.group_hint)
        return g

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
        msg.setStyleSheet(f"color: {_tok('label-secondary')};")
        msg.setWordWrap(True)
        gl.addWidget(msg)
        return g, t, hb, msg

    # ---------------------------------------------------------------- tab 4
    # ---------------------------------------------------------------- 정량·그래프 (spec §6)
    def _build_results_tab(self):
        """정량 page of the quantification window: sidebar (images, control pairing, graph items,
        formulas) | graph with its editor (top 70 %) and the data table (bottom 30 %)."""
        side = QtWidgets.QWidget()
        side.setObjectName("hig_sidebar")
        side.setAttribute(Qt.WA_StyledBackground, True)
        lay = QtWidgets.QVBoxLayout(side)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        lay.addWidget(controls.section("이미지"))
        self.quant_img_box = QtWidgets.QVBoxLayout()
        lay.addLayout(self.quant_img_box)
        lay.addWidget(controls.section("Control 짝짓기"))
        f = QtWidgets.QFormLayout()
        self.auto_norm = QtWidgets.QCheckBox()  # Target ÷ Control (always on in 1.0; formulas cover the rest)
        self.auto_norm.setChecked(True)
        self.basis_combo = QtWidgets.QComboBox()
        self.basis_combo.setToolTip("어떤 Control 박스로 어떤 Target 박스를 나눌지: 같은 레인끼리(기본) 또는 측정 묶음 세트의 그룹끼리")
        self.basis_combo.currentIndexChanged.connect(self.on_basis)
        f.addRow("기준", self.basis_combo)
        self.ctrl_combo = QtWidgets.QComboBox()
        self.ctrl_combo.currentIndexChanged.connect(self.on_ctrl)
        f.addRow("Control", self.ctrl_combo)
        self.ref_lane = QtWidgets.QComboBox()
        self.ref_lane.currentIndexChanged.connect(self.on_ref_lane)
        self.ref_label = QtWidgets.QLabel("1.0 기준")
        f.addRow(self.ref_label, self.ref_lane)
        lay.addLayout(f)
        self.basis_msg = QtWidgets.QLabel()
        self.basis_msg.setWordWrap(True)
        lay.addWidget(self.basis_msg)
        self.sets_btn = QtWidgets.QPushButton("측정 묶음 세트 · 편집…")
        self.sets_btn.setFlat(True)
        self.sets_btn.setStyleSheet(f"text-align: left; color: {_tok('blue')}; background: transparent;")
        self.sets_btn.clicked.connect(self.edit_groupings)
        lay.addWidget(self.sets_btn)
        lay.addWidget(controls.section("그래프 항목"))
        self.series_box = QtWidgets.QVBoxLayout()
        self.series_box.setSpacing(2)
        self.series_on: Dict[str, bool] = {}
        lay.addLayout(self.series_box)
        self.add_formula_btn = QtWidgets.QPushButton("＋ 수식 추가")
        self.add_formula_btn.setFlat(True)
        self.add_formula_btn.setStyleSheet(f"text-align: left; color: {_tok('blue')}; background: transparent; font-weight: 600;")
        self.add_formula_btn.clicked.connect(lambda: self.open_formula_card())
        lay.addWidget(self.add_formula_btn)
        self.formula_card = QtWidgets.QFrame()
        self.formula_card.setObjectName("formulaCard")
        self.formula_card.setStyleSheet("#formulaCard { background: rgba(0,102,204,0.06); border: 1px solid rgba(0,102,204,0.35); border-radius: 11px; }")
        cl = QtWidgets.QVBoxLayout(self.formula_card)
        self.formula_card_title = QtWidgets.QLabel()
        self.formula_card_title.setStyleSheet("font-weight: 600;")
        cl.addWidget(self.formula_card_title)
        self.formula_editor_box = QtWidgets.QVBoxLayout()
        cl.addLayout(self.formula_editor_box)
        hb = QtWidgets.QHBoxLayout()
        hb.addStretch(1)
        b = QtWidgets.QPushButton("취소")
        b.clicked.connect(self.close_formula_card)
        hb.addWidget(b)
        self.formula_ok = controls.prominent(QtWidgets.QPushButton("그래프에 추가"))
        self.formula_ok.clicked.connect(self.accept_formula_card)
        hb.addWidget(self.formula_ok)
        cl.addLayout(hb)
        self.formula_card.setVisible(False)
        lay.addWidget(self.formula_card)
        self.formula_editor: Optional[FormulaEditor] = None
        self.formula_row: Optional[tuple] = None
        self.formula_msg = controls.hint(FORMULA_HINT)
        lay.addWidget(self.formula_msg)
        lay.addStretch(1)
        lay.addWidget(controls.hint("저장: 창 위 '내보내기' 또는 파일 › 내보내기… (⌘E)"))
        side_scroll = self._scroll(side)
        side_scroll.setMinimumWidth(330)  # narrow sidebar on the window floor, wide enough that no row is cut off
        side_scroll.setMaximumWidth(340)

        # graph + editor
        gp = self.prefs["graph"]
        self.graph_opts = {k: gp[k] for k in ("kind", "err", "colors", "y0")}
        self.graph_opts.update({"points": True, "split_images": False, "by_condition": True, "bar_width": 0.8,
                                "size_mm": [90, 60], "x_rot": "auto", "series": {}, "labels": {}, "texts": [],
                                "legend": True, "legend_pos": None, "y_title": "", "title": "", "y_min": None, "y_max": None})
        self.plot = PlotCanvas()
        self.plot.toolbar.setVisible(False)
        self.plot.style_fn = self.graph_style
        self.graph_title = QtWidgets.QLabel("정량 그래프")
        self.graph_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.graph_sub = controls.hint("")
        gw = QtWidgets.QWidget()
        gl = QtWidgets.QVBoxLayout(gw)
        gl.setContentsMargins(16, 12, 12, 8)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.graph_title)
        hb.addWidget(self.graph_sub, 1)
        self.graph_head = hb  # the graph's zoom chip goes at its end (quant window)
        gl.addLayout(hb)
        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setWidget(self.plot)
        self.plot_scroll.setStyleSheet("QScrollArea { background: white; border-radius: 10px; }")
        self.plot_scroll.setMinimumSize(460, 340)  # smaller and the axis titles would not fit
        gl.addWidget(self.plot_scroll, 1)
        self.graph_editor = self._build_graph_editor()
        top = gw

        # data table
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(16, 6, 16, 8)
        hb = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("데이터")
        t.setStyleSheet("font-weight: 700;")
        hb.addWidget(t)
        self.table_seg = controls.Segmented([("drawn", "그린 값"), ("bands", "밴드별"), ("summary", "요약")], small=True)
        self.table_seg.changed.connect(lambda _: self.fill_results_table())
        hb.addWidget(self.table_seg)
        hb.addStretch(1)
        hb.addWidget(controls.hint("행을 고르면 메인 창에서 그 박스가 선택됩니다"))
        bl.addLayout(hb)
        self.results_table = QtWidgets.QTableWidget(0, 0)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.itemSelectionChanged.connect(self.on_table_pick)
        bl.addWidget(self.results_table, 1)
        right = QtWidgets.QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(bottom)
        right.setStretchFactor(0, 7)
        right.setStretchFactor(1, 3)
        canvas = QtWidgets.QWidget()
        canvas.setObjectName("hig_content")
        canvas.setAttribute(Qt.WA_StyledBackground, True)
        QtWidgets.QVBoxLayout(canvas).addWidget(right)
        canvas.layout().setContentsMargins(4, 4, 4, 4)
        from .ui.hig_widgets import PaneSplitter

        root = PaneSplitter()  # sidebar on the base | graph + table box | editor box
        root.setChildrenCollapsible(False)
        root.addWidget(side_scroll)
        root.addWidget(canvas)
        root.addWidget(self.graph_editor)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setStretchFactor(2, 0)
        root.setSizes([330, 100000, theme.INSPECTOR_W])  # the graph box takes the rest: no strip right of the editor
        self.results_title = QtWidgets.QLabel()  # (kept for older code paths)
        self.by_condition = QtWidgets.QCheckBox()
        self.by_condition.setChecked(True)
        self.show_points = QtWidgets.QCheckBox()
        self.show_points.setChecked(True)
        return root

    def _build_graph_editor(self):
        """Graph editor (380 px): chip tabs 그래프 / 데이터 시리즈 / 축 / 글꼴 / 범례·글자."""
        from .ui.hig_widgets import ChipTabs

        tabs = ChipTabs()
        o = self.graph_opts
        gp = self.prefs["graph"]

        def page():
            w = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(w)
            return w, lay

        def combo(items, key):
            cb = QtWidgets.QComboBox()
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(8)  # long choices elide instead of widening the inspector
            for k, t in items:
                cb.addItem(t, k)
            cb.setCurrentIndex(max(0, cb.findData(o.get(key))))
            cb.currentIndexChanged.connect(lambda _=None: self.set_graph_opt(key, cb.currentData()))
            self._graph_widgets[key] = cb
            return cb

        def sw(key):
            c = controls.switch("", bool(o.get(key)))
            c.toggled.connect(lambda on: self.set_graph_opt(key, on))
            self._graph_widgets[key] = c
            return c

        self._graph_widgets = {}
        w, lay = page()
        g, f = controls.card()
        st = QtWidgets.QComboBox()
        st.addItem("PRISM 기본")
        f.addRow("스타일", st)
        f.addRow("종류", combo(prefs.GRAPH_KINDS, "kind"))
        f.addRow("X 묶음", combo([(True, "레인 이름 (반복은 평균)"), (False, "각 레인")], "by_condition"))
        f.addRow("색", combo(prefs.COLOR_SCHEMES, "colors"))
        f.addRow("오차 막대", combo(prefs.ERRORS, "err"))
        lay.addWidget(g)
        g, f = controls.card()
        f.addRow("각 레인 값 점", sw("points"))
        f.addRow("이미지마다 따로 그리기", sw("split_images"))
        f.addRow("Y축 0부터", sw("y0"))
        lay.addWidget(g)
        g, f = controls.card()
        hb = QtWidgets.QHBoxLayout()
        self.size_w = _spin(20, 400, o["size_mm"][0], 5, suffix=" mm")
        self.size_h = _spin(20, 400, o["size_mm"][1], 5, suffix=" mm")
        for sp in (self.size_w, self.size_h):
            sp.valueChanged.connect(lambda _=None: self.set_graph_opt("size_mm", [self.size_w.value(), self.size_h.value()]))
            theme.plain_number(sp)  # [90] × [60] mm, two small fields (mockup)
            sp.setSuffix("")
            sp.setFixedWidth(64)
        hb.addStretch(1)
        hb.addWidget(self.size_w)
        hb.addWidget(QtWidgets.QLabel("×"))
        hb.addWidget(self.size_h)
        mm = QtWidgets.QLabel("mm")
        theme.hint(mm)
        hb.addWidget(mm)
        f.addRow("그림 크기", hb)
        bw = QtWidgets.QDoubleSpinBox()
        bw.setRange(0.2, 1.0)
        bw.setSingleStep(0.05)
        bw.setValue(float(o["bar_width"]))
        bw.valueChanged.connect(lambda v: self.set_graph_opt("bar_width", float(v)))
        f.addRow("막대 폭", controls.slider_row(bw))
        lay.addWidget(g)
        lay.addWidget(controls.hint("그래프의 범례·제목·글자 상자는 끌어서 옮기고, 더블클릭으로 편집합니다."))
        lay.addStretch(1)
        tabs.addTab(self._scroll(w), "그래프")

        w, lay = page()
        self.series_style_box = QtWidgets.QVBoxLayout()
        g = QtWidgets.QGroupBox("데이터 시리즈")
        g.setLayout(self.series_style_box)
        lay.addWidget(g)
        lay.addWidget(controls.hint("색 우물을 눌러 막대 색을, 모양으로 점 모양을, 이름으로 범례 글자를 바꿉니다."))
        lay.addStretch(1)
        tabs.addTab(self._scroll(w), "데이터 시리즈")

        w, lay = page()
        g, f = controls.card()
        yt = QtWidgets.QLineEdit(o.get("y_title") or "")
        yt.setPlaceholderText("자동 (Relative intensity)")
        yt.editingFinished.connect(lambda: self.set_graph_opt("y_title", yt.text().strip()))
        self._graph_widgets["y_title"] = yt
        f.addRow("Y축 이름", yt)
        for key, label in (("y_min", "Y 최소"), ("y_max", "Y 최대")):
            hb = QtWidgets.QHBoxLayout()
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-1e9, 1e9)
            sp.setDecimals(3)
            sp.setValue(float(o.get(key) or 0))
            au = controls.switch("", o.get(key) is None)
            sp.setEnabled(not au.isChecked())
            sp.valueChanged.connect(lambda v, key=key, au=au: (not au.isChecked()) and self.set_graph_opt(key, float(v)))
            au.toggled.connect(lambda on, key=key, sp=sp: (sp.setEnabled(not on), self.set_graph_opt(key, None if on else sp.value())))
            hb.addWidget(sp)
            lab = QtWidgets.QLabel("자동")
            theme.hint(lab)
            hb.addWidget(lab)
            hb.addWidget(au)
            f.addRow(label, hb)
        f.addRow("X 글자 기울기", combo([("auto", "자동"), (0, "0°"), (45, "45°"), (90, "90°")], "x_rot"))
        ts = controls.Segmented([("out", "바깥"), ("in", "안쪽")], small=True)
        ts.set_value(o.get("tick_dir") or gp.get("tick_dir", "out"))
        ts.changed.connect(lambda v: self.set_graph_opt("tick_dir", v))
        f.addRow("눈금 방향", ts)
        tl = QtWidgets.QDoubleSpinBox()
        tl.setRange(0, 20)
        tl.setValue(float(o.get("tick_len") or gp.get("tick_len", 6)))
        tl.setSuffix(" pt")
        tl.valueChanged.connect(lambda v: self.set_graph_opt("tick_len", float(v)))
        f.addRow("눈금 길이", tl)
        lay.addWidget(g)
        lay.addStretch(1)
        tabs.addTab(self._scroll(w), "축")

        w, lay = page()
        g, f = controls.card()
        fc = QtWidgets.QComboBox()
        fc.setEditable(True)
        fc.addItems(["Arial", "Helvetica", "Helvetica Neue", "Times New Roman"])
        fc.setCurrentText(o.get("font") or gp.get("font", "Arial"))
        fc.currentTextChanged.connect(lambda t: self.set_graph_opt("font", t))
        f.addRow("글꼴", fc)
        for key, label in (("label_size", "축 이름"), ("tick_size", "눈금 숫자"), ("legend_size", "범례"), ("title_size", "제목")):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(4, 40)
            sp.setSingleStep(0.5)
            sp.setSuffix(" pt")
            sp.setValue(float(o.get(key) or gp.get(key, 12)))
            sp.valueChanged.connect(lambda v, key=key: self.set_graph_opt(key, float(v)))
            f.addRow(label, sp)
        f.addRow("굵게", sw("bold") if "bold" in o else self._bold_switch())
        lay.addWidget(g)
        lay.addStretch(1)
        tabs.addTab(self._scroll(w), "글꼴")

        w, lay = page()
        g, f = controls.card()
        f.addRow("범례 보이기", sw("legend"))
        b = QtWidgets.QPushButton("범례 위치 처음대로")
        b.clicked.connect(lambda: self.set_graph_opt("legend_pos", None))
        f.addRow(b)
        tt = QtWidgets.QLineEdit(o.get("title") or "")
        tt.setPlaceholderText("제목 없음")
        tt.editingFinished.connect(lambda: self.set_graph_opt("title", tt.text().strip()))
        self._graph_widgets["title"] = tt
        f.addRow("제목", tt)
        lay.addWidget(g)
        g = QtWidgets.QGroupBox("글자 상자")
        self.textbox_box = QtWidgets.QVBoxLayout(g)
        lay.addWidget(g)
        b = QtWidgets.QPushButton("＋ 글자 상자")
        b.clicked.connect(self.add_text_box)
        lay.addWidget(b)
        lay.addStretch(1)
        tabs.addTab(self._scroll(w), "범례·글자")

        pane = QtWidgets.QWidget()
        pane.setObjectName("hig_inspector")
        pane.setAttribute(Qt.WA_StyledBackground, True)
        pl = QtWidgets.QVBoxLayout(pane)
        pl.setContentsMargins(12, 12, 4, 10)
        pl.addWidget(tabs.chips)
        pl.addWidget(tabs, 1)
        pane.setFixedWidth(theme.INSPECTOR_W)  # same inspector box as the main window
        self.graph_tabs = tabs
        for i in range(tabs.count()):
            theme.unify(tabs.widget(i).widget())
        return pane

    def _bold_switch(self):
        c = controls.switch("", bool(self.prefs["graph"].get("bold", True)))
        c.toggled.connect(lambda on: self.set_graph_opt("bold", on))
        return c

    def graph_style(self) -> dict:
        return graph.style(self.prefs["graph"], **{k: v for k, v in self.graph_opts.items() if v is not None
                                                   or k in ("y_min", "y_max", "legend_pos")})

    def set_graph_opt(self, key, value):
        self.graph_opts[key] = value
        if key in ("by_condition", "split_images"):
            self.fill_results_table()
        self.replot()
        if hasattr(self, "multi_plot"):
            self.replot_multi()

    def set_graph_opts(self, opts: dict):
        for k, v in (opts or {}).items():
            self.graph_opts[k] = v
        for key, w in getattr(self, "_graph_widgets", {}).items():
            if key not in self.graph_opts:
                continue
            v = self.graph_opts[key]
            w.blockSignals(True)
            if isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(max(0, w.findData(v)))
            elif isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QtWidgets.QLineEdit):
                w.setText(v or "")
            w.blockSignals(False)
        self.replot()
        if hasattr(self, "multi_plot"):
            self.replot_multi()

    def fill_series_styles(self, names):
        box = self.series_style_box
        while box.count():
            it = box.takeAt(0)
            if it.layout():
                while it.layout().count():
                    w = it.layout().takeAt(0).widget()
                    if w:
                        w.deleteLater()
            elif it.widget():
                it.widget().deleteLater()
        pal = graph.PALETTES.get(self.graph_opts.get("colors"), graph.PALETTES["bw"])
        for si, name in enumerate(names):
            own = self.graph_opts["series"].get(name, {})
            hb = QtWidgets.QHBoxLayout()
            col = own.get("color") or pal[si % len(pal)]
            b = QtWidgets.QPushButton()
            b.setFixedSize(22, 22)
            b.setStyleSheet(f"background: {col}; border: 1px solid #888; border-radius: 4px;")
            b.setToolTip("색 바꾸기")
            b.clicked.connect(lambda _=None, n=name: self.pick_series_color(n))
            hb.addWidget(b)
            e = QtWidgets.QLineEdit(self.graph_opts["labels"].get(name, name))
            e.editingFinished.connect(lambda n=name, e=e: self._set_series_label(n, e.text()))
            hb.addWidget(e, 1)
            mk = QtWidgets.QComboBox()
            for m, t in (("o", "●"), ("s", "■"), ("^", "▲"), ("D", "◆"), ("v", "▼")):
                mk.addItem(t, m)
            mk.setCurrentIndex(max(0, mk.findData(own.get("marker") or graph.MARKERS[si % len(graph.MARKERS)])))
            mk.currentIndexChanged.connect(lambda _=None, n=name, mk=mk: self._set_series_style(n, "marker", mk.currentData()))
            hb.addWidget(mk)
            box.addLayout(hb)

    def pick_series_color(self, name):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.graph_opts["series"].get(name, {}).get("color") or "#ffffff"), self)
        if c.isValid():
            self._set_series_style(name, "color", c.name())

    def _set_series_style(self, name, key, value):
        self.graph_opts["series"].setdefault(name, {})[key] = value
        self.replot()

    def _set_series_label(self, name, text):
        self.graph_opts["labels"][name] = text.strip() or name
        self.replot()

    def add_text_box(self):
        self.graph_opts["texts"].append({"text": "글자", "x": 0.1, "y": 0.92})
        self.fill_text_boxes()
        self.replot()

    def fill_text_boxes(self):
        box = self.textbox_box
        while box.count():
            it = box.takeAt(0)
            if it.layout():
                while it.layout().count():
                    w = it.layout().takeAt(0).widget()
                    if w:
                        w.deleteLater()
        for i, tb in enumerate(self.graph_opts["texts"]):
            hb = QtWidgets.QHBoxLayout()
            e = QtWidgets.QLineEdit(tb.get("text", ""))
            e.editingFinished.connect(lambda i=i, e=e: (self.graph_opts["texts"][i].__setitem__("text", e.text()), self.replot()))
            hb.addWidget(e, 1)
            b = QtWidgets.QToolButton()
            b.setText("✕")
            b.clicked.connect(lambda _=None, i=i: (self.graph_opts["texts"].pop(i), self.fill_text_boxes(), self.replot()))
            hb.addWidget(b)
            box.addLayout(hb)

    # ---- which images, which control, which reference
    def quant_docs(self) -> List["ImageDoc"]:
        return [d for d in self.loaded_docs() if d.measurements and getattr(d, "in_quant", True)]

    def fill_quant_images(self):
        box = self.quant_img_box
        while box.count():
            it = box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for k, d in enumerate(self.docs):
            if d.raw is None:
                continue
            cb = QtWidgets.QCheckBox(f"{self.doc_key(k)} · {d.name}")
            cb.setChecked(getattr(d, "in_quant", True))
            cb.setToolTip(d.project or "")
            cb.toggled.connect(lambda on, d=d: (setattr(d, "in_quant", on), self.recompute_results()))
            box.addWidget(cb)

    def basis_of(self, d):
        """(analysis, measurements) as results see them: per lane, or per group of the chosen set."""
        if d.basis and core.grouping(d.analysis, d.basis) is None:
            d.basis = LANES_BASIS
        return core.basis_view(d.analysis, d.measurements, d.basis or None)

    def ref_of(self, d):
        a, _ = self.basis_of(d)
        if self.ref_name and self.ref_name in a.lane_labels:
            return a.lane_labels.index(self.ref_name)
        return None

    def _scale(self, v, labels):
        """1.0 기준: divide by the mean of the lanes (groups) with the reference name."""
        v = np.asarray(v, float)
        if not self.ref_name or v.ndim != 1:
            return v
        idx = [i for i, lab in enumerate(labels) if lab == self.ref_name and i < v.size]
        ref = np.nanmean(v[idx]) if idx else np.nan
        return v / ref if np.isfinite(ref) and ref != 0 else v

    def fill_basis_combo(self):
        cb = self.basis_combo
        cur = self.doc.basis
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("레인 (기본)", LANES_BASIS)
        names = []
        for d in self.loaded_docs() or [self.doc]:
            for g in d.analysis.groupings:
                if g["name"] not in names:
                    names.append(g["name"])
        for n in names:
            cb.addItem(f"묶음 세트 '{n}'", n)
        cb.setCurrentIndex(max(0, cb.findData(cur)))
        cb.blockSignals(False)
        n_sets = len(names)
        self.sets_btn.setText(f"측정 묶음 세트 {n_sets}개 · 편집…")
        self.fill_ctrl_combo()
        self.fill_ref_combo()

    def on_basis(self, *_):
        b = self.basis_combo.currentData()
        for d in self.loaded_docs() or [self.doc]:
            d.basis = b if b and core.grouping(d.analysis, b) is not None else LANES_BASIS
        if b:
            self.fill_groupings(select=b)
            self.view.refresh_items()
            self.view.rebuild_extras()
        self.fill_ref_combo()
        self.recompute_results()
        self.recompute_multi()
        self._figure_changed(self.doc)

    def fill_ctrl_combo(self):
        cb = self.ctrl_combo
        names = []
        for d in self.loaded_docs() or [self.doc]:
            for r in d.analysis.rows:
                if r.name not in names and r.role != core.ROLE_IGNORE:
                    names.append(r.name)
        default = next((r.name for r in self.analysis.rows if r.role == core.ROLE_CONTROL), None)
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("없음 (Net 값)", "")
        for n in names:
            cb.addItem(n, n)
        want = self.ctrl_name if self.ctrl_name is not None else default
        cb.setCurrentIndex(max(0, cb.findData(want or "")))
        cb.blockSignals(False)

    def on_ctrl(self, *_):
        self.ctrl_name = self.ctrl_combo.currentData()
        self.recompute_results()
        self.recompute_multi()

    def fill_ref_combo(self):
        names = []
        for d in self.quant_docs() or [self.doc]:
            a, _ = self.basis_of(d)
            for lab in a.lane_labels:
                if lab not in names:
                    names.append(lab)
        cb = self.ref_lane
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("나누지 않음", None)
        for n in names:
            cb.addItem(n, n)
        if names and self.quant_docs() and (self._ref_default and self.ref_name is None
                                             or getattr(self, "_ref_auto", False) and self.ref_name not in names):
            self.ref_name = names[0]  # 1.0 기준: the first lane group (e.g. DMSO) until the user picks another
            self._ref_default = False
            self._ref_auto = True
        cb.setCurrentIndex(max(0, cb.findData(self.ref_name)))
        cb.blockSignals(False)
        self.ref_name = cb.currentData()
        d = self.doc
        st = core.group_status(d.analysis, d.basis) if d.basis else []
        bad = [f"{x['group']}: {', '.join(x['problems'])}" for x in st if x["problems"]]
        self.basis_msg.setText(("⚠ " + " · ".join(bad)) if bad else
                               ("그룹마다 Target ÷ 같은 그룹의 Control" if d.basis else "레인마다 Target ÷ 같은 레인의 Control"))
        self.basis_msg.setStyleSheet(f"color: {_tok('red')}; font-size: 11px;" if bad else f"color: {_tok('label-secondary')}; font-size: 11px;")

    def on_ref_lane(self, *_):
        self.ref_name = self.ref_lane.currentData()
        self._ref_auto = False
        self.recompute_results()
        self.recompute_multi()

    # ---- results of one image
    def _control_row(self, a) -> Optional[int]:
        if self.ctrl_name == "":
            return None
        if self.ctrl_name:
            i = next((i for i, r in enumerate(a.rows) if r.name == self.ctrl_name), None)
            if i is not None:
                return i
        return next((i for i, r in enumerate(a.rows) if r.role == core.ROLE_CONTROL), None)

    def image_variables(self, k: int, a, meas) -> dict:
        """Formula names for image k: its proteins / bands, and Ij.<protein> of the other images
        matched by lane name."""
        v = core.formula_variables(a, meas)
        for j, dj in enumerate(self.docs):
            if dj.raw is None or not dj.measurements:
                continue
            aj, mj = self.basis_of(dj)
            for ri, r in enumerate(aj.rows):
                vec = core.row_vector(aj, mj, ri)
                v[f"I{j + 1}.{core._safe_name(r.name)}"] = (vec if j == k else
                                                            core.to_common(vec, aj.lane_labels, a.lane_labels))
        return v

    def all_formulas(self, d) -> List[tuple]:
        """(scope, index, formula): formulas of every image (graph sidebar) and of this image."""
        return [("all", i, f) for i, f in enumerate(self.formulas_all)] + \
               [("image", i, f) for i, f in enumerate(d.analysis.formulas)]

    def image_results(self, d: ImageDoc):
        """Results of one image: Target / Control and the formulas, per lane or per group of the
        chosen grouping set, divided by the 1.0 reference. Returns (results, messages)."""
        res: Dict[str, np.ndarray] = {}
        msgs = []
        if not d.measurements:
            return res, msgs
        a, meas = self.basis_of(d)
        labels = a.lane_labels
        c = self._control_row(a)
        if c is not None:
            for i, r in enumerate(a.rows):
                if i == c or r.role == core.ROLE_IGNORE or (a.ladder_row and r.name == a.ladder_row):
                    continue
                v = core.normalize_to_control(a, meas, i, c, None)
                if not np.isfinite(v).any():
                    continue
                res[f"{r.name} ÷ {a.rows[c].name}"] = self._scale(v, labels)
        else:
            for i, r in enumerate(a.rows):
                if r.role != core.ROLE_IGNORE and not (a.ladder_row and r.name == a.ladder_row):
                    res[f"{r.name} (Net)"] = core.row_vector(a, meas, i)
        k = self.docs.index(d) if d in self.docs else 0
        variables = self.image_variables(k, a, meas)
        for _, _, f in self.all_formulas(d):
            name = f.get("name") or f.get("expr")
            expr = str(f.get("expr", "")).strip()
            if not expr:
                continue
            try:
                val = core.evaluate(expr, variables)
                if f.get("normalize", "1") == "1" and np.ndim(val) == 1:
                    val = self._scale(val, labels)
                res[name] = np.atleast_1d(np.asarray(val, float))
            except Exception as e:
                msgs.append(f"{self.doc_key(k)} {name}: {e}")
        return res, msgs

    def quant_pooled(self):
        """[(image key, results, lane names)] of the checked images, and messages."""
        out, msgs = [], []
        for d in self.quant_docs():
            res, m = self.image_results(d)
            a, _ = self.basis_of(d)
            out.append((self.doc_key(self.docs.index(d)), res, list(a.lane_labels)))
            msgs += m
        return out, msgs

    def recompute_results(self, *_):
        if hasattr(self, "ref_lane") and self.quant_docs() and (self._ref_default and self.ref_name is None
                                                                 or getattr(self, "_ref_auto", False)):
            self.fill_ref_combo()
        self.results, _ = self.image_results(self.doc)
        if not hasattr(self, "series_box"):
            return
        self.fill_quant_images()
        self._pooled, msgs = self.quant_pooled()
        self.formula_msg.setText("\n".join(msgs) if msgs else FORMULA_HINT)
        self.formula_msg.setStyleSheet(f"color: {_tok('red')}; font-size: 11px;" if msgs else "")
        self.fill_series_rows()
        self.fill_results_table()
        self.replot()
        if hasattr(self, "fig_win") and not self.fig_win.isVisible():
            self.figure_editor.dirty = True  # numbers in the figure may have changed

    def series_names(self) -> List[str]:
        names = []
        for _, res, _ in getattr(self, "_pooled", []):
            for n, v in res.items():
                if n not in names and np.size(v) > 1:
                    names.append(n)
        return names

    def shown_series(self) -> List[str]:
        return [n for n in self.series_names() if self.series_on.get(n, True)]

    def graph_xy(self):
        """x labels and [(series, [values per x])] of the graph."""
        shown = set(self.shown_series())
        per = [(key, {n: v for n, v in res.items() if n in shown}, labels) for key, res, labels in getattr(self, "_pooled", [])]
        return graph.pool(per, bool(self.graph_opts.get("by_condition", True)), bool(self.graph_opts.get("split_images")))

    def replot(self, *_):
        if not hasattr(self, "plot"):
            return
        xl, data = self.graph_xy()
        names = [n for n, _ in data]
        if names and all(n.endswith(" (Net)") for n in names):
            ylabel = "Net intensity (raw)"
        else:
            ylabel = "Relative intensity" if self.ref_name else "Ratio"
        st = self.graph_style()
        self._graph_ax = graph._draw(self.plot.fig, xl, data, ylabel, st)
        self.plot.canvas.draw_idle()
        n_img = len(getattr(self, "_pooled", []))
        reps = max((len(v) for _, groups in data for v in groups), default=0)
        self.graph_sub.setText(f"이미지 {n_img}개 · n = 최대 {reps} · 막대 = 평균, 오차 막대 = {'SEM' if st.get('err') == 'sem' else 'SD'}, 점 = 각 레인")
        if hasattr(self, "series_style_box"):
            self.fill_series_styles(names)
            self.fill_text_boxes()

    # ---- graph items and formulas (sidebar)
    def fill_formulas(self):
        self.fill_series_rows()

    def formula_terms(self):
        a = self.analysis
        ids = sorted({m.band_id for m in self.measurements})
        return [
            ("단백질", [(r.name, core._safe_name(r.name)) for r in a.rows]),
            ("밴드 (값 하나)", [(i, i) for i in ids]),
        ]

    def fill_series_rows(self):
        box = self.series_box
        while box.count():
            it = box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        mine = {}
        for scope, i, f in self.all_formulas(self.doc):
            mine.setdefault(str(f.get("name") or f.get("expr")), (scope, i, f))
        names = self.series_names()
        if not names:
            box.addWidget(controls.hint("결과가 없습니다 — 단백질 역할(Control/Target)을 정하거나 수식을 추가하세요"))
        for name in names:
            row = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(self.series_on.get(name, True))
            cb.toggled.connect(lambda on, n=name: (self.series_on.__setitem__(n, on), self.replot(), self.fill_results_table()))
            hl.addWidget(cb, 1)
            if name in mine:
                scope, i, f = mine[name]
                cb.setStyleSheet(f"color: {_tok('blue')}; font-weight: 600;")
                for text, tip, fn in (("✎", "수식 편집", lambda _=None, s=scope, i=i: self.open_formula_card((s, i))),
                                      ("✕", "수식 삭제", lambda _=None, s=scope, i=i: self.delete_formula((s, i)))):
                    b = QtWidgets.QToolButton()
                    b.setText(text)
                    b.setToolTip(tip)
                    b.clicked.connect(fn)
                    hl.addWidget(b)
            else:
                sub = QtWidgets.QLabel("Net" if name.endswith(" (Net)") else "자동")
                theme.hint(sub)
                hl.addWidget(sub)
            box.addWidget(row)

    def _formula_list(self, scope):
        return self.formulas_all if scope == "all" else self.analysis.formulas

    def open_formula_card(self, which: Optional[tuple] = None):
        """The formula editor opens inside the sidebar (new formula, or ``which`` = (scope, index))."""
        if not self.measurements:
            self.statusBar().showMessage("먼저 이미지를 열고 밴드를 검출하세요")
            return
        self.close_formula_card()
        va, vm = self.basis_of(self.doc)
        if which is not None:
            formula = dict(self._formula_list(which[0])[which[1]])
        else:
            formula = {"name": f"수식{len(self.formulas_all) + 1}"}
        self.formula_row = which
        k = self.docs.index(self.doc)
        ed = FormulaEditor(self.formula_card, self.formula_terms(), self.image_variables(k, va, vm), formula, compact=True)
        # 항목 넣기: which image the protein buttons take (same image = per lane of each image)
        hb = QtWidgets.QHBoxLayout()
        img = QtWidgets.QComboBox()
        img.addItem("각 이미지", "")
        for j, d in enumerate(self.docs):
            if d.raw is not None:
                img.addItem(self.doc_key(j), f"I{j + 1}.")
        lane = QtWidgets.QComboBox()
        lane.addItem("같은 레인/그룹")
        hb.addWidget(img)
        hb.addWidget(lane)
        cap = controls.hint("항목 넣기 (고르면 커서 자리에 들어감)")
        ed.top.addWidget(cap)
        ed.top.addLayout(hb)
        ed.prefix_fn = lambda: img.currentData() or ""
        ed.normalize.setText(f"1.0 기준({self.ref_name})으로 나누기" if self.ref_name else "1.0 기준으로 나누기")
        labels = va.lane_labels

        def preview(vals):
            groups = {}
            for i, v in enumerate(vals):
                if i < len(labels) and np.isfinite(v):
                    groups.setdefault(labels[i], []).append(v)
            if ed.normalize.isChecked():
                vals = self._scale(vals, labels)
                groups = {}
                for i, v in enumerate(vals):
                    if i < len(labels) and np.isfinite(v):
                        groups.setdefault(labels[i], []).append(v)
            n = max((len(g) for g in groups.values()), default=0)
            txt = " · ".join(f"{g} {np.mean(v):.2f}" for g, v in list(groups.items())[:6])
            return f"미리보기 · 평균 (n = {n})\n{txt}"
        ed.preview_fn = preview
        ed.normalize.toggled.connect(lambda _=None: ed.refresh())
        ed.validChanged.connect(self.formula_ok.setEnabled)
        ed.submitted.connect(self.accept_formula_card)
        ed.refresh()
        self.formula_editor = ed
        self.formula_editor_box.addWidget(ed)
        self.formula_ok.setEnabled(ed.valid)
        self.formula_ok.setText("저장" if which is not None else "그래프에 추가")
        self.formula_card_title.setText("수식 편집" if which is not None else "새 수식")
        self.formula_card.setVisible(True)
        self.add_formula_btn.setVisible(False)
        ed.display.setFocus()

    def close_formula_card(self):
        if self.formula_editor is not None:
            self.formula_editor.setParent(None)
            self.formula_editor.deleteLater()
            self.formula_editor = None
        self.formula_card.setVisible(False)
        self.add_formula_btn.setVisible(True)

    def accept_formula_card(self):
        ed = self.formula_editor
        if ed is None or not ed.valid:
            return
        f = ed.result()
        if self.formula_row is not None:
            lst = self._formula_list(self.formula_row[0])
            if self.formula_row[1] < len(lst):
                lst[self.formula_row[1]] = f
            msg = f"수식 '{f['name']}'을(를) 고쳤습니다"
        else:
            self.formulas_all.append(f)
            msg = f"✓ '{f['name']}'을(를) 그래프에 추가했습니다"
        self.series_on[f["name"]] = True
        self.close_formula_card()
        self.recompute_results()
        self.statusBar().showMessage(msg)

    def add_formula(self):
        self.open_formula_card()

    def edit_formula(self, which=None):
        if which is not None and which is not False:
            self.open_formula_card(which if isinstance(which, tuple) else ("all", which))

    def delete_formula(self, which=None):
        if which is None or which is False:
            return
        which = which if isinstance(which, tuple) else ("all", which)
        lst = self._formula_list(which[0])
        if which[1] >= len(lst):
            return
        name = lst[which[1]].get("name")
        del lst[which[1]]
        self.close_formula_card()
        self.recompute_results()
        self.statusBar().showMessage(f"수식 '{name}'을(를) 지웠습니다")

    # ---- data table (그린 값 · 밴드별 · 요약)
    def fill_results_table(self):
        if not hasattr(self, "table_seg"):
            return
        mode = self.table_seg.value()
        t = self.results_table
        rows, picks, header = [], [], []
        err = "SEM" if self.graph_opts.get("err") == "sem" else "SD"
        if mode == "bands":
            header = ["이미지", "밴드 ID", "단백질", "역할", "레인", "레인 이름", "Raw", "배경", "Net"]
            for d in self.quant_docs():
                k = self.docs.index(d)
                a = d.analysis
                for m in d.measurements:
                    r = a.rows[m.row]
                    rows.append([self.doc_key(k), m.band_id, r.name, ROLE_SHORT.get(r.role, ""), f"L{m.lane + 1}",
                                 a.lane_labels[m.lane] if m.lane < len(a.lane_labels) else "",
                                 f"{m.raw:,.0f}", f"{m.background:,.0f}", f"{m.net:,.0f}"])
                    picks.append((k, [m.band_id]))
        elif mode == "summary":
            xl, data = self.graph_xy()
            header = ["항목", "그룹", "평균", err, "n"]
            for name, groups in data:
                for x, v in zip(xl, groups):
                    m_, e_, n_ = graph.summary(v, self.graph_opts.get("err", "sd"))
                    if n_:
                        rows.append([name, x, f"{m_:.4g}", f"{e_:.3g}" if n_ > 1 else "", str(n_)])
                        picks.append(None)
        else:
            ref = self.ref_name
            header = ["이미지", "레인", "이름", "항목", "Control 짝", "Net", "Control Net", "정규화", f"{ref} = 1" if ref else "값"]
            shown = set(self.shown_series())
            for d in self.quant_docs():
                k = self.docs.index(d)
                a, meas = self.basis_of(d)
                res, _ = self.image_results(d)
                c = self._control_row(a)
                ids = {}
                for m in meas:
                    ids.setdefault((m.row, m.lane), []).append(m.band_id)
                for name, v in res.items():
                    if name not in shown or np.size(v) <= 1:
                        continue
                    ri = next((i for i, r in enumerate(a.rows) if c is not None and name == f"{r.name} ÷ {a.rows[c].name}"), None)
                    tv = core.row_vector(a, meas, ri) if ri is not None else None
                    cv = core.row_vector(a, meas, c) if (ri is not None and c is not None) else None
                    for i in range(np.size(v)):
                        lab = a.lane_labels[i] if i < len(a.lane_labels) else ""
                        rows.append([self.doc_key(k), f"L{i + 1}" if not d.basis else lab, lab, name,
                                     f"{lab} {a.rows[c].name}" if ri is not None else "",
                                     f"{tv[i]:,.0f}" if tv is not None and np.isfinite(tv[i]) else "",
                                     f"{cv[i]:,.0f}" if cv is not None and np.isfinite(cv[i]) else "",
                                     f"{tv[i] / cv[i]:.3f}" if tv is not None and cv is not None and cv[i] else "",
                                     f"{v[i]:.3f}" if np.isfinite(v[i]) else ""])
                        picks.append((k, (ids.get((ri, i), []) + (ids.get((c, i), []) if c is not None else [])) if ri is not None else []))
        t.blockSignals(True)
        t.clear()
        t.setColumnCount(len(header))
        t.setHorizontalHeaderLabels(header)
        t.setRowCount(len(rows))
        for r_, row in enumerate(rows):
            for c_, x in enumerate(row):
                it = QtWidgets.QTableWidgetItem(str(x))
                if c_ >= len(header) - 4 and mode != "summary":
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r_, c_, it)
        t.blockSignals(False)
        t.resizeColumnsToContents()
        self._table_picks = picks

    def on_table_pick(self):
        rows = sorted({i.row() for i in self.results_table.selectedIndexes()})
        picks = getattr(self, "_table_picks", [])
        if not rows or rows[0] >= len(picks) or picks[rows[0]] is None:
            return
        k, ids = picks[rows[0]]
        if k != self.cur:
            self.switch_doc(k)
        ids = set(ids)
        bid = self.analysis.band_ids()
        sel = [b for b in self.analysis.bands if bid[id(b)] in ids]
        self.view.select_bands(sel)
        if sel:
            self.view.centerOn(sel[0].cx, sel[0].cy)

    def edit_groupings(self):
        """측정 묶음 세트 편집 (moved here from the lanes tab)."""
        dlg = QtWidgets.QDialog(self.quant_win)
        dlg.setWindowTitle("측정 묶음 세트")
        lay = QtWidgets.QVBoxLayout(dlg)
        self._groupings_widget.setParent(None)
        lay.addWidget(self._groupings_widget)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.resize(520, 480)
        self.fill_groupings()
        dlg.exec()
        self._groupings_widget.setParent(None)
        self.fill_basis_combo()

    # ---------------------------------------------------------------- figure window
    def open_figure_window(self):
        self.fig_win.show_for(self.doc)

    def open_quant_window(self, page: Optional[str] = None):
        self.quant_win.show_page(page)

    def _figure_changed(self, d):
        ed = self.figure_editor
        if self.fig_win.isVisible():
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
            "이미지 사이에서 레인은 레인 그룹 이름(또는 레인 큰 그룹)으로 맞춰집니다.\n"
            "여러 프로젝트를 함께 비교하려면 파일 → '프로젝트 열기'에서 여러 개를 고르거나 '프로젝트 추가로 불러오기'를 쓰세요.\n"
            "각 이미지는 I1, I2 …로 불리고 [프로젝트 이름]이 붙습니다. '이미지 간 수식'에서 예) I1.Vinculin / I3.Vinculin"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {_tok('label-secondary')};")
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
        self.multi_plot.style_fn = self.graph_style
        lay.addWidget(self.multi_plot)
        self.multi_table = QtWidgets.QTableWidget(0, 0)
        self.multi_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.multi_table.setMinimumHeight(200)
        lay.addWidget(self.multi_table, 1)
        return self._scroll(w)

    # ---------------------------------------------------------------- image
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile()]
        if paths:
            e.acceptProposedAction()
            QtCore.QTimer.singleShot(0, lambda: self.open_paths(paths))

    @staticmethod
    def is_marker_file(p: str) -> bool:
        return "marker" in os.path.basename(p).lower()

    def open_path(self, p: str):
        self.open_paths([p])

    def open_paths(self, paths):
        """Open analysis files and blots; files named '…marker…' become the marker of their blot
        (or of the current image when no blot is opened with them)."""
        paths = [p for p in paths if p]
        markers = [p for p in paths if self.is_marker_file(p) and not projfile.is_project_file(p)]
        projects = [p for p in paths if projfile.is_project_file(p)]
        if projects:
            if any(d.raw is not None for d in self.docs) and len(projects) == 1 and not self._ask_replace():
                self.load_project(projects[0], ask_save=False, append=True)
            else:
                self.open_projects(projects)
        blots = [p for p in paths if p not in markers and not projfile.is_project_file(p)]
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
        """Step 2 after opening a blot: the marker image. A marker opened/dropped together with
        the blot is attached directly; otherwise the step window explains and asks for it."""
        path = self.find_marker_for(d.image_path, candidates)
        if path and path in candidates:
            if self.set_marker(d, path):
                turned = int(d.marker_tf.get("rot90", 0)) % 2 == 1
                self.statusBar().showMessage(
                    f"함께 연 마커 이미지를 붙였습니다: {os.path.basename(path)}" +
                    (f" — 가로·세로가 바뀌어 있어 '{core.orientation_label(d.marker_tf)}'로 돌려 맞췄습니다. "
                     "방향이 다르면 1번 탭 '마커 방향'에서 고르세요." if turned else ""))
            return
        dlg = MarkerStepDialog(self, d, path)
        if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.path:
            if self.set_marker(d, dlg.path, marker_tf=dlg.marker_tf):
                self.statusBar().showMessage(f"마커 이미지를 붙였습니다: {os.path.basename(dlg.path)} — 왼쪽 마커 위 파란 상자로 띠 영역을 고칠 수 있습니다")

    def open_marker(self):
        if self.doc.raw is None:
            self.statusBar().showMessage("먼저 블롯 TIF를 여세요")
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "마커(ladder) 이미지 열기", os.path.dirname(self.doc.image_path or ""),
                                                     "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if p:
            self.set_marker(self.doc, p)

    def set_marker(self, d: ImageDoc, path: str, quiet: bool = False, marker_tf: Optional[dict] = None) -> bool:
        """Attach a marker image (same geometry as the blot) to image ``d``.

        ``marker_tf`` turns/mirrors the photo first; None = keep it as is, or, when
        its width and height are swapped relative to the blot and the option is on,
        turn it 90° so they fit (the step window lets the user confirm the direction).
        """
        try:
            m = core.load_image(path)
            base = d.raw_orig if d.raw_orig is not None else d.raw
            turned = False
            if marker_tf is None:
                marker_tf = {"rot90": 0, "flip": False}
                if self.auto_orient_marker.isChecked() and not core._aspect_ok(m.shape, base.shape):
                    cands = core.marker_orientations(m.shape, base.shape)
                    if cands:
                        marker_tf = {"rot90": cands[0]["rot90"], "flip": cands[0]["flip"]}
                        turned = True
            marker_tf = {"rot90": int(marker_tf.get("rot90", 0)) % 4, "flip": bool(marker_tf.get("flip", False))}
            m = core.orient(m, marker_tf)
            resized = m.shape != base.shape
            if resized:
                m = figure.match_marker(m, base.shape)
            d.marker_orig = m
            m = core.apply_transform(m, d.transform)
        except Exception as e:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "마커 이미지", f"쓸 수 없는 마커 이미지입니다:\n{e}")
            return False
        if resized and not quiet:
            QtWidgets.QMessageBox.information(self, "마커 이미지 크기", "마커 이미지의 크기가 블롯과 달라 같은 비율로 맞췄습니다. "
                                              "두 이미지가 같은 위치에서 찍힌 것인지 확인하세요.")
        d.marker_tf = marker_tf
        d.marker = m
        if turned:
            self.statusBar().showMessage(f"마커의 가로·세로가 블롯과 바뀌어 있어 '{core.orientation_label(marker_tf)}'로 돌려 맞췄습니다. "
                                         "방향이 다르면 1번 탭 '마커 방향'에서 고르세요.")
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

    def on_marker_orient(self, *_):
        d = self.doc
        if self._updating or d.marker is None or not (d.figure or {}).get("marker_path"):
            return
        k, flip = (int(x) for x in self.marker_orient.currentData().split(","))
        flip = bool(flip)
        base = d.raw_orig if d.raw_orig is not None else d.raw
        try:
            raw_m = core.load_image(d.figure["marker_path"])
        except Exception:
            raw_m = None
        if raw_m is not None and not core._aspect_ok(core.orient(raw_m, {"rot90": k, "flip": flip}).shape, base.shape):
            QtWidgets.QMessageBox.information(self, "마커 방향", "이 방향으로는 마커의 가로·세로가 블롯과 맞지 않습니다.")
            self.sync_marker_orient()
            return
        self.set_marker(d, d.figure["marker_path"], marker_tf={"rot90": k, "flip": flip})
        self.statusBar().showMessage(f"마커 방향: {core.orientation_label(d.marker_tf)}")

    def sync_marker_orient(self):
        d = self.doc
        self._updating = True
        try:
            tf = d.marker_tf or {}
            i = self.marker_orient.findData(f"{int(tf.get('rot90', 0)) % 4},{int(bool(tf.get('flip', False)))}")
            self.marker_orient.setCurrentIndex(max(0, i))
            self.marker_orient.setEnabled(d.marker is not None)
        finally:
            self._updating = False

    def clear_marker(self):
        d = self.doc
        d.marker = None
        d.marker_orig = None
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
        self.sync_marker_orient()
        if m is None or not self.show_marker.isChecked():
            self.view.set_marker(None)
            self.view.marker_s = None
            self.view.rebuild_lines()
            self.refresh_marker_card()
            return
        x = figure._stretch(m, 1, 99.5, 1.0)
        f = d.figure or {}
        if f.get("marker_invert", True):
            x = 1.0 - x  # dark ladder on white, like the blot preview
        self.view.set_marker(to_qimage(x, 0.0, 1.0))
        self.view.marker_s = self._marker_s()
        self.view.rebuild_lines()
        self.refresh_marker_card()

    def set_strip_region(self, d: ImageDoc, x0, x1, dy):
        """Marker strip region dragged on the main view's marker preview."""
        if d.figure is None:
            d.figure = figure.default_settings()
        d.figure.update(strip_x0=int(x0), strip_x1=int(x1), strip_dy=int(dy), show_strip=True)
        if self.figure_editor.doc is d:
            self.figure_editor.set_strip_region(x0, x1, dy)
        else:
            self.refresh_marker_view()
            self._figure_changed(d)
        self.statusBar().showMessage(f"마커 띠 영역: x {x0}~{x1} px, 세로 보정 {dy} px (Figure에 쓰이는 범위)")

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
        d.raw_orig = raw
        d.transform = core.default_transform()
        d.marker_orig = None
        d.raw = raw
        d.image_path = path
        d.analysis = core.Analysis()
        d.measurements = []
        d.marker = None
        d.figure = None
        self._make_signal(d)
        d.mode = self.suggest_mode(d, marker_candidates)
        self.refresh_doc_tabs()
        self.switch_doc(self.docs.index(d))
        if d.mode == core.MODE_GEL and self.auto_gel_crop.isChecked():
            self.auto_crop_gel(redetect=False)
        if detect:
            self.detect()
        if d.mode == core.MODE_GEL:
            self.statusBar().showMessage(self.statusBar().currentMessage() +
                                         "  · 자동 선택: DNA gel 모드 (이미지 안에 ladder 레인이 보입니다) — 1번 탭에서 바꿀 수 있습니다")
        elif ask_marker and self.ask_marker.isChecked():
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
    # ---------------------------------------------------------------- experiment type and orientation
    def _build_orient(self):
        g = QtWidgets.QGroupBox("실험 종류 · 이미지 방향 · 자르기")
        f = QtWidgets.QFormLayout(g)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Western blot (마커 사진이 따로 있음)", core.MODE_WESTERN)
        self.mode_combo.addItem("DNA gel 전기영동 (ladder가 이미지 안에 있음)", core.MODE_GEL)
        self.mode_combo.setToolTip("이미지를 열 때 자동으로 정합니다: 짝이 되는 마커 파일이 있으면 Western, "
                                   "이미지 안에 ladder 레인이 보이면 DNA gel, 애매하면 Western. 언제든 바꿀 수 있습니다.")
        self.mode_combo.currentIndexChanged.connect(lambda _: self.set_mode(self.doc, self.mode_combo.currentData()))
        f.addRow("실험 종류", self.mode_combo)
        hb = QtWidgets.QHBoxLayout()
        for text, tip, fn in (("⟲ 90°", "반시계 방향 90° 회전", lambda: self.rotate90(1)),
                              ("⟳ 90°", "시계 방향 90° 회전", lambda: self.rotate90(-1)),
                              ("180°", "180° 회전", lambda: self.rotate90(2))):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            hb.addWidget(b)
        self.angle_spin = _spin(-45.0, 45.0, 0.0, 0.1, 1, "°")
        self.angle_spin.setToolTip("미세 회전 (반시계 방향 +). 회전·자르기를 바꾸면 밴드를 다시 검출합니다")
        self.angle_spin.setKeyboardTracking(False)
        self.angle_spin.valueChanged.connect(self.set_fine_angle)
        hb.addWidget(QtWidgets.QLabel("미세"))
        hb.addWidget(self.angle_spin)
        f.addRow("회전", hb)
        hb = QtWidgets.QHBoxLayout()
        self.line_btn = QtWidgets.QPushButton("선으로 수평 맞추기")
        self.line_btn.setCheckable(True)
        self.line_btn.setToolTip("이미지 위에서 well이나 밴드를 따라 선을 그으면 그 선이 수평이 되도록 돌립니다")
        self.line_btn.toggled.connect(lambda on: self.set_draw_mode("line" if on else None))
        hb.addWidget(self.line_btn)
        self.crop_btn = QtWidgets.QPushButton("자르기 영역 그리기")
        self.crop_btn.setCheckable(True)
        self.crop_btn.setToolTip("이미지 위에서 남길 영역(겔)을 드래그해 그립니다. 트레이나 배경을 잘라냅니다")
        self.crop_btn.toggled.connect(lambda on: self.set_draw_mode("crop" if on else None))
        hb.addWidget(self.crop_btn)
        b = QtWidgets.QPushButton("자르기 해제")
        b.clicked.connect(lambda: self.set_transform(self.doc, dict(self.doc.transform, crop=None)))
        hb.addWidget(b)
        b = QtWidgets.QPushButton("원본으로")
        b.clicked.connect(lambda: self.set_transform(self.doc, core.default_transform()))
        hb.addWidget(b)
        f.addRow(hb)
        hb = QtWidgets.QHBoxLayout()
        self.auto_gel_crop = QtWidgets.QCheckBox("겔 사진을 열면 겔 영역만 자동으로 자르기")
        self.auto_gel_crop.setChecked(True)
        self.auto_gel_crop.setToolTip("DNA gel 모드로 열린 사진에서 트레이·배경을 빼고 겔 부분만 남깁니다. '자르기 해제'로 되돌립니다")
        hb.addWidget(self.auto_gel_crop)
        b = QtWidgets.QPushButton("겔 영역 자동 찾기")
        b.clicked.connect(self.auto_crop_gel)
        hb.addWidget(b)
        f.addRow(hb)
        self.orient_label = QtWidgets.QLabel()
        self.orient_label.setStyleSheet(f"color: {_tok('label-secondary')}; font-size: 11px;")
        f.addRow(self.orient_label)
        return g

    def set_draw_mode(self, mode):
        for b, m in ((self.line_btn, "line"), (self.crop_btn, "crop")):
            if m != mode and b.isChecked():
                b.blockSignals(True)
                b.setChecked(False)
                b.blockSignals(False)
        self.view.set_draw_mode(mode)
        if mode:
            self.statusBar().showMessage("이미지 위에서 " + ("well/밴드를 따라 선을 그으세요 (그 선이 수평이 됩니다)" if mode == "line"
                                                       else "남길 영역을 드래그하세요") + " · Esc: 취소")

    def on_drawn(self, kind, x0, y0, x1, y1):
        self.set_draw_mode(None)
        d = self.doc
        if d.raw is None:
            return
        tf = dict(d.transform)
        if kind == "line":
            delta = core.line_angle(x0, y0, x1, y1)
            if abs(delta) < 0.05:
                return
            tf["angle"] = round(float(tf.get("angle", 0.0)) + delta, 2)  # ndimage.rotate(+a) turns image-y-down lines by -a
            tf["crop"] = None  # the old crop was in the old orientation
        else:
            ox, oy = (tf["crop"][0], tf["crop"][1]) if tf.get("crop") else (0, 0)
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            tf["crop"] = [int(ox + xa), int(oy + ya), int(ox + xb), int(oy + yb)]
        self.set_transform(d, tf)

    def auto_crop_gel(self, redetect: bool = True) -> bool:
        d = self.doc
        if d.raw_orig is None:
            return False
        base = core.apply_transform(d.raw_orig, dict(d.transform, crop=None))
        crop = core.suggest_gel_crop(base)
        if not crop:
            self.statusBar().showMessage("겔 영역을 자동으로 찾지 못했습니다. '자르기 영역 그리기'로 직접 정하세요")
            return False
        self.set_transform(d, dict(d.transform, crop=crop), redetect=redetect)
        return True

    def rotate90(self, k):
        d = self.doc
        if d.raw is None:
            return
        tf = dict(d.transform)
        tf["rot90"] = (int(tf.get("rot90", 0)) + k) % 4
        tf["crop"] = None
        self.set_transform(d, tf)

    def set_fine_angle(self, v):
        d = self.doc
        if self._updating or d.raw is None or abs(float(d.transform.get("angle", 0.0)) - v) < 1e-6:
            return
        self.set_transform(d, dict(d.transform, angle=float(v), crop=None))

    def set_transform(self, d: ImageDoc, tf: dict, redetect: bool = True):
        """Rotate/crop image ``d`` (and its marker); boxes are found again in the new geometry."""
        if d.raw_orig is None:
            return
        d.transform = {"rot90": int(tf.get("rot90", 0)) % 4, "angle": float(tf.get("angle", 0.0) or 0.0),
                       "crop": tf.get("crop")}
        d.raw = core.apply_transform(d.raw_orig, d.transform)
        if d.marker_orig is not None:
            d.marker = core.apply_transform(d.marker_orig, d.transform)
        self._make_signal(d)
        if d.figure:  # positions in pixels belong to the old geometry
            d.figure["kda"] = []
            d.figure["gel_crop"] = None
            if d.marker is not None and d.analysis.bands:
                d.figure["strip_x0"], d.figure["strip_x1"] = 0, 0
        if d is self.doc:
            self.sync_orient_widgets()
            self.update_display(reset_zoom=True)
            self.refresh_marker_view()
        if redetect:
            cur = self.cur
            self.cur = self.docs.index(d)
            self.detect()
            self.cur = cur
            if d.figure and d.marker is not None and d.analysis.bands and not d.figure.get("strip_x1"):
                d.figure["strip_x0"], d.figure["strip_x1"] = figure.guess_ladder_x(d.marker, d.analysis)
        H, W = d.raw.shape
        self.statusBar().showMessage(f"이미지 방향/자르기를 적용했습니다 ({W}×{H}). 밴드를 다시 검출했습니다.")

    def sync_orient_widgets(self):
        d = self.doc
        self._updating = True
        try:
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(d.mode)))
            self.angle_spin.setValue(float(d.transform.get("angle", 0.0)))
        finally:
            self._updating = False
        tf = d.transform
        parts = []
        if tf.get("rot90"):
            parts.append(f"{90 * int(tf['rot90'])}° 회전")
        if abs(float(tf.get("angle", 0.0))) > 1e-6:
            parts.append(f"미세 {tf['angle']:+.1f}°")
        if tf.get("crop"):
            x0, y0, x1, y1 = tf["crop"]
            parts.append(f"자름 {x1 - x0}×{y1 - y0}")
        self.orient_label.setText("현재: " + (", ".join(parts) if parts else "원본 그대로") +
                                  " · 정량은 회전/자른 이미지의 raw 값으로 합니다 (미세 회전은 선형 보간)")

    def set_mode(self, d: ImageDoc, mode: str, redetect: bool = True):
        if self._updating or d.raw is None or mode == d.mode:
            return
        d.mode = mode
        if mode == core.MODE_GEL and d.bg.method == "rolling_ball" and d is self.doc:
            pass  # keep the user's background choice
        if d is self.doc:
            self.sync_orient_widgets()
        if redetect:
            self.detect()
        if d.figure is not None:
            figure.apply_mode_defaults(d.figure, d.analysis, mode)
            self._figure_changed(d)
        self.statusBar().showMessage(("DNA gel 모드: 레인별로 밴드를 찾고, ladder 레인을 따로 묶었습니다 (Figure는 겔 전체 한 패널)."
                                      if mode == core.MODE_GEL else "Western blot 모드: 단백질(행)과 레인으로 밴드를 찾습니다."))

    def suggest_mode(self, d: ImageDoc, marker_candidates=()) -> str:
        """Western when a separate marker file belongs to the image; DNA gel when the image
        itself has ladder-like lanes; Western when unsure."""
        if self.find_marker_for(d.image_path, marker_candidates):
            return core.MODE_WESTERN
        try:
            corr = self.get_corrected(d)
            return core.MODE_GEL if core.looks_like_gel(corr) else core.MODE_WESTERN
        except Exception:
            return core.MODE_WESTERN

    def set_ladder_selected(self, on: bool):
        a = self.analysis
        lanes = {b.lane for b in self.view.selected_bands()}
        if not lanes:
            return
        cur = set(core.ladder_lanes(a))
        core.set_ladder_lanes(a, (cur | lanes) if on else (cur - lanes))
        if self.doc.figure is not None:
            self.doc.figure["kda"] = []
            figure.apply_mode_defaults(self.doc.figure, a, self.doc.mode)
        self.structure_changed()
        self.statusBar().showMessage(f"ladder 레인: {', '.join(a.lane_labels[l] for l in core.ladder_lanes(a)) or '없음'}")

    def detect(self, use_lines: bool = False):
        """Find the boxes. ``use_lines``: along the blue lane lines and red protein lines the user
        drew or edited (선 기준으로 다시 검출)."""
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
        user_bands = [(b.cx, b.cy, old.rows[b.row].name) for b in old.bands if getattr(b, "user", False)]
        self._tagged = ([(b.cx, b.cy, dict(b.tags)) for b in old.bands if b.tags],
                        [dict(g, groups=list(g["groups"])) for g in old.groupings])
        if use_lines and not old.lane_lines:
            self.statusBar().showMessage("레인 선이 없습니다 — '레인 선 자동으로 찾기'나 레인 선 도구로 먼저 그리세요")
            return
        if self.method_combo.currentData() == "lane" or use_lines:
            self._detect_lane_first(corr, settings, old, user_bands, use_lines)
            return
        if self.doc.mode == core.MODE_GEL:
            a = core.detect_gel(corr, settings)
            if old.lane_labels and len(old.lane_labels) == a.n_lanes:
                a.lane_labels = list(old.lane_labels)
            a.formulas = list(old.formulas)
            self._restore_groups(a)
            core.lines_from_analysis(a, corr.shape)
            self.set_analysis(a)
            if self.doc.figure is not None:
                figure.apply_mode_defaults(self.doc.figure, a, core.MODE_GEL)
            lad = core.ladder_lanes(a)
            self.statusBar().showMessage(
                f"DNA gel: 레인 {a.n_lanes}개, 밴드 {len(a.bands)}개, ladder 레인 {len(lad)}개"
                f"({', '.join(a.lane_labels[l] for l in lad) or '없음'}). 밴드를 선택해 우클릭 → 'ladder로 지정/빼기'로 고칠 수 있습니다.")
            return
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
        self._restore_groups(a)
        core.lines_from_analysis(a, corr.shape)
        self.set_analysis(a)
        self.statusBar().showMessage(
            f"자동 그룹 제안: 단백질 {len(a.rows)}개({', '.join(r.name for r in a.rows)}), 레인 {a.n_lanes}개, 박스 {len(a.bands)}개. "
            "바로 '6. Figure 만들기'로 가거나, 밴드를 드래그로 선택 → 우클릭으로 그룹 이름·구성을 고치세요."
        )

    def _detect_lane_first(self, corr, settings, old, user_bands, use_lines: bool = False):
        d = self.doc
        if use_lines:
            a = lanes.detect(corr, settings, d.mode, lane_lines=old.lane_lines, protein_lines=old.protein_lines,
                             only_on_lines=self.only_on_lines.isChecked())
        else:
            a = lanes.detect(corr, settings, d.mode)
        if d.mode != core.MODE_GEL:
            for i, r in enumerate(a.rows):  # keep the user's names and roles when re-detecting
                if i < len(old.rows):
                    r.name = core.unique_name(old.rows[i].name, [x.name for x in a.rows[:i]])
                    r.role = old.rows[i].role
            if a.rows and not any(r.role == core.ROLE_CONTROL for r in a.rows):
                a.rows[0].role = core.ROLE_CONTROL
        if old.lane_labels and len(old.lane_labels) == a.n_lanes:
            a.lane_labels = list(old.lane_labels)
            a.named_lanes = [n for n in old.named_lanes if n in a.lane_labels]
        a.formulas = list(old.formulas)
        # boxes the user placed or moved stay
        kept = 0
        for x, y, _name in user_bands:
            near = any(abs(b.cx - x) < 0.5 * a.rows[b.row].width and abs(b.cy - y) < 0.5 * a.rows[b.row].height for b in a.bands)
            if not near and a.rows:
                nb = core.add_band(a, x, y)
                nb.user = True
                kept += 1
        if kept and a.method == "lane":
            core.auto_assign_lanes(a)
        self._restore_groups(a)
        if use_lines:  # the user's lines stay as they are
            a.lane_lines = [[list(q) for q in ln] for ln in old.lane_lines]
            a.protein_lines = [dict(p) for p in old.protein_lines]
        else:
            core.lines_from_analysis(a, corr.shape)
        if len(old.lane_roles) == a.n_lanes:
            a.lane_roles = list(old.lane_roles)
        # tilt: corrected only by turning the whole image (and the marker with it)
        ang = lanes.tilt_angle(a)
        if self.auto_tilt.isChecked() and not use_lines and not self._tilt_done and 0.3 <= abs(ang) <= 10:
            self._tilt_done = True
            try:
                self.set_transform(d, dict(d.transform, angle=round(float(d.transform.get("angle", 0.0)) + ang, 2)))
            finally:
                self._tilt_done = False
            self.statusBar().showMessage(
                f"밴드 줄이 {ang:+.1f}° 기울어 있어 이미지 전체(마커 포함)를 돌려 맞추고 다시 검출했습니다 — 1번 탭 '미세' 회전에서 되돌릴 수 있습니다")
            return
        self.set_analysis(a)
        if d.figure is not None:
            figure.apply_mode_defaults(d.figure, a, d.mode)
        lad = core.ladder_lanes(a)
        msg = (f"레인 {a.n_lanes}개, 박스 {len(a.bands)}개, 줄 {len(a.rows)}개"
               + (f", ladder 레인 {', '.join(a.lane_labels[l] for l in lad)}" if lad else "")
               + ". 박스는 끌어서 옮기고, 더블클릭하면 크기를 고칩니다.")
        if kept:
            msg += f" 직접 고친 박스 {kept}개는 그대로 두었습니다."
        self.statusBar().showMessage(msg)

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
            self.statusBar().showMessage("박스 위치에 맞춰 레인을 다시 지정했습니다 (이름 붙인 레인 그룹과 측정 묶음은 유지)")
        self.view.rebuild_extras()
        self.update_sel_card()  # box sizes may have been changed by dragging in resize mode
        self.remeasure()
        self._figure_changed(self.doc)

    def add_band_at(self, x, y):
        core.add_band(self.analysis, x, y).user = True
        if self.auto_lanes.isChecked():
            core.auto_assign_lanes(self.analysis)
        self.structure_changed()

    def delete_selected(self):
        if self.view.tool != "select":
            self.view.delete_selection()
            return
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
            gm = menu.addMenu(f"그룹 설정 ({len(sel)}개 박스)")
            gm.addAction("단백질(행) 그룹으로 묶기…", self.make_protein_group)
            gm.addAction("레인(열) 그룹으로 묶기…", self.make_lane_group)
            mm = menu.addMenu(f"측정 묶음에 넣기 ({len(sel)}개 박스) — Control·Target 짝짓기")
            mm.addAction("새 그룹… (세트·그룹 고르기)", self.assign_group)
            for g in self.analysis.groupings:
                sm = mm.addMenu(f"'{g['name']}'에 넣기")
                for gname in g["groups"]:
                    sm.addAction(f"그룹 {gname}", lambda s_=g["name"], g_=gname: self.put_in_group(sel, s_, g_))
                sm.addAction(f"새 그룹 {core.next_group_name(self.analysis, g['name'])}",
                             lambda s_=g["name"]: self.put_in_group(sel, s_, core.next_group_name(self.analysis, s_)))
            in_sets = sorted({k for b in sel for k in b.tags})
            for sname in in_sets:
                mm.addAction(f"'{sname}'에서 빼기", lambda s_=sname: self.remove_from_group(s_))
            if self.doc.mode == core.MODE_GEL:
                gm.addAction("선택한 박스의 레인을 ladder로 지정", lambda: self.set_ladder_selected(True))
                gm.addAction("선택한 박스의 레인을 ladder에서 빼기", lambda: self.set_ladder_selected(False))
            rows = sorted({b.row for b in sel})
            if len(rows) == 1:
                ro = gm.addMenu(f"'{self.analysis.rows[rows[0]].name}' 역할")
                for role in core.ROLES:
                    ro.addAction(ROLE_LABELS[role], lambda role=role, r=rows[0]: self.set_row_role(r, role))
            menu.addSeparator()
            if len(sel) >= 2:
                menu.addAction(f"병합 (선택한 {len(sel)}개 박스를 하나로)", self.merge_selected)
            if self.resize_mode.isChecked():
                menu.addAction("크기 수정 끝내기 (Esc)", lambda: self.resize_mode.setChecked(False))
            else:
                menu.addAction("크기 수정 (더블클릭)", lambda: self.resize_mode.setChecked(True))
            menu.addAction("삭제 (Backspace)", self.delete_selected)
        else:
            menu.addAction("여기에 박스 추가 (빈 곳 더블클릭)", lambda: self.add_band_at(*self._scene_pos(gpos)))
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

    def _restore_groups(self, a):
        tagged, sets = self._tagged
        a.groupings = [dict(g, groups=list(g["groups"])) for g in sets]
        core.restore_tags(a, tagged)

    # ---------------------------------------------------------------- measurement groups (grouping sets)
    def assign_group(self):
        sel = self.view.selected_bands()
        if not sel:
            self.statusBar().showMessage("이미지에서 박스를 먼저 고르세요 (클릭, Shift/⌘-클릭, 드래그)")
            return
        dlg = GroupAssignDialog(self, self.analysis, sel, self.current_set())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        sname, gname = dlg.values()
        if sname and gname:
            self.put_in_group(sel, sname, gname)

    def put_in_group(self, bands, sname, gname):
        a = self.analysis
        core.set_group(a, bands, sname, gname)
        self.groups_changed(show=sname)
        st = next((x for x in core.group_status(a, sname) if x["group"] == gname), None)
        warn = f" — 아직: {', '.join(st['problems'])}" if st and st["problems"] else ""
        self.statusBar().showMessage(f"'{sname}' › 그룹 '{gname}'에 박스 {len(bands)}개를 넣었습니다{warn}. "
                                     "정량 탭의 'Control 짝짓기 기준'에서 이 세트를 고르면 그룹별로 계산합니다")

    def remove_from_group(self, sname):
        sel = self.view.selected_bands()
        core.clear_group(self.analysis, sel, sname)
        self.groups_changed(show=sname)
        self.statusBar().showMessage(f"박스 {len(sel)}개를 '{sname}'에서 뺐습니다")

    def current_set(self) -> Optional[str]:
        return self.set_list.currentData() if hasattr(self, "set_list") else None

    def groups_changed(self, show: Optional[str] = None):
        """Grouping sets or their members changed: tables, image, results."""
        self.fill_groupings(select=show)
        self.view.refresh_items()
        self.view.rebuild_extras()
        self.fill_basis_combo()
        self.recompute_results()
        self.recompute_multi()
        self._figure_changed(self.doc)

    def fill_groupings(self, select: Optional[str] = None):
        a = self.analysis
        core.sync_groupings(a)  # boxes may have been deleted
        cur = select or self.current_set()
        self.set_list.blockSignals(True)
        self.set_list.clear()
        self.set_list.addItem("(표시 안 함)", None)
        for g in a.groupings:
            self.set_list.addItem(g["name"], g["name"])
        i = self.set_list.findData(cur) if cur else 0
        self.set_list.setCurrentIndex(i if i >= 0 else (1 if a.groupings else 0))
        self.set_list.blockSignals(False)
        self.view.show_set = self.set_list.currentData()
        self.fill_group_table()

    def on_set_chosen(self, *_):
        self.view.show_set = self.set_list.currentData()
        self.view.refresh_items()
        self.view.rebuild_extras()
        self.fill_group_table()

    def fill_group_table(self):
        a = self.analysis
        sname = self.current_set()
        t = self.group_table
        t.blockSignals(True)
        rows = core.group_status(a, sname) if sname else []
        t.setRowCount(len(rows))
        ids = a.band_ids()
        for i, st in enumerate(rows):
            it = QtWidgets.QTableWidgetItem(st["group"])
            it.setForeground(QtGui.QColor(GROUP_COLORS[i % len(GROUP_COLORS)]))
            f = it.font()
            f.setBold(True)
            it.setFont(f)
            t.setItem(i, 0, it)
            boxes = ", ".join(f"{ids[id(b)]} ({ROLE_SHORT.get(a.rows[b.row].role, '')})" for b in st["bands"])
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(boxes))
            ok = QtWidgets.QTableWidgetItem("✓" if not st["problems"] else "⚠ " + ", ".join(st["problems"]))
            ok.setForeground(QtGui.QColor("#2e7d32" if not st["problems"] else "#c62828"))
            t.setItem(i, 2, ok)
            for c in (1, 2):
                t.item(i, c).setFlags(t.item(i, c).flags() & ~Qt.ItemIsEditable)
        t.blockSignals(False)
        t.resizeColumnToContents(2)
        self.group_hint.setText(
            "박스를 고르고 우클릭 → '측정 묶음에 넣기'로 그룹을 만드세요. 그룹마다 Control 1개 + Target 박스를 넣습니다."
            if not rows else f"그룹 {len(rows)}개 · 그룹 이름은 표에서 더블클릭해 바꿉니다 · 줄을 고르면 그 박스들이 선택됩니다")

    def on_group_row_selected(self):
        sname = self.current_set()
        rows = sorted({i.row() for i in self.group_table.selectedIndexes()})
        if not sname or not rows:
            return
        g = core.grouping(self.analysis, sname)
        mem = [b for r in rows if r < len(g["groups"]) for b in core.group_members(self.analysis, sname, g["groups"][r])]
        self.view.select_bands(mem)

    def on_group_renamed(self, item):
        sname = self.current_set()
        g = core.grouping(self.analysis, sname) if sname else None
        if g is None or item.column() != 0 or item.row() >= len(g["groups"]):
            return
        old = g["groups"][item.row()]
        new = core.rename_group(self.analysis, sname, old, item.text().strip())
        self.groups_changed(show=sname)
        if new != item.text().strip():
            self.statusBar().showMessage(f"이미 있는 이름이라 '{new}'(으)로 붙였습니다")

    def new_set(self):
        name = self._ask_name("새 묶음 세트", "묶음 세트 이름 (예: 샘플 짝, 시간별):", f"묶음 {len(self.analysis.groupings) + 1}")
        if name:
            self.groups_changed(show=core.add_grouping(self.analysis, name))

    def rename_set(self):
        sname = self.current_set()
        if not sname:
            return
        name = self._ask_name("묶음 세트 이름 바꾸기", "새 이름:", sname)
        if name:
            new = core.rename_grouping(self.analysis, sname, name)
            for d in self.docs:
                if d.basis == sname:
                    d.basis = new
                    d.basis_ref[new] = d.basis_ref.pop(sname, 0)
            self.groups_changed(show=new)

    def delete_set(self):
        sname = self.current_set()
        if not sname:
            return
        if QtWidgets.QMessageBox.question(self, "묶음 세트 삭제", f"'{sname}' 세트와 그 그룹을 모두 지울까요? (박스와 레인은 그대로입니다)") \
                != QtWidgets.QMessageBox.Yes:
            return
        core.delete_grouping(self.analysis, sname)
        if self.doc.basis == sname:
            self.doc.basis = LANES_BASIS
        self.groups_changed()

    def delete_groups(self):
        sname = self.current_set()
        g = core.grouping(self.analysis, sname) if sname else None
        rows = sorted({i.row() for i in self.group_table.selectedIndexes()})
        if g is None or not rows:
            self.statusBar().showMessage("지울 그룹을 표에서 고르세요")
            return
        names = [g["groups"][r] for r in rows if r < len(g["groups"])]
        core.clear_group(self.analysis, [b for n in names for b in core.group_members(self.analysis, sname, n)], sname)
        self.groups_changed(show=sname)

    def move_group(self, d):
        sname = self.current_set()
        g = core.grouping(self.analysis, sname) if sname else None
        rows = sorted({i.row() for i in self.group_table.selectedIndexes()})
        if g is None or len(rows) != 1:
            return
        i, j = rows[0], rows[0] + d
        if 0 <= j < len(g["groups"]):
            g["groups"][i], g["groups"][j] = g["groups"][j], g["groups"][i]
            self.groups_changed(show=sname)
            self.group_table.selectRow(j)

    # ---------------------------------------------------------------- results basis (lanes or a grouping set)
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
    def set_row_role(self, i, role):
        if i >= len(self.analysis.rows):
            return
        self.analysis.rows[i].role = role
        self.view.refresh_items()
        self.recompute_results()
        self.recompute_multi()

    def set_row_size(self, i, attr, v):
        setattr(self.analysis.rows[i], attr, int(v))
        self.view.refresh_items()
        self.apply_auto_lanes()
        self.remeasure()

    def delete_row(self):
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

    def delete_lane(self):
        lanes = sorted({b.lane for b in self.view.selected_bands()}, reverse=True)
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

    def _edit_formula_dialog(self, title, terms, variables, formula):
        dlg = FormulaDialog(self, title, terms, variables, formula)
        return dlg.result() if dlg.exec() == QtWidgets.QDialog.Accepted else None

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
        self.update_sel_card()
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
        return {k: [m.get((k, lab), lab) for lab in self.basis_of(d)[0].lane_labels]
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
            top = QtWidgets.QTreeWidgetItem([self.doc_label(k), ""])
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
            src = f" [{d.project}]" if d.project and self._n_projects() > 1 else ""
            terms.append((f"{key}{src} 단백질 (레인별 값)", [(f"{key}: {r.name}", f"{key}.{core._safe_name(r.name)}") for r in d.analysis.rows]))
            ids = sorted({m.band_id for m in d.measurements})
            terms.append((f"{key}{src} 밴드 (값 하나)", [(f"{key}: {i}", f"{key}.{i}") for i in ids]))
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
                a, meas = self.basis_of(d)
                for ri, r in enumerate(a.rows):
                    v[core._safe_name(r.name)] = core.to_common(core.row_vector(a, meas, ri), keys[k], common)
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
                a, meas = self.basis_of(d)
                ri = next((i for i, r in enumerate(a.rows) if r.name == name), None)
                if ri is None:
                    msgs.append(f"{g['name']}: {self.doc_key(k)}에 '{name}' 단백질 그룹이 없습니다")
                    continue
                if g.get("value") == "ratio":
                    ctrl = next((i for i, r in enumerate(a.rows) if r.role == core.ROLE_CONTROL), None)
                    if ctrl is None or ctrl == ri:
                        msgs.append(f"{g['name']}: {self.doc_key(k)}에 Control 단백질이 없어 비율을 계산할 수 없습니다")
                        continue
                    vec = core.normalize_to_control(a, meas, ri, ctrl, None)
                else:
                    vec = core.row_vector(a, meas, ri)
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
        self.multi_msg.setStyleSheet(f"color: {_tok('red')};" if msgs else f"color: {_tok('label-secondary')};")
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

    def project_dict(self, project_path: Optional[str] = None, arc: Optional[Dict[str, str]] = None) -> dict:
        """Everything needed to restore the session. ``arc`` = embedded files {source: archive name}."""
        pp = project_path or self.project_path or os.path.join(os.getcwd(), "project.blotproj")
        arc = arc or {}
        images = []
        for d in self.docs:
            if d.raw is None:
                continue
            if d.figure is None:  # the figure tab would create these; saved and reopened states stay equal
                d.figure = figure.apply_mode_defaults(figure.default_settings(), d.analysis, d.mode)
            figure.sync_settings(d.figure, d.analysis, d.raw.shape)
            marker = (d.figure or {}).get("marker_path") if d.marker is not None else None
            images.append({
                "image_path": d.image_path,  # older versions read this
                "image": projfile.file_ref(d.image_path, pp, arc.get(d.image_path)),
                "marker": projfile.file_ref(marker, pp, arc.get(marker)) if marker else None,
                "invert": d.invert, "background": vars(d.bg), "reference_lane": d.ref_lane,
                "basis": d.basis, "basis_reference": d.basis_ref,
                "mode": d.mode, "transform": d.transform, "marker_tf": d.marker_tf,
                "analysis": d.analysis.to_dict(), "figure": d.figure,
            })
            if self._n_projects() > 1:  # several projects combined: remember where each image came from
                images[-1]["project"] = d.project
        return {
            "app": "Blot Profiler",
            "version": __version__,
            "format": 4,
            "detect": {
                "sensitivity": self.sens.value(), "smooth": self.smooth.value(), "min_area": self.min_area.value(),
                "pad": self.pad.value(), "edge": self.edge.value(), "lut_gamma": self.lut_gamma.value(),
                "fill_lanes": self.fill_lanes.isChecked(), "snap_lanes": self.snap_lanes.isChecked(),
                "method": self.method_combo.currentData(),
                "auto_tilt": self.auto_tilt.isChecked(),
            },
            "images": images,
            "big_groups": self.big_groups,
            "multi_formulas": self.multi_formulas,
            "multi_reference": self.multi_ref,
            "ui": {
                "current_image": self.cur, "auto_norm": self.auto_norm.isChecked(),
                "by_condition": self.by_condition.isChecked(), "show_points": self.show_points.isChecked(),
                "multi_points": self.multi_points.isChecked(), "auto_lanes": self.auto_lanes.isChecked(),
                "show_marker": self.show_marker.isChecked(), "ask_marker": self.ask_marker.isChecked(),
                "auto_gel_crop": self.auto_gel_crop.isChecked(), "graph": dict(self.graph_opts),
                "quant": {"ref_name": self.ref_name, "ctrl_name": self.ctrl_name, "formulas": self.formulas_all,
                          "series_on": self.series_on, "excluded": [k for k, d in enumerate(self.docs)
                                                                    if not getattr(d, "in_quant", True)]},
                "detect_only_on_lines": self.only_on_lines.isChecked(),
            },
        }

    # ---------------------------------------------------------------- unsaved changes
    def _fingerprint(self) -> str:
        d = self.project_dict(project_path="/project.blotproj")
        d.pop("ui", None)
        for im in d["images"]:
            im.pop("image_path", None)
        return json.dumps(d, sort_keys=True, default=str)

    def mark_saved(self):
        self._saved_fp = self._fingerprint()
        self.update_title()

    def is_dirty(self) -> bool:
        if not self.loaded_docs():
            return False
        return self._fingerprint() != self._saved_fp

    def update_title(self):
        if self.project_path:
            name = os.path.basename(self.project_path)
        elif self.doc.raw is not None:
            name = os.path.basename(self.doc.image_path)
        else:
            self.setWindowTitle("Blot Profiler")
            return
        dirty = self._saved_fp is None or self.is_dirty()
        self.setWindowTitle(f"{'● ' if dirty else ''}Blot Profiler – {name}{' (저장 안 됨)' if dirty else ''}")

    def maybe_save(self) -> bool:
        """Ask to save unsaved changes. False = the user cancelled."""
        if not self.is_dirty():
            return True
        r = QtWidgets.QMessageBox.question(
            self, "저장하지 않은 변경", "저장하지 않은 변경이 있습니다. 프로젝트를 저장할까요?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save)
        if r == QtWidgets.QMessageBox.Cancel:
            return False
        if r == QtWidgets.QMessageBox.Save:
            return self.save_project()
        return True

    def closeEvent(self, e):
        if self.maybe_save():
            e.accept()
        else:
            e.ignore()

    # ---------------------------------------------------------------- save / open project
    def save_project(self, save_as: bool = False) -> bool:
        if not self.loaded_docs():
            self.statusBar().showMessage("저장할 이미지가 없습니다")
            return False
        p = self.project_path
        if save_as or not p:
            base = p or (os.path.splitext(self.doc.image_path)[0] + projfile.PROJECT_EXT)
            p, flt = QtWidgets.QFileDialog.getSaveFileName(
                self, "프로젝트 저장", base,
                "Blot Profiler 프로젝트 — 이미지 포함 (*.blotproj);;JSON — 이미지 미포함, 예전 방식 (*.json)")
            if not p:
                return False
            if not os.path.splitext(p)[1]:
                p += ".json" if flt.startswith("JSON") else projfile.PROJECT_EXT
        embed = p.lower().endswith(projfile.PROJECT_EXT)
        if embed:
            files = self._project_files()
            size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
            if size > projfile.EMBED_WARN_BYTES:
                r = QtWidgets.QMessageBox.question(
                    self, "이미지 포함", f"이미지가 {size / 1e6:.0f} MB입니다. 프로젝트 파일에 이미지를 포함할까요?\n"
                    "(포함하면 다른 컴퓨터에서도 바로 열리고, 포함하지 않으면 원본 파일 위치를 기억합니다)")
                embed = r == QtWidgets.QMessageBox.Yes
        try:
            self.write_project(p, embed)
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "저장 실패", str(e))
            return False
        self.statusBar().showMessage(f"프로젝트를 저장했습니다: {p}" + (" (이미지 포함)" if embed and p.lower().endswith(projfile.PROJECT_EXT) else ""))
        return True

    def _project_files(self):
        files = []
        for d in self.loaded_docs():
            files.append(d.image_path)
            mp = (d.figure or {}).get("marker_path")
            if d.marker is not None and mp and os.path.exists(mp):
                files.append(mp)
        return [f for f in files if f and os.path.exists(f)]

    def write_project(self, path: str, embed: bool = True):
        """Save everything to ``path`` (.blotproj = zip, with the images when ``embed``)."""
        arc = projfile.archive_names(self._project_files()) if embed and path.lower().endswith(projfile.PROJECT_EXT) else {}
        projfile.write(path, self.project_dict(project_path=path, arc=arc), arc)
        self.project_path = path
        self.mark_saved()

    def open_project_dialog(self):
        if not self.maybe_save():
            return
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "프로젝트 열기 (여러 개 선택 가능)", "",
                                                       "Blot Profiler 프로젝트 (*.blotproj *.json);;All files (*)")
        self.open_projects(ps, ask_save=False)

    def add_project_dialog(self):
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "프로젝트 추가로 불러오기 (여러 개 선택 가능)", "",
                                                       "Blot Profiler 프로젝트 (*.blotproj *.json);;All files (*)")
        for p in ps:
            self.load_project(p, ask_save=False, append=True)

    def open_projects(self, paths, ask_save: bool = True):
        """Open the first project and add the others to it (their images are compared in tab 5)."""
        paths = [p for p in paths if p]
        if not paths:
            return
        if ask_save and not self.maybe_save():
            return
        self.load_project(paths[0], ask_save=False)
        for p in paths[1:]:
            self.load_project(p, ask_save=False, append=True)

    def _relocate(self, what: str, ref, project_path: str, embedded: Optional[str], optional: bool = False) -> Optional[str]:
        path = projfile.resolve(ref, project_path, embedded)
        if path:
            return path
        if isinstance(ref, str):
            ref = {"path": ref, "name": os.path.basename(ref)}
        orig = (ref or {}).get("path") or (ref or {}).get("name") or "?"
        buttons = QtWidgets.QMessageBox.Open | (QtWidgets.QMessageBox.Ignore if optional else QtWidgets.QMessageBox.Cancel)
        r = QtWidgets.QMessageBox.question(
            self, "파일 다시 찾기", f"{what}을(를) 찾을 수 없습니다:\n{orig}\n\n파일을 직접 찾을까요?", buttons, QtWidgets.QMessageBox.Open)
        if r != QtWidgets.QMessageBox.Open:
            return None
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"{what} 다시 찾기 — {(ref or {}).get('name', '')}",
                                                     os.path.dirname(project_path), "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        return p or None

    def _ask_replace(self) -> bool:
        """Dropping a project while images are open: replace them, or add the project?"""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("프로젝트 열기")
        box.setText("지금 열린 작업을 이 프로젝트로 바꿀까요, 아니면 지금 작업에 추가할까요?\n"
                    "(추가하면 이미지가 함께 열려 5번 탭에서 수식으로 비교할 수 있습니다)")
        add = box.addButton("추가하기", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("바꾸기", QtWidgets.QMessageBox.DestructiveRole)
        box.exec()
        return box.clickedButton() is not add

    def load_project(self, p=None, ask_save: bool = True, append: bool = False):
        """Open project ``p``; with ``append`` its images are added to the ones already open
        (image references I1, I2 … in its formulas and big groups are renumbered)."""
        if ask_save and not append and not self.maybe_save():
            return
        if not p:
            self.open_project_dialog()
            return
        try:
            d, embedded = projfile.read(p)
            if d.get("app") == self.ROI_APP:  # a band ROI file, not a project
                self.load_roi(p)
                return
            if "images" in d:
                images = d["images"]
            else:  # format 1: a single image
                images = [{k: d.get(k) for k in ("image_path", "invert", "background", "reference_lane", "analysis")}]
            det = d.get("detect", {})
            proj_name = os.path.splitext(os.path.basename(p))[0]
            if append:
                return self._append_project(p, d, embedded, images, proj_name)
            self.sens.setValue(det.get("sensitivity", 1.0))
            self.smooth.setValue(det.get("smooth", 1.5))
            self.min_area.setValue(det.get("min_area", 30))
            self.pad.setValue(det.get("pad", 3))
            self.edge.setValue(det.get("edge", 0))
            self.lut_gamma.setValue(det.get("lut_gamma", 0.6))
            self.fill_lanes.setChecked(det.get("fill_lanes", True))
            self.snap_lanes.setChecked(det.get("snap_lanes", True))
            self.method_combo.setCurrentIndex(max(0, self.method_combo.findData(det.get("method", "legacy"))))
            self.auto_tilt.setChecked(bool(det.get("auto_tilt", True)))
            docs, index = self._project_docs(p, d, embedded, images, proj_name)
            if not docs:
                return
            self.docs = docs
            self.big_groups = [
                dict(g, members=[[index[int(k)], n] for k, n in g.get("members", []) if int(k) in index])
                for g in d.get("big_groups", [])
            ]
            self.multi_formulas = [dict(f) for f in d.get("multi_formulas", [])]
            self.multi_ref = d.get("multi_reference")
            ui = d.get("ui", {})
            for w, key in ((self.auto_norm, "auto_norm"), (self.by_condition, "by_condition"), (self.show_points, "show_points"),
                           (self.multi_points, "multi_points"), (self.auto_lanes, "auto_lanes"),
                           (self.show_marker, "show_marker"), (self.ask_marker, "ask_marker"),
                           (self.auto_gel_crop, "auto_gel_crop")):
                if key in ui:
                    w.blockSignals(True)
                    w.setChecked(bool(ui[key]))
                    w.blockSignals(False)
            q = ui.get("quant") or {}
            self.formulas_all = [dict(f) for f in q.get("formulas", [])]
            self.series_on = dict(q.get("series_on", {}))
            self.ctrl_name = q.get("ctrl_name")
            self.ref_name = q.get("ref_name")
            self._ref_default = "ref_name" not in q
            if self._ref_default and docs:  # older projects: the first image's reference lane
                d0 = docs[0]
                r = d0.ref_lane
                if r is not None and r < len(d0.analysis.lane_labels):
                    self.ref_name, self._ref_default = d0.analysis.lane_labels[r], False
            for k in q.get("excluded", []):
                if k < len(docs):
                    docs[k].in_quant = False
            self.only_on_lines.setChecked(bool(ui.get("detect_only_on_lines", False)))
            self.set_graph_opts(ui.get("graph") or {})
            self.cur = min(int(ui.get("current_image", 0)), len(docs) - 1)
            self.refresh_doc_tabs()
            self.fill_multi_formulas()
            self.switch_doc(self.cur)
            self.figure_editor.dirty = True
            self.project_path = p
            self.mark_saved()
            self.statusBar().showMessage(f"프로젝트를 열었습니다: {p} (이미지 {len(docs)}개)")
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "열기 실패", str(e))

    def _append_project(self, p, d, embedded, images, proj_name):
        docs, index = self._project_docs(p, d, embedded, images, proj_name)
        if not docs:
            return
        self.docs = [x for x in self.docs if x.raw is not None]  # drop an empty first tab
        off = len(self.docs)
        index = {i: k + off for i, k in index.items()}
        self.docs.extend(docs)
        # names of big groups / image formulas must stay unique; image references move by ``off``
        renames = {}
        for g in d.get("big_groups", []):
            name = core.unique_name(g["name"], [x["name"] for x in self.big_groups])
            if name != g["name"]:
                renames[core._safe_name(g["name"])] = core._safe_name(name)
            self.big_groups.append(dict(g, name=name, members=[[index[int(k)], n] for k, n in g.get("members", [])
                                                               if int(k) in index]))

        def shift(expr):
            def img(m):
                k = index.get(int(m.group(1)) - 1)
                return f"I{k + 1}." if k is not None else m.group(0)
            expr = re.sub(r"\bI(\d+)\.", img, expr)
            for old, new in renames.items():
                expr = re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w])", new, expr)
            return expr

        for f in d.get("multi_formulas", []):
            name = core.unique_name(f.get("name", "수식"), [x.get("name") for x in self.multi_formulas])
            self.multi_formulas.append(dict(f, name=name, expr=shift(f.get("expr", ""))))
        self.project_path = None  # several projects together: saving asks for a new file
        self.refresh_doc_tabs()
        self.fill_multi_formulas()
        self.switch_doc(off)
        self.refresh_multi()
        self.figure_editor.dirty = True
        self.statusBar().showMessage(
            f"프로젝트 '{proj_name}'를 추가했습니다: 이미지 {len(docs)}개 → {self.doc_key(off)}"
            + (f"~{self.doc_key(off + len(docs) - 1)}" if len(docs) > 1 else "")
            + ". 5번 탭에서 이미지(I1, I2 …) 값으로 수식을 만들 수 있습니다. 저장하면 합친 프로젝트를 새 파일로 저장합니다.")

    def _project_docs(self, p, d, embedded, images, proj_name):
        """ImageDocs of a project file; also {index in the file: index in the list}."""
        docs, index = [], {}
        for i, im in enumerate(images):
            path = self._relocate("블롯 이미지", im.get("image") or im.get("image_path"), p, embedded)
            if not path:
                continue
            doc = ImageDoc(invert=int(im.get("invert") or 0), bg=core.BackgroundSettings(**(im.get("background") or {})))
            doc.project = im.get("project") or proj_name
            doc.raw_orig = core.load_image(path)
            doc.transform = dict(core.default_transform(), **(im.get("transform") or {}))
            doc.raw = core.apply_transform(doc.raw_orig, doc.transform)
            doc.mode = im.get("mode", core.MODE_WESTERN)
            doc.image_path = path
            doc.ref_lane = im.get("reference_lane", 0)
            doc.basis = im.get("basis", LANES_BASIS) or LANES_BASIS
            doc.basis_ref = dict(im.get("basis_reference") or {})
            doc.analysis = core.Analysis.from_dict(im.get("analysis") or {})
            doc.analysis.ensure_lane_labels()
            doc.figure = im.get("figure") or figure.default_settings()
            figure.sync_settings(doc.figure, doc.analysis, doc.raw.shape)  # as the figure tab would
            self._make_signal(doc)
            self.measure_doc(doc)
            mref = im.get("marker") or ((doc.figure or {}).get("marker_path") if doc.figure else None)
            if mref:
                mpath = self._relocate("마커 이미지", mref, p, embedded, optional=True)
                if mpath:
                    self.set_marker(doc, mpath, quiet=True, marker_tf=im.get("marker_tf"))
                elif doc.figure:
                    doc.figure["marker_path"] = None
            index[i] = len(docs)
            docs.append(doc)
        # format 2 kept formulas for all images in one list, each with a base image
        for fml in d.get("formulas", []):
            k = index.get(int(fml.get("image", 0)))
            if k is not None:
                docs[k].analysis.formulas.append({x: fml[x] for x in ("name", "expr", "normalize") if x in fml})
        return docs, index

    def export_csv(self):
        if not any(d.measurements for d in self.loaded_docs()):
            self.statusBar().showMessage("내보낼 결과가 없습니다")
            return
        p = self._save_path("전체 결과 CSV 내보내기", "results.csv", "CSV (*.csv)")
        if p:
            self.write_results_csv(p)
            self.statusBar().showMessage(f"저장했습니다: {p}")

    def write_results_csv(self, p: str):
        """Every band and every result of all open images (the full table)."""
        docs = self.loaded_docs()
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
                        "raw_intensity", "background", "net_intensity", "mean_per_px", "measurement_groups"])
            for d in docs:
                a = d.analysis
                k = self.doc_key(self.docs.index(d))
                ids = a.band_ids()
                by_id = {ids[id(b)]: b for b in a.bands}
                for m in d.measurements:
                    x, y, wd, ht = by_id[m.band_id].box(a.rows)
                    r = a.rows[m.row]
                    w.writerow([k, m.band_id, r.name, r.role, m.lane + 1, a.lane_labels[m.lane],
                                x, y, wd, ht, f"{m.raw:.3f}", f"{m.background:.3f}", f"{m.net:.3f}", f"{m.mean:.4f}",
                                "; ".join(f"{s_}: {g_}" for s_, g_ in by_id[m.band_id].tags.items())])
            w.writerow([])
            w.writerow(["[Results per image]"])
            w.writerow(["image", "series", "reference", "lane_or_group", "lane_group_or_group_name", "value", "pairing"])
            cond_rows = []
            for d in docs:
                a, _ = self.basis_of(d)
                k = self.doc_key(self.docs.index(d))
                r_ = self.ref_of(d)
                ref = "" if r_ is None else r_ + 1
                pairing = f"grouping set: {d.basis}" if d.basis else "lanes"
                res, _ = self.image_results(d)
                for n, v in res.items():
                    if v.size == 1:
                        w.writerow([k, n, ref, "", "", f"{v[0]:.6g}", pairing])
                        continue
                    for i in range(v.size):
                        w.writerow([k, n, ref, i + 1, a.lane_labels[i] if i < len(a.lane_labels) else "",
                                    f"{v[i]:.6g}" if np.isfinite(v[i]) else "", pairing])
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

    # ---------------------------------------------------------------- exporting (toolbar, File → 내보내기)
    def _base_name(self) -> str:
        return os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "blot"

    def _save_path(self, title: str, suffix: str, filters: str) -> Optional[str]:
        start = os.path.join(prefs.export_folder(self.prefs, self.image_path or ""), f"{self._base_name()}_{suffix}")
        p, flt = QtWidgets.QFileDialog.getSaveFileName(self, title, start, filters)
        if not p:
            return None
        if not os.path.splitext(p)[1]:
            p += "." + flt.split("*.")[1].split()[0].rstrip(")") if "*." in flt else ""
        self.remember_folder(os.path.dirname(p))
        return p

    def remember_folder(self, folder: str):
        self.prefs["export"]["last_folder"] = folder
        try:
            prefs.save(self.prefs)
        except OSError:
            pass

    def _image_filter(self, first: Optional[str] = None) -> str:
        order = [first or self.prefs["export"].get("image_format", "png")] + [k for k, _ in prefs.IMAGE_FORMATS]
        seen, out = set(), []
        for k in order:
            if k not in seen:
                seen.add(k)
                out.append(f"{dict(prefs.IMAGE_FORMATS)[k]} (*.{k})")
        return ";;".join(out)

    @staticmethod
    def save_figure_file(fig, p: str, dpi: int, tight: bool = True):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if p.lower().endswith((".tif", ".tiff")) else {}
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight" if tight else None, **kw)

    def _graph_canvas(self, scope: str):
        return self.multi_plot if scope == "multi" else self.plot

    def graph_scope(self) -> str:
        return "multi" if getattr(self, "quant_win", None) is not None and self.quant_win.page() == "multi" else "current"

    def graph_data(self, scope: str = "current"):
        """Only the numbers drawn in the graph: (header, rows, comment lines). One row per value
        (이미지, 항목, 그룹, 레인, 값), then mean / SD (SEM) / n per group (spec §9).
        scope = 'current' (the images ticked in the quantification window), 'all' (every open
        image) or 'multi' (여러 이미지)."""
        err = self.graph_opts.get("err", "sd")
        if scope == "multi":
            series = {k: self.multi_results[k] for k in self._checked(self.multi_series) if k in self.multi_results}
            header, rows = graph.summary_table(self.multi_keys, series, err)
            notes = [f"# 여러 이미지 비교 · 기준 레인: {self.multi_ref or '없음'}"]
            return header, rows, notes
        if scope == "all":
            pooled = []
            for d in self.loaded_docs():
                if d.measurements:
                    res, _ = self.image_results(d)
                    pooled.append((self.doc_key(self.docs.index(d)), res, list(self.basis_of(d)[0].lane_labels)))
        else:
            pooled = getattr(self, "_pooled", None) or self.quant_pooled()[0]
        shown = set(self.shown_series()) or {n for _, res, _ in pooled for n in res}
        by_cond = bool(self.graph_opts.get("by_condition", True))
        header = ["이미지", "항목", "그룹", "레인", "값"]
        rows, groups = [], {}
        for key, res, labels in pooled:
            for name, v in res.items():
                if name not in shown or np.size(v) <= 1:
                    continue
                for i, val in enumerate(np.atleast_1d(v)):
                    lab = labels[i] if i < len(labels) else str(i + 1)
                    g = lab if by_cond else f"{key} {lab}"
                    rows.append([key, name, lab, f"L{i + 1}", f"{val:.6g}" if np.isfinite(val) else ""])
                    if np.isfinite(val):
                        groups.setdefault((name, g), []).append(float(val))
        lab_err = "SEM" if err == "sem" else "SD"
        for (name, g), vals in groups.items():
            m_, e_, n_ = graph.summary(vals, err)
            rows.append(["", name, g, "평균", f"{m_:.6g}"])
            rows.append(["", name, g, lab_err, f"{e_:.6g}" if n_ > 1 else ""])
            rows.append(["", name, g, "n", str(n_)])
        imgs = ", ".join(f"{k} {self.docs[int(k[1:]) - 1].name}" for k, _, _ in pooled)
        ctrl = self.ctrl_name if self.ctrl_name else next((r.name for r in self.analysis.rows if r.role == core.ROLE_CONTROL), "없음")
        basis = f"묶음 세트 {self.doc.basis}" if self.doc.basis else "레인 기준"
        notes = [f"# 이미지: {imgs}", f"# Control: {ctrl} ({basis})", f"# 1.0 기준: {self.ref_name or '나누지 않음'}"]
        return header, rows, notes

    def write_graph_csv(self, p: str, scope: str = "current"):
        header, rows, notes = self.graph_data(scope)
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([f"# Blot Profiler {__version__} · 그래프에 쓰인 데이터"])
            for n in notes:
                w.writerow([n])
            w.writerow(header)
            w.writerows(rows)

    def new_project(self):
        if not self.maybe_save():
            return
        self.docs = [ImageDoc()]
        self.cur = 0
        self.big_groups, self.multi_formulas, self.formulas_all = [], [], []
        self.ref_name, self.ctrl_name, self._ref_default = None, None, True
        self.project_path = None
        self.refresh_doc_tabs()
        self.switch_doc(0)
        self.mark_saved()
        self.statusBar().showMessage("새 프로젝트")

    def open_any(self):
        """열기…: images and projects (several at once)."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "열기 (이미지 · 프로젝트, 여러 개 가능)", "",
            "이미지·프로젝트 (*.tif *.tiff *.png *.jpg *.jpeg *.blotproj *.json);;All files (*)")
        self.open_paths(paths)

    def export_graph_csv(self):
        scope = self.graph_scope()
        p = self._save_path("그래프 데이터 CSV", "graph_data.csv", "CSV (*.csv)")
        if p:
            self.write_graph_csv(p, scope)
            self.statusBar().showMessage(f"그래프에 쓰인 데이터를 저장했습니다: {p}")

    def export_plot(self, canvas=None, fmt: Optional[str] = None):
        canvas = canvas or self._graph_canvas(self.graph_scope())
        fmt = fmt or self.prefs['export'].get('image_format', 'png')
        p = self._save_path("그래프 저장", f"graph.{fmt}", self._image_filter(fmt))
        if p:
            self.write_graph_image(p, canvas)
            self.statusBar().showMessage(f"그래프 저장: {p}")

    def write_graph_image(self, p: str, canvas=None, dpi: Optional[int] = None):
        """The graph at its set size (그림 크기, mm)."""
        canvas = canvas or self.plot
        fig = canvas.fig
        old = fig.get_size_inches().copy()
        w, h = self.graph_opts.get("size_mm") or [90, 60]
        fig.set_size_inches(w / 25.4, h / 25.4)
        if canvas is self.plot:
            self.replot()
        else:
            self.replot_multi()
        fig.tight_layout()
        try:
            self.save_figure_file(fig, p, int(dpi or self.prefs["figure"].get("export_dpi", 600)))
        finally:
            fig.set_size_inches(*old)
            canvas.canvas.draw_idle()

    def export_graph_both(self):
        p = self._save_path("그래프 데이터 + 이미지", "graph_data.csv", "CSV (*.csv)")
        if p:
            self.write_graph_csv(p, self.graph_scope())
            img = os.path.splitext(p)[0].replace("_graph_data", "") + f"_graph.{self.prefs['export'].get('image_format', 'png')}"
            self.write_graph_image(img, self._graph_canvas(self.graph_scope()))
            self.statusBar().showMessage(f"저장했습니다: {p}, {img}")

    def _figure_ready(self) -> bool:
        ed = self.figure_editor
        if ed.doc is not self.doc or ed.dirty:
            ed.set_doc(self.doc)
        if not ed.ready():
            self.statusBar().showMessage("Figure를 만들 이미지가 없습니다 (밴드를 먼저 검출하세요)")
            return False
        return True

    def export_figure_numbers(self):
        if not self._figure_ready():
            return
        p = self._save_path("Figure 숫자 CSV", "figure_numbers.csv", "CSV (*.csv)")
        if p:
            self.figure_editor.write_numbers_csv(p)
            self.statusBar().showMessage(f"Figure에 들어간 숫자를 저장했습니다: {p}")

    def open_export_dialog(self):
        dlg = ExportDialog(self)
        dlg.exec()

    def open_settings(self):
        dlg = SettingsDialog(self, self.prefs)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.prefs = dlg.result_prefs()
            prefs.save(self.prefs)
            figure.set_user_defaults(self.prefs["figure"])
            self.set_graph_opts({k: self.prefs["graph"][k] for k in ("kind", "err", "colors", "y0")})
            if dlg.apply_figure:
                for key in ("font", "label_size", "tick_size", "legend_size", "title_size", "bold", "tick_len", "tick_dir"):
                    self.graph_opts.pop(key, None)  # the graph takes the new defaults too
                self.replot()
                for d in self.docs:
                    if d.figure is not None:
                        figure.apply_look(d.figure)
                self._figure_changed(self.doc)
            self.statusBar().showMessage("설정을 저장했습니다 — 새 그래프·Figure에 적용됩니다"
                                         + (" (열린 Figure에도 적용)" if dlg.apply_figure else ""))

    def export_overlay(self):
        if self.view.pix_item is None:
            return
        base = os.path.splitext(self.image_path)[0] + "_boxes.png" if self.image_path else "boxes.png"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "박스 표시 이미지 저장", base, "PNG (*.png)")
        if not p:
            return
        self.write_overlay(p)

    def write_overlay(self, p: str):
        """The blot with its boxes (to check the detection)."""
        if self.view.pix_item is None:
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

HELP_TEXT = """Blot Profiler 1.0 사용 순서

메인 창: 왼쪽 사이드바(이미지 · 단계 · 결과 창) | 가운데 이미지 | 오른쪽 속성창.

1. 검출: 이미지를 열면(파일 › 열기 또는 끌어다 놓기) 레인과 박스를 자동으로 찾습니다.
   · 위쪽 도구: 선택 / 레인 선 / 단백질 선 / 마커 영역
   · 레인 선(파랑): 빈 곳 클릭 = 새 선, 선 더블클릭 = 점 추가, 점 끌기 = 휘기, ⌫ = 지우기, 우클릭 = 메뉴
   · 단백질 선(빨간 점선): 가로로 끌어 추가, 양끝 네모로 기울이기, 오른쪽 카드에서 어느 단백질인지 고름
   · '선 기준으로 다시 검출': 그린 선을 기준으로 박스를 다시 찾습니다(빨간 선 위에서만 찾기 가능)
   · 마커: 마커 영역을 끌어 정하고 영역 안을 클릭해 밴드 위치를 넣은 뒤, 첫 마커 크기 하나만 넣으면
     나머지는 래더 순서로 채워집니다(칩을 눌러 고침).

2. 레인 / 단백질: 레인 이름과 역할(시료/마커/제외), 단백질 이름과 역할(Control/Target).
   '＋ 새 단백질'로 박스 하나를 그리고 'ROI 박스 채우기'로 모든 레인에 같은 크기로 채웁니다.
   박스: 클릭·드래그 선택, 끌어서 이동, 더블클릭 크기 수정, ⌫ 삭제, 우클릭 그룹·측정 묶음·병합.

정량·그래프 창: 밴드 값 / 정량 / 여러 이미지. 이미지를 체크해 함께 그리고, Control 짝짓기(레인 또는
측정 묶음 세트), Control 단백질, 1.0 기준을 고릅니다. '＋ 수식 추가'로 사이드바에서 수식을 만듭니다.
그래프는 PRISM 스타일이고 오른쪽 편집기에서 모양·시리즈·축·글꼴·범례를 바꿉니다.

Figure 창: 레인별 자르기 / Full blot, 왼쪽 추가 도구, 오른쪽 속성, 더블클릭으로 편집.

[ 확대   ] 축소   Space + 드래그: 화면 이동.  저장은 모두 파일 메뉴(내보내기… ⌘E, 설정… ⌘,).
정량값(Net) = 박스 안 원본(raw) 픽셀 intensity 합 − 배경.
"""


class App(QtWidgets.QApplication):
    """Handles files opened from Finder ("Open With" / dropping on the Dock icon)."""

    def __init__(self, argv):
        super().__init__(argv)
        theme.apply(self)  # Apple HIG design system (desktop profile), see ui/theme.py
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
        w.open_paths([args[0]])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
