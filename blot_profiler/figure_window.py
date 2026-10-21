"""'Figure 만들기' editor (a main-window tab): edit and export a publication-style blot figure."""

from __future__ import annotations

import io
import os
from typing import List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from matplotlib.figure import Figure

from . import core, figure

FONTS = ["Arial", "Helvetica", "Times New Roman", "AppleGothic"]
PREVIEW_DPI = 200


def _gray_qimage(a01: np.ndarray) -> QtGui.QImage:
    a = np.ascontiguousarray((np.clip(a01, 0, 1) * 255).astype(np.uint8))
    h, w = a.shape
    return QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()


def _spin(lo, hi, val, step=1.0, decimals=None, suffix=""):
    w = QtWidgets.QSpinBox() if decimals is None else QtWidgets.QDoubleSpinBox()
    if decimals is not None:
        w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".json")


def _dropped_paths(e) -> List[str]:
    md = e.mimeData()
    if not md.hasUrls():
        return []
    return [u.toLocalFile() for u in md.urls() if u.toLocalFile().lower().endswith(IMAGE_EXT)]


class ReferenceView(QtWidgets.QGraphicsView):
    """Marker image and blot side by side; click to add a kDa mark at that height."""

    clicked = QtCore.Signal(float)  # y in blot pixels
    markerDropped = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QtGui.QColor("#d9d9d9"))
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.cursor_line: Optional[QtWidgets.QGraphicsLineItem] = None
        self.width_px = 0

    def wheelEvent(self, e):
        if e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            f = 1.0015 ** e.angleDelta().y()
            self.scale(f, f)
        else:
            super().wheelEvent(e)

    def mouseMoveEvent(self, e):
        if self.cursor_line is not None:
            y = self.mapToScene(e.position().toPoint()).y()
            self.cursor_line.setLine(0, y, self.width_px, y)
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.mapToScene(e.position().toPoint()).y())
            return
        super().mousePressEvent(e)

    def dragEnterEvent(self, e):
        if _dropped_paths(e):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if _dropped_paths(e):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = _dropped_paths(e)
        if paths:
            e.acceptProposedAction()
            self.markerDropped.emit(paths[0])

    def fit(self):
        r = self.scene().itemsBoundingRect()
        if not r.isEmpty():
            self.fitInView(r, Qt.KeepAspectRatio)


class PreviewLabel(QtWidgets.QLabel):
    """Rendered figure.

    Edit mode: drag panel edges (crop), texts (offset) or the marker strip.
    Right-click on a text: its own size/angle menu. Files dropped here are opened.
    """

    edited = QtCore.Signal(str, str, float)  # edge (top/bottom/left/right), protein, delta in inches
    textMoved = QtCore.Signal(str, float, float)  # key, dx, dy (inches)
    stripMoved = QtCore.Signal(float)  # dx (inches)
    textMenu = QtCore.Signal(str, QtCore.QPoint)  # key, global position
    modeToggled = QtCore.Signal(bool)
    filesDropped = QtCore.Signal(list)

    CURSORS = {"top": Qt.SizeVerCursor, "bottom": Qt.SizeVerCursor, "left": Qt.SizeHorCursor,
               "right": Qt.SizeHorCursor, "text": Qt.SizeAllCursor, "strip": Qt.SizeHorCursor}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.layout_ = {"panels": [], "texts": [], "strips": [], "size": (1.0, 1.0)}
        self.edit_mode = False
        self.drag = None  # (kind, target, start inches)
        self.cur = None

    def set_layout(self, layout):
        self.layout_ = layout
        self.update()

    def _k(self):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return 0.0, 0.0, 0.0
        k = pm.width() / max(self.layout_["size"][0], 1e-6)  # screen px per inch
        return k, (self.width() - pm.width()) / 2.0, (self.height() - pm.height()) / 2.0

    def to_in(self, pos):
        k, ox, oy = self._k()
        if not k:
            return None
        return (pos.x() - ox) / k, (pos.y() - oy) / k

    def hit_text(self, pos):
        p_in = self.to_in(pos)
        if p_in is None:
            return None
        k, _, _ = self._k()
        x, y = p_in
        pad = 2.0 / k
        best = None
        for t in self.layout_.get("texts", []):
            if t["x0"] - pad <= x <= t["x1"] + pad and t["y0"] - pad <= y <= t["y1"] + pad:
                area = (t["x1"] - t["x0"]) * (t["y1"] - t["y0"])
                if best is None or area < best[0]:
                    best = (area, t)
        return best[1] if best else None

    def hit(self, pos):
        """Panel edge, then text, then marker strip under ``pos``: (kind, target)."""
        p_in = self.to_in(pos)
        k, _, _ = self._k()
        if p_in is None:
            return None
        x, y = p_in
        tol = 6.0 / k
        for p in self.layout_["panels"]:
            xr = p["x_blot"] + self.layout_.get("panel_w", 0)
            if p["x_blot"] - tol <= x <= xr + tol:
                if abs(y - p["y"]) < tol:
                    return "top", p
                if abs(y - (p["y"] + p["h"])) < tol:
                    return "bottom", p
            if p["y"] - tol <= y <= p["y"] + p["h"] + tol:
                if abs(x - p["x_blot"]) < tol:
                    return "left", p
                if abs(x - xr) < tol:
                    return "right", p
        t = self.hit_text(pos)
        if t:
            return "text", t
        for st in self.layout_.get("strips", []):
            if st["x"] <= x <= st["x"] + st["w"] and st["y"] <= y <= st["y"] + st["h"]:
                return "strip", st
        return None

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if self.drag:
            self.cur = pos
            self.update()
        elif self.edit_mode:
            h = self.hit(pos)
            if h:
                self.setCursor(self.CURSORS[h[0]])
            else:
                self.unsetCursor()
        else:
            self.setCursor(Qt.PointingHandCursor if self.hit_text(pos) else Qt.ArrowCursor)
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if self.edit_mode and e.button() == Qt.LeftButton:
            pos = e.position().toPoint()
            h = self.hit(pos)
            if h:
                self.drag = (h[0], h[1], self.to_in(pos))
                self.cur = pos
                return
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self.drag:
            kind, target, start = self.drag
            end = self.to_in(e.position().toPoint())
            self.drag = None
            self.cur = None
            self.update()
            if end is None:
                return
            dx, dy = end[0] - start[0], end[1] - start[1]
            if max(abs(dx), abs(dy)) < 1e-4:
                return
            if kind == "text":
                self.textMoved.emit(target["key"], dx, dy)
            elif kind == "strip":
                self.stripMoved.emit(dx)
            else:
                self.edited.emit(kind, target["row"], dy if kind in ("top", "bottom") else dx)
            return
        super().mouseReleaseEvent(e)

    def contextMenuEvent(self, e):
        t = self.hit_text(e.pos())
        if t:
            self.textMenu.emit(t["key"], e.globalPos())
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("편집 모드 (패널 가장자리·글자·마커 띠 드래그)")
        act.setCheckable(True)
        act.setChecked(self.edit_mode)
        act.toggled.connect(self.modeToggled.emit)
        menu.exec(e.globalPos())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.edit_mode:
            self.modeToggled.emit(False)
            return
        super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if _dropped_paths(e):
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if _dropped_paths(e):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = _dropped_paths(e)
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.edit_mode:
            return
        k, ox, oy = self._k()
        if not k:
            return
        qp = QtGui.QPainter(self)
        pw = self.layout_.get("panel_w", 0)
        qp.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120, 150), 1, Qt.DotLine))
        for t in self.layout_.get("texts", []):
            qp.drawRect(QtCore.QRectF(ox + t["x0"] * k, oy + t["y0"] * k, (t["x1"] - t["x0"]) * k, (t["y1"] - t["y0"]) * k))
        qp.setPen(QtGui.QPen(QtGui.QColor("#1e88ff"), 1.5, Qt.DashLine))
        for p in self.layout_["panels"]:
            r = QtCore.QRectF(ox + p["x_blot"] * k, oy + p["y"] * k, pw * k, p["h"] * k)
            qp.drawRect(r)
            qp.setBrush(QtGui.QColor("#1e88ff"))
            for hx, hy in ((r.center().x(), r.top()), (r.center().x(), r.bottom()), (r.left(), r.center().y()), (r.right(), r.center().y())):
                qp.drawRect(QtCore.QRectF(hx - 3, hy - 3, 6, 6))
            qp.setBrush(Qt.NoBrush)
        if self.drag and self.cur is not None:
            kind, target, start = self.drag
            qp.setPen(QtGui.QPen(QtGui.QColor("#e65100"), 2))
            if kind in ("top", "bottom"):
                qp.drawLine(0, self.cur.y(), self.width(), self.cur.y())
            elif kind in ("left", "right"):
                qp.drawLine(self.cur.x(), 0, self.cur.x(), self.height())
            else:
                now = self.to_in(self.cur)
                dx, dy = now[0] - start[0], now[1] - start[1]
                if kind == "strip":
                    dy = 0.0
                r = (target["x0"], target["y0"], target["x1"] - target["x0"], target["y1"] - target["y0"]) if kind == "text" \
                    else (target["x"], target["y"], target["w"], target["h"])
                qp.drawRect(QtCore.QRectF(ox + (r[0] + dx) * k, oy + (r[1] + dy) * k, r[2] * k, r[3] * k))
        qp.end()


class FigureEditor(QtWidgets.QWidget):
    """Figure editor for the image shown in the main window (see :meth:`set_doc`)."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.doc = None
        self.s = figure.default_settings()
        self.fig = Figure()
        self._png: Optional[QtGui.QPixmap] = None
        self._loading = False
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.render)

        controls = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(controls)
        cl.addWidget(self._build_panels())
        cl.addWidget(self._build_tiers())
        cl.addWidget(self._build_fonts())
        cl.addWidget(self._build_style())
        cl.addWidget(self._build_contrast())
        cl.addWidget(self._build_ladder())
        cl.addWidget(self._build_numbers())
        cl.addWidget(self._build_export())
        cl.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(470)

        self.preview = PreviewLabel()
        self.preview.setStyleSheet("background: white;")
        self.preview.edited.connect(self.on_crop_dragged)
        self.preview.textMoved.connect(self.on_text_moved)
        self.preview.stripMoved.connect(self.on_strip_moved)
        self.preview.textMenu.connect(self.show_text_menu)
        self.preview.filesDropped.connect(self.on_files_dropped)
        self.preview.modeToggled.connect(lambda on: self.w_editmode.setChecked(on))
        pv = QtWidgets.QScrollArea()
        pv.setWidgetResizable(True)
        pv.setWidget(self.preview)
        self.preview_area = pv
        self.size_label = QtWidgets.QLabel()
        pw = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        self.w_editmode = QtWidgets.QCheckBox("편집 모드: 패널 가장자리·글자·마커 띠 드래그 (우클릭으로도 켜고 끔 · Esc 종료)")
        self.w_editmode.setToolTip("패널 위/아래 가장자리: 그 단백질 패널의 높이, 왼쪽/오른쪽: 모든 패널의 너비 (레인 정렬 유지).\n"
                                   "글자: 끌어서 이동 (우클릭: 크기·각도). 마커 띠: 왼쪽으로 끌면 블롯과 간격이 생깁니다.\n"
                                   "마커 띠와 kDa 눈금은 패널 범위를 따라갑니다. TIF를 미리보기에 끌어다 놓으면 이미지로 열립니다.")
        self.w_editmode.toggled.connect(self.on_edit_mode)
        pl.addWidget(self.w_editmode)
        pl.addWidget(pv, 1)
        pl.addWidget(self.size_label)

        self.ref = ReferenceView()
        self.ref.clicked.connect(self.add_kda_at)
        self.ref.markerDropped.connect(lambda p: self._load_marker(p))
        rw = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rw)
        rl.setContentsMargins(0, 0, 0, 0)
        hint = QtWidgets.QLabel("왼쪽: 마커 이미지, 오른쪽: 블롯. ladder 밴드 높이를 클릭하면 kDa 값을 입력합니다. "
                                "파란 상자 = 그림에 쓰는 마커 띠, 점선 = 단백질 패널 범위. ⌘/Ctrl+휠: 확대")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        rl.addWidget(hint)
        rl.addWidget(self.ref, 1)

        self.right = QtWidgets.QTabWidget()
        self.right.addTab(pw, "미리보기")
        self.right.addTab(rw, "마커 참고 (kDa 입력)")
        self.right.currentChanged.connect(lambda i: i == 1 and QtCore.QTimer.singleShot(0, self.ref.fit))
        split = QtWidgets.QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.right)
        split.setStretchFactor(1, 1)
        split.setSizes([480, 900])
        note = QtWidgets.QLabel("그림의 대비는 보기용입니다. 정량값은 원본 raw intensity로 계산됩니다.")
        note.setStyleSheet("color: #1565c0; font-size: 11px;")
        editor = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(editor)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(note)
        el.addWidget(split, 1)
        self.placeholder = QtWidgets.QLabel(
            "TIF를 열고 '밴드 자동 검출'을 누르면 단백질/레인 그룹이 자동으로 제안되고, 여기서 바로 그림을 만들 수 있습니다.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.stack = QtWidgets.QStackedLayout(self)
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(editor)
        self.dirty = True

    def status(self, msg):
        self.main.statusBar().showMessage(msg)

    def set_doc(self, doc):
        """Show the figure of ``doc`` (the image of the main window)."""
        self.doc = doc
        self.dirty = False
        if doc is None or doc.raw is None or not doc.analysis.bands:
            self.stack.setCurrentIndex(0)
            return
        if doc.figure is None:
            doc.figure = figure.default_settings()
        self.s = doc.figure
        figure.sync_settings(self.s, doc.analysis, doc.raw.shape)
        if self.s.get("marker_path") and os.path.exists(self.s["marker_path"]) and doc.marker is None:
            self._load_marker(self.s["marker_path"], quiet=True)
        self.stack.setCurrentIndex(1)
        self.load_from_settings()

    # ------------------------------------------------------------ building blocks
    def _group(self, title):
        g = QtWidgets.QGroupBox(title)
        return g

    def _build_panels(self):
        g = self._group("단백질 패널")
        l = QtWidgets.QVBoxLayout(g)
        self.panel_table = QtWidgets.QTableWidget(0, 4)
        self.panel_table.setHorizontalHeaderLabels(["표시", "라벨", "위 여백", "아래 여백"])
        self.panel_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.panel_table.verticalHeader().setVisible(False)
        self.panel_table.setMaximumHeight(170)
        self.panel_table.itemChanged.connect(self.on_panel_item)
        l.addWidget(self.panel_table)
        return g

    def _build_style(self):
        g = self._group("모양")
        f = QtWidgets.QFormLayout(g)
        self.w_width = _spin(1.0, 12.0, 3.2, 0.1, 2, " in")
        self.w_width.setToolTip("블롯 패널의 가로 길이 (인치). 전체 그림은 라벨만큼 더 넓습니다.")
        f.addRow("패널 폭", self.w_width)
        self.w_font = QtWidgets.QComboBox()
        self.w_font.addItems(FONTS)
        self.w_font.setEditable(True)
        f.addRow("글꼴", self.w_font)
        self.w_bold = QtWidgets.QCheckBox("IB 라벨 굵게")
        f.addRow(self.w_bold)
        self.w_gap = _spin(0.0, 1.0, 0.06, 0.01, 2, " in")
        f.addRow("패널 간격", self.w_gap)
        hb = QtWidgets.QHBoxLayout()
        self.w_ml = _spin(-500, 2000, 8, 1, None, " px")
        self.w_mr = _spin(-500, 2000, 8, 1, None, " px")
        for w_ in (self.w_ml, self.w_mr):
            w_.setToolTip("바깥쪽 레인 박스 밖으로 더 보여줄 블롯 픽셀 (음수 = 안쪽으로 자름). 모든 패널에 같이 적용")
        hb.addWidget(QtWidgets.QLabel("왼쪽"))
        hb.addWidget(self.w_ml)
        hb.addWidget(QtWidgets.QLabel("오른쪽"))
        hb.addWidget(self.w_mr)
        f.addRow("좌우 여백", hb)
        self.w_frame = _spin(0.0, 3.0, 0.8, 0.1, 1, " pt")
        f.addRow("테두리 두께", self.w_frame)
        for w, key in ((self.w_width, "width_in"), (self.w_gap, "panel_gap_in"),
                       (self.w_ml, "margin_left_px"), (self.w_mr, "margin_right_px"), (self.w_frame, "frame_lw")):
            w.valueChanged.connect(lambda v, key=key: self.set(key, v))
        self.w_font.currentTextChanged.connect(lambda t: self.set("font", t))
        self.w_bold.toggled.connect(lambda on: self.set("label_bold", on))
        return g

    def _build_contrast(self):
        g = self._group("표시 대비 (그림에만 적용, 정량과 무관)")
        f = QtWidgets.QFormLayout(g)
        self.w_lo = _spin(0.0, 50.0, 0.5, 0.5, 1, " %")
        self.w_lo.setToolTip("이 퍼센타일 이하는 흰색")
        f.addRow("흰색 기준", self.w_lo)
        self.w_hi = _spin(50.0, 100.0, 99.8, 0.1, 2, " %")
        self.w_hi.setToolTip("이 퍼센타일 이상은 검은색 (낮추면 진해짐)")
        f.addRow("검은색 기준", self.w_hi)
        self.w_gamma = _spin(0.2, 3.0, 1.0, 0.05, 2)
        self.w_gamma.setToolTip("1보다 작으면 흐린 밴드가 진해집니다")
        f.addRow("감마", self.w_gamma)
        self.w_perpanel = QtWidgets.QCheckBox("패널마다 따로 대비 맞춤")
        f.addRow(self.w_perpanel)
        self.w_corr = QtWidgets.QCheckBox("배경 제거한 이미지로 표시")
        f.addRow(self.w_corr)
        for w, key in ((self.w_lo, "lo_pct"), (self.w_hi, "hi_pct"), (self.w_gamma, "gamma")):
            w.valueChanged.connect(lambda v, key=key: self.set(key, v))
        self.w_perpanel.toggled.connect(lambda on: self.set("per_panel", on))
        self.w_corr.toggled.connect(lambda on: self.set("use_corrected", on))
        return g

    def _build_ladder(self):
        g = self._group("Ladder (마커)")
        l = QtWidgets.QVBoxLayout(g)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("마커 이미지 불러오기…")
        b.setToolTip("블롯과 같은 크기(가로세로 비율)의 마커/white-light TIF")
        b.clicked.connect(self.open_marker)
        hb.addWidget(b)
        self.marker_label = QtWidgets.QLabel("없음")
        self.marker_label.setWordWrap(True)
        hb.addWidget(self.marker_label, 1)
        l.addLayout(hb)
        f = QtWidgets.QFormLayout()
        self.w_strip = QtWidgets.QCheckBox("마커 이미지 띠를 패널 왼쪽에 표시")
        f.addRow(self.w_strip)
        hb = QtWidgets.QHBoxLayout()
        self.w_sx0 = _spin(0, 100000, 0, 1, None, " px")
        self.w_sx1 = _spin(0, 100000, 0, 1, None, " px")
        hb.addWidget(self.w_sx0)
        hb.addWidget(QtWidgets.QLabel("~"))
        hb.addWidget(self.w_sx1)
        b = QtWidgets.QPushButton("자동")
        b.setToolTip("마커 이미지에서 ladder 위치를 찾습니다")
        b.clicked.connect(self.auto_strip)
        hb.addWidget(b)
        f.addRow("띠 가로 범위", hb)
        self.w_minv = QtWidgets.QCheckBox("마커 띠 반전 (어두운 ladder / 흰 배경)")
        f.addRow(self.w_minv)
        self.w_kda = QtWidgets.QCheckBox("kDa 눈금 표시")
        f.addRow(self.w_kda)
        l.addLayout(f)
        self.w_strip.toggled.connect(lambda on: self.set("show_strip", on))
        self.w_sx0.valueChanged.connect(lambda v: self.set("strip_x0", int(v)))
        self.w_sx1.valueChanged.connect(lambda v: self.set("strip_x1", int(v)))
        self.w_minv.toggled.connect(lambda on: self.set("marker_invert", on))
        self.w_kda.toggled.connect(lambda on: self.set("show_kda", on))
        self.kda_table = QtWidgets.QTableWidget(0, 2)
        self.kda_table.setHorizontalHeaderLabels(["kDa", "y (px)"])
        self.kda_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.kda_table.verticalHeader().setVisible(False)
        self.kda_table.setMaximumHeight(170)
        self.kda_table.itemChanged.connect(self.on_kda_item)
        l.addWidget(self.kda_table)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("kDa 추가 (직접 입력)")
        b.clicked.connect(self.add_kda_row)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("선택 삭제")
        b.clicked.connect(self.delete_kda)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("마커 보며 클릭으로 입력")
        b.clicked.connect(lambda: self.right.setCurrentIndex(1))
        hb.addWidget(b)
        l.addLayout(hb)
        return g

    def _build_export(self):
        g = self._group("내보내기")
        hb = QtWidgets.QHBoxLayout(g)
        hb.addWidget(QtWidgets.QLabel("해상도"))
        self.w_dpi = _spin(150, 2400, 600, 50, None, " dpi")
        hb.addWidget(self.w_dpi)
        b = QtWidgets.QPushButton("그림 저장 (PNG/TIFF/PDF/SVG)…")
        b.setStyleSheet("font-weight: bold;")
        b.clicked.connect(self.export)
        hb.addWidget(b, 1)
        return g

    # ------------------------------------------------------------ settings <-> widgets
    def load_from_settings(self):
        s = self.s
        self._loading = True
        try:
            pairs = ((self.w_width, "width_in"), (self.w_gap, "panel_gap_in"),
                     (self.w_frame, "frame_lw"), (self.w_lo, "lo_pct"),
                     (self.w_hi, "hi_pct"), (self.w_gamma, "gamma"), (self.w_sx0, "strip_x0"), (self.w_sx1, "strip_x1"))
            for w, key in pairs:
                w.setValue(s[key])
            for w_, key in ((self.w_ml, "margin_left_px"), (self.w_mr, "margin_right_px")):
                w_.setValue(int(s["margin_px"] if s.get(key) is None else s[key]))
            self.w_font.setCurrentText(s["font"])
            for w, key in ((self.w_bold, "label_bold"), (self.w_perpanel, "per_panel"), (self.w_corr, "use_corrected"),
                           (self.w_strip, "show_strip"), (self.w_minv, "marker_invert"), (self.w_kda, "show_kda")):
                w.setChecked(bool(s[key]))
            for key, sp in self.w_fonts.items():
                sp.setValue(float(s["fonts"].get(key, 8.0)))
            self.w_prot_side.setCurrentIndex(0 if s.get("protein_side", "left") == "left" else 1)
            self.w_kda_side.setCurrentIndex(0 if s.get("kda_side", "left") == "left" else 1)
            self.fill_panels()
            self.fill_tiers()
            self.fill_kda()
            self.fill_numbers()
            self._update_marker_label()
        finally:
            self._loading = False
        self.refresh_reference()
        self.schedule()

    def set(self, key, value):
        if self._loading:
            return
        self.s[key] = value
        if key in ("strip_x0", "strip_x1", "marker_invert", "use_corrected"):
            self.refresh_reference()
        self.schedule()

    def schedule(self):
        self._timer.start()

    def analysis_changed(self):
        """Proteins/lanes/boxes changed in the main window."""
        self.set_doc(self.doc)

    # panels
    def fill_panels(self):
        t = self.panel_table
        t.blockSignals(True)
        t.setRowCount(len(self.s["panels"]))
        for i, p in enumerate(self.s["panels"]):
            it = QtWidgets.QTableWidgetItem(p["row"])
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if p.get("show", True) else Qt.Unchecked)
            t.setItem(i, 0, it)
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(p["label"]))
            for col, key in ((2, "pad_top"), (3, "pad_bottom")):
                sp = _spin(-500, 2000, int(p.get(key, 0)), 1, None, " px")
                sp.valueChanged.connect(lambda v, p=p, key=key: self._panel_pad(p, key, v))
                t.setCellWidget(i, col, sp)
        t.blockSignals(False)
        t.resizeColumnToContents(0)

    def _panel_pad(self, p, key, v):
        p[key] = int(v)
        self.refresh_reference()
        self.schedule()

    def on_panel_item(self, item):
        p = self.s["panels"][item.row()]
        if item.column() == 0:
            p["show"] = item.checkState() == Qt.Checked
        elif item.column() == 1:
            p["label"] = item.text()
        self.refresh_reference()
        self.schedule()

    # ------------------------------------------------------------ header tiers
    ROTATIONS = (0, 30, 45, 60, 90)

    def _build_tiers(self):
        g = self._group("레인 위 층 (위에서 아래 순서 · '|' = 줄바꿈)")
        l = QtWidgets.QVBoxLayout(g)
        info = QtWidgets.QLabel("층마다 '레인별 이름'(예: + − + −, 12h 24h) 또는 '그룹 막대'(여러 레인을 묶는 막대)를 고르고, "
                                "옆에 조건 설명(예: Dox, Cell number)을 넣을 수 있습니다. 표에서 층을 고르면 아래에서 내용을 고칩니다.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        l.addWidget(info)
        self.tier_table = QtWidgets.QTableWidget(0, 6)
        self.tier_table.setHorizontalHeaderLabels(["층", "종류", "조건 설명", "설명 위치", "회전", "기울임"])
        self.tier_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tier_table.verticalHeader().setVisible(False)
        self.tier_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tier_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tier_table.setMaximumHeight(170)
        self.tier_table.itemChanged.connect(self.on_tier_item)
        self.tier_table.itemSelectionChanged.connect(self.on_tier_selected)
        l.addWidget(self.tier_table)
        hb = QtWidgets.QHBoxLayout()
        for text, fn in (("+ 레인별 이름 층", lambda: self.add_tier("lanes")), ("+ 그룹 막대 층", lambda: self.add_tier("groups")),
                         ("▲", lambda: self.move_tier(-1)), ("▼", lambda: self.move_tier(1)), ("층 삭제", self.delete_tier)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            hb.addWidget(b)
        hb.addStretch(1)
        l.addLayout(hb)
        # editor of the selected tier
        self.tier_detail = QtWidgets.QStackedWidget()
        self.tier_lane_table = QtWidgets.QTableWidget(0, 2)
        self.tier_lane_table.setHorizontalHeaderLabels(["레인", "텍스트"])
        self.tier_lane_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tier_lane_table.verticalHeader().setVisible(False)
        self.tier_lane_table.itemChanged.connect(self.on_tier_lane_item)
        self.tier_detail.addWidget(self.tier_lane_table)
        gw = QtWidgets.QWidget()
        gl = QtWidgets.QVBoxLayout(gw)
        gl.setContentsMargins(0, 0, 0, 0)
        self.group_table = QtWidgets.QTableWidget(0, 3)
        self.group_table.setHorizontalHeaderLabels(["그룹 텍스트", "시작 레인", "끝 레인"])
        self.group_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.itemChanged.connect(self.on_group_item)
        gl.addWidget(self.group_table, 1)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("그룹 추가")
        b.clicked.connect(self.add_group)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("선택 그룹 삭제")
        b.clicked.connect(self.delete_group)
        hb.addWidget(b)
        self.w_bar = QtWidgets.QCheckBox("막대 표시")
        self.w_bar.toggled.connect(self.on_bar_toggled)
        hb.addWidget(self.w_bar)
        hb.addStretch(1)
        gl.addLayout(hb)
        self.tier_detail.addWidget(gw)
        self.tier_detail.setMinimumHeight(200)
        l.addWidget(self.tier_detail)
        self.cur_tier = 0
        return g

    def fill_tiers(self):
        t = self.tier_table
        t.blockSignals(True)
        tiers = self.s["tiers"]
        t.setRowCount(len(tiers))
        for i, tier in enumerate(tiers):
            it = QtWidgets.QTableWidgetItem(str(i + 1))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(i, 0, it)
            kind = "레인별 이름" + (" (레인 그룹)" if tier.get("auto") else "") if tier["kind"] == "lanes" else "그룹 막대"
            it = QtWidgets.QTableWidgetItem(kind)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            t.setItem(i, 1, it)
            t.setItem(i, 2, QtWidgets.QTableWidgetItem(str(tier.get("condition", ""))))
            side = QtWidgets.QComboBox()
            side.addItem("왼쪽", "left")
            side.addItem("오른쪽", "right")
            side.setCurrentIndex(0 if tier.get("cond_side", "left") == "left" else 1)
            side.currentIndexChanged.connect(lambda _=None, tier=tier, cb=side: self._tier_set(tier, "cond_side", cb.currentData()))
            t.setCellWidget(i, 3, side)
            rot = QtWidgets.QComboBox()
            for r in self.ROTATIONS:
                rot.addItem(f"{r}°", r)
            rot.setCurrentIndex(max(0, rot.findData(int(tier.get("rotation", 0)))))
            rot.currentIndexChanged.connect(lambda _=None, tier=tier, cb=rot: self._tier_set(tier, "rotation", cb.currentData()))
            t.setCellWidget(i, 4, rot)
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if tier.get("italic") else Qt.Unchecked)
            t.setItem(i, 5, it)
        self.cur_tier = min(self.cur_tier, len(tiers) - 1) if tiers else 0
        if tiers:
            t.selectRow(self.cur_tier)
        t.blockSignals(False)
        t.resizeColumnToContents(1)
        self.fill_tier_detail()

    def _tier_set(self, tier, key, value):
        tier[key] = value
        self.schedule()

    def on_tier_item(self, item):
        tier = self.s["tiers"][item.row()]
        if item.column() == 2:
            tier["condition"] = item.text()
        elif item.column() == 5:
            tier["italic"] = item.checkState() == Qt.Checked
        self.schedule()

    def on_tier_selected(self):
        rows = sorted({i.row() for i in self.tier_table.selectedIndexes()})
        if rows:
            self.cur_tier = rows[0]
            self.fill_tier_detail()

    def current_tier(self):
        tiers = self.s["tiers"]
        return tiers[self.cur_tier] if 0 <= self.cur_tier < len(tiers) else None

    def fill_tier_detail(self):
        tier = self.current_tier()
        if tier is None:
            self.tier_detail.setEnabled(False)
            return
        self.tier_detail.setEnabled(True)
        if tier["kind"] == "lanes":
            self.tier_detail.setCurrentIndex(0)
            t = self.tier_lane_table
            t.blockSignals(True)
            texts = figure.tier_texts(self.s, tier)
            t.setRowCount(len(texts))
            for i, tx in enumerate(texts):
                lab = self.s["lanes"][i]["label"] if i < len(self.s["lanes"]) else str(i + 1)
                it = QtWidgets.QTableWidgetItem(f"{i + 1}: {lab}")
                it.setFlags(Qt.ItemIsEnabled)
                t.setItem(i, 0, it)
                t.setItem(i, 1, QtWidgets.QTableWidgetItem(tx))
            t.blockSignals(False)
            t.resizeColumnToContents(0)
        else:
            self.tier_detail.setCurrentIndex(1)
            self.w_bar.blockSignals(True)
            self.w_bar.setChecked(bool(tier.get("bar", True)))
            self.w_bar.blockSignals(False)
            self.fill_groups()

    def on_tier_lane_item(self, item):
        tier = self.current_tier()
        if tier is None or item.column() != 1:
            return
        if tier.get("auto"):
            self.s["lanes"][item.row()]["text"] = item.text()
        else:
            tier["texts"][item.row()] = item.text()
        self.schedule()

    def add_tier(self, kind):
        tier = figure.default_tier(kind, len(self.s["lanes"]))
        if kind == "groups":
            n = len(self.s["lanes"])
            tier["groups"] = [{"text": "Group", "first": 0, "last": max(0, min(1, n - 1))}] if n else []
        # new tiers go above the selected one (headers usually grow upwards)
        self.s["tiers"].insert(max(0, self.cur_tier), tier)
        self.fill_tiers()
        self.schedule()

    def move_tier(self, d):
        tiers = self.s["tiers"]
        i, j = self.cur_tier, self.cur_tier + d
        if 0 <= i < len(tiers) and 0 <= j < len(tiers):
            tiers[i], tiers[j] = tiers[j], tiers[i]
            self.cur_tier = j
            self.fill_tiers()
            self.schedule()

    def delete_tier(self):
        tier = self.current_tier()
        if tier is None:
            return
        if tier.get("auto"):
            QtWidgets.QMessageBox.information(self, "층 삭제", "레인 그룹 이름 층은 지울 수 없습니다. 텍스트를 비우면 그림에서 사라집니다.")
            return
        del self.s["tiers"][self.cur_tier]
        self.cur_tier = max(0, self.cur_tier - 1)
        self.fill_tiers()
        self.schedule()

    # groups of the selected group tier
    def _lane_combo(self, value, on_change):
        cb = QtWidgets.QComboBox()
        for i, l in enumerate(self.s["lanes"]):
            cb.addItem(f"{i + 1}: {l.get('text', '').replace('|', ' ')}", i)
        cb.setCurrentIndex(max(0, min(int(value), cb.count() - 1)))
        cb.currentIndexChanged.connect(lambda _: on_change(cb.currentData()))
        return cb

    def fill_groups(self):
        tier = self.current_tier()
        t = self.group_table
        t.blockSignals(True)
        groups = tier.get("groups", []) if tier and tier["kind"] == "groups" else []
        t.setRowCount(len(groups))
        for i, g in enumerate(groups):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(str(g["text"])))
            t.setCellWidget(i, 1, self._lane_combo(g["first"], lambda v, g=g: self._group_lane(g, "first", v)))
            t.setCellWidget(i, 2, self._lane_combo(g["last"], lambda v, g=g: self._group_lane(g, "last", v)))
        t.blockSignals(False)

    def _group_lane(self, g, key, v):
        g[key] = int(v)
        if g["first"] > g["last"]:
            g["first"], g["last"] = g["last"], g["first"]
            self.fill_groups()
        self.schedule()

    def on_group_item(self, item):
        tier = self.current_tier()
        if tier and item.column() == 0:
            tier["groups"][item.row()]["text"] = item.text()
            self.schedule()

    def on_bar_toggled(self, on):
        tier = self.current_tier()
        if tier and tier["kind"] == "groups":
            tier["bar"] = bool(on)
            self.schedule()

    def add_group(self):
        tier = self.current_tier()
        n = len(self.s["lanes"])
        if tier is None or tier["kind"] != "groups" or not n:
            return
        used = max((g["last"] for g in tier["groups"]), default=-1)
        first = min(used + 1, n - 1)
        tier["groups"].append({"text": "Group", "first": first, "last": min(first + 1, n - 1)})
        self.fill_groups()
        self.schedule()

    def delete_group(self):
        tier = self.current_tier()
        if tier is None or tier["kind"] != "groups":
            return
        for r in sorted({i.row() for i in self.group_table.selectedIndexes()}, reverse=True):
            del tier["groups"][r]
        self.fill_groups()
        self.schedule()

    # ------------------------------------------------------------ fonts and sides
    def _build_fonts(self):
        g = self._group("글자 크기 · 위치 (글자를 우클릭하면 그 글자만 따로 조절)")
        f = QtWidgets.QFormLayout(g)
        self.w_fonts = {}
        for key, (label, _) in figure.FONT_CATEGORIES.items():
            sp = _spin(3.0, 36.0, 8.0, 0.5, 1, " pt")
            sp.valueChanged.connect(lambda v, key=key: self.set_font(key, v))
            self.w_fonts[key] = sp
            f.addRow(label, sp)
        hb = QtWidgets.QHBoxLayout()
        for text, d in (("모두 −0.5", -0.5), ("모두 +0.5", 0.5)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=None, d=d: self.bump_fonts(d))
            hb.addWidget(b)
        b = QtWidgets.QPushButton("글자별 조절 모두 초기화")
        b.setToolTip("우클릭/드래그로 바꾼 개별 글자의 크기·각도·위치를 모두 되돌립니다")
        b.clicked.connect(self.reset_overrides)
        hb.addWidget(b)
        f.addRow(hb)
        self.w_prot_side = QtWidgets.QComboBox()
        self.w_kda_side = QtWidgets.QComboBox()
        for cb, key in ((self.w_prot_side, "protein_side"), (self.w_kda_side, "kda_side")):
            cb.addItem("왼쪽", "left")
            cb.addItem("오른쪽", "right")
            cb.currentIndexChanged.connect(lambda _=None, cb=cb, key=key: self.set(key, cb.currentData()))
        f.addRow("단백질 이름 위치", self.w_prot_side)
        f.addRow("kDa 눈금 위치", self.w_kda_side)
        return g

    def set_font(self, key, v):
        if self._loading:
            return
        self.s["fonts"][key] = float(v)
        self.schedule()

    def bump_fonts(self, d):
        for k in self.s["fonts"]:
            self.s["fonts"][k] = max(3.0, round(float(self.s["fonts"][k]) + d, 1))
        for o in self.s["overrides"].values():
            if o.get("size"):
                o["size"] = max(3.0, round(float(o["size"]) + d, 1))
        self.load_from_settings()

    def reset_overrides(self):
        self.s["overrides"] = {}
        self.schedule()

    # ------------------------------------------------------------ per-text edits in the preview
    def _text_info(self, key):
        return next((t for t in self.preview.layout_.get("texts", []) if t["key"] == key), None)

    def show_text_menu(self, key, gpos):
        info = self._text_info(key) or {"cat": "lane", "text": key}
        cat = info["cat"]
        o = self.s["overrides"].get(key, {})
        size = float(o.get("size") or self.s["fonts"].get(cat, 8.0))
        menu = QtWidgets.QMenu(self)
        title = menu.addAction(f"'{info['text'][:24]}'")
        title.setEnabled(False)
        menu.addAction(f"이 글자 크기… ({size:g} pt)", lambda: self._ask_text_size(key, size))
        menu.addAction(f"이 글자 각도… ({float(o.get('rot') or 0):g}°)", lambda: self._ask_text_rot(key, o))
        if o.get("dx") or o.get("dy"):
            menu.addAction("위치 되돌리기", lambda: self._override(key, dx=None, dy=None))
        if o:
            menu.addAction("이 글자 조절 모두 초기화", lambda: self._override(key, clear=True))
        menu.addSeparator()
        label = figure.FONT_CATEGORIES.get(cat, (cat,))[0]
        menu.addAction(f"'{label}' 전체 글자 크기… ({self.s['fonts'].get(cat, 8):g} pt)", lambda: self._ask_cat_size(cat))
        menu.addSeparator()
        act = menu.addAction("편집 모드 (글자·패널·마커 띠 드래그)")
        act.setCheckable(True)
        act.setChecked(self.preview.edit_mode)
        act.toggled.connect(self.w_editmode.setChecked)
        menu.exec(gpos)

    def _override(self, key, clear=False, **kw):
        ov = self.s["overrides"]
        if clear:
            ov.pop(key, None)
        else:
            o = ov.setdefault(key, {})
            for k, v in kw.items():
                if v is None:
                    o.pop(k, None)
                else:
                    o[k] = v
            if not o:
                ov.pop(key, None)
        self.render()

    def _ask_text_size(self, key, size):
        v, ok = QtWidgets.QInputDialog.getDouble(self, "글자 크기", "이 글자의 크기 (pt):", size, 3.0, 72.0, 1)
        if ok:
            self._override(key, size=float(v))

    def _ask_text_rot(self, key, o):
        v, ok = QtWidgets.QInputDialog.getDouble(self, "글자 각도", "각도 (도, 반시계 방향 +):", float(o.get("rot") or 0), -180.0, 180.0, 0)
        if ok:
            self._override(key, rot=float(v) if v else None)

    def _ask_cat_size(self, cat):
        v, ok = QtWidgets.QInputDialog.getDouble(self, "글자 크기", f"{figure.FONT_CATEGORIES.get(cat, (cat,))[0]} 크기 (pt):",
                                                 float(self.s["fonts"].get(cat, 8.0)), 3.0, 72.0, 1)
        if ok:
            self.s["fonts"][cat] = float(v)
            self.load_from_settings()

    def on_text_moved(self, key, dx, dy):
        o = self.s["overrides"].get(key, {})
        self._override(key, dx=float(o.get("dx", 0.0)) + dx, dy=float(o.get("dy", 0.0)) + dy)

    def on_strip_moved(self, dx):
        # dragging the strip left (dx < 0) opens a gap between the ladder and the blot
        self.s["strip_gap_in"] = max(0.0, float(self.s.get("strip_gap_in", 0.0)) - dx)
        self.render()

    def on_files_dropped(self, paths):
        self.main.open_paths(paths)
        self.main.open_figure_window()

    # ladder / kDa
    def _update_marker_label(self):
        p = self.s.get("marker_path")
        if self.doc.marker is not None and p:
            self.marker_label.setText(os.path.basename(p))
        else:
            self.marker_label.setText("없음 (kDa 눈금은 마커 없이도 입력 가능)")

    def open_marker(self):
        start = os.path.dirname(self.doc.image_path or "")
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "마커(ladder) 이미지 열기", start,
                                                     "Images (*.tif *.tiff *.png *.jpg);;All files (*)")
        if p:
            self._load_marker(p)

    def _load_marker(self, p, quiet=False):
        if self.doc is None or not self.main.set_marker(self.doc, p, quiet=quiet):
            return
        if not quiet:
            self.right.setCurrentIndex(1)
            self.load_from_settings()
        self._update_marker_label()

    def auto_strip(self):
        if self.doc.marker is None:
            self.status("먼저 마커 이미지를 불러오세요")
            return
        x0, x1 = figure.guess_ladder_x(self.doc.marker, self.doc.analysis)
        self.w_sx0.setValue(x0)
        self.w_sx1.setValue(x1)

    def fill_kda(self):
        t = self.kda_table
        t.blockSignals(True)
        self.s["kda"].sort(key=lambda k: k["y"])
        t.setRowCount(len(self.s["kda"]))
        for i, k in enumerate(self.s["kda"]):
            t.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{float(k['kda']):g}"))
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{float(k['y']):.0f}"))
        t.blockSignals(False)

    def on_kda_item(self, item):
        k = self.s["kda"][item.row()]
        try:
            v = float(item.text())
        except ValueError:
            self.fill_kda()
            return
        k["kda" if item.column() == 0 else "y"] = v
        self.fill_kda()
        self.refresh_reference()
        self.schedule()

    def add_kda_at(self, y: float):
        H = self.doc.raw.shape[0]
        if not 0 <= y < H:
            return
        v, ok = QtWidgets.QInputDialog.getDouble(self, "kDa 입력", f"y = {y:.0f} px 의 ladder 밴드 크기 (kDa):", 100.0, 0.1, 10000.0, 1)
        if not ok:
            return
        self.s["kda"].append({"kda": v, "y": float(round(y))})
        self.s["show_kda"] = True
        self.w_kda.setChecked(True)
        self.fill_kda()
        self.refresh_reference()
        self.schedule()

    def add_kda_row(self):
        v, ok = QtWidgets.QInputDialog.getDouble(self, "kDa 추가", "ladder 밴드 크기 (kDa):", 100.0, 0.1, 10000.0, 1)
        if not ok:
            return
        y, ok = QtWidgets.QInputDialog.getInt(self, "kDa 추가", "그 밴드의 y 위치 (블롯 픽셀):", 0, 0, self.doc.raw.shape[0] - 1)
        if not ok:
            return
        self.s["kda"].append({"kda": v, "y": float(y)})
        self.w_kda.setChecked(True)
        self.fill_kda()
        self.refresh_reference()
        self.schedule()

    def delete_kda(self):
        for r in sorted({i.row() for i in self.kda_table.selectedIndexes()}, reverse=True):
            del self.s["kda"][r]
        self.fill_kda()
        self.refresh_reference()
        self.schedule()

    # ------------------------------------------------------------ crop editing in the preview
    def on_edit_mode(self, on):
        self.preview.edit_mode = bool(on)
        if not on:
            self.preview.unsetCursor()
        else:
            self.right.setCurrentIndex(0)
            self.preview.setFocus()
        self.preview.update()
        self.status("편집 모드: 패널 가장자리(크기), 글자(위치), 마커 띠(간격)를 드래그하세요. 글자 우클릭: 크기·각도"
                    if on else "편집 모드를 끝냈습니다")

    def on_crop_dragged(self, edge, row, d_in):
        """A panel edge was dragged by ``d_in`` inches in the preview."""
        scale = self.preview.layout_.get("scale") or 0
        if not scale:
            return
        d = int(round(d_in / scale))  # blot pixels
        if edge in ("top", "bottom"):
            p = next((p for p in self.s["panels"] if p["row"] == row), None)
            if p is None:
                return
            if edge == "top":
                p["pad_top"] = int(p.get("pad_top", 0)) - d
            else:
                p["pad_bottom"] = int(p.get("pad_bottom", 0)) + d
        else:
            key = "margin_left_px" if edge == "left" else "margin_right_px"
            cur = self.s.get(key)
            cur = int(self.s["margin_px"] if cur is None else cur)
            self.s[key] = cur - d if edge == "left" else cur + d
        self.load_from_settings()
        self.render()

    # ------------------------------------------------------------ numbers under the lanes
    def _build_numbers(self):
        g = self._group("정량 숫자 넣기 (레인 아래 숫자 줄)")
        l = QtWidgets.QVBoxLayout(g)
        info = QtWidgets.QLabel("정량한 값을 골라 원하는 위치에 레인별 숫자로 넣습니다. '기준 레인=1'을 고르면 그 레인을 1.00으로 맞춥니다.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        l.addWidget(info)
        self.num_table = QtWidgets.QTableWidget(0, 5)
        self.num_table.setHorizontalHeaderLabels(["값", "위치", "왼쪽 라벨", "소수점", "기준 레인=1"])
        self.num_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.num_table.verticalHeader().setVisible(False)
        self.num_table.setMaximumHeight(170)
        self.num_table.itemChanged.connect(self.on_number_item)
        l.addWidget(self.num_table)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("숫자 줄 추가")
        b.clicked.connect(self.add_number)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("선택 줄 삭제")
        b.clicked.connect(self.delete_number)
        hb.addWidget(b)
        hb.addStretch(1)
        l.addLayout(hb)
        return g

    def number_sources(self):
        """[(key, label, values per lane)] of the current image's quantification."""
        d = self.doc
        out = []
        if d is None or not d.measurements:
            return out
        a = d.analysis
        res, _ = self.main.image_results(d)
        for name, v in res.items():
            if np.size(v) > 1:
                out.append((f"res:{name}", name, np.asarray(v, float)))
        for ri, r in enumerate(a.rows):
            out.append((f"net:{r.name}", f"{r.name} (Net, raw)", core.row_vector(a, d.measurements, ri)))
        out += self.multi_sources()
        return out

    def multi_sources(self):
        """Results of tab 5 (big groups, cross-image formulas) on this image's lanes.

        Lanes are matched by lane group name (or lane big group) as in tab 5.
        """
        m = self.main
        if self.doc not in m.docs or len(m.loaded_docs()) < 2:
            return []
        k = m.docs.index(self.doc)
        try:
            common, results, _, _ = m.compute_multi()
        except Exception:
            return []
        keys = m.image_lane_keys().get(k)
        if not keys or not common:
            return []
        out = []
        for name, r in results.items():
            mean = np.atleast_1d(np.asarray(r["mean"], float))
            if mean.size != len(common):
                continue
            vals = np.array([mean[common.index(key)] if key in common else np.nan for key in keys])
            label = f"{name} (평균, n={int(np.nanmax(r['n']))})" if r.get("n") is not None and np.size(r["n"]) else name
            out.append((f"multi:{name}", f"[여러 이미지] {label}", vals))
        return out

    def positions(self):
        out = [("bottom", "맨 아래"), ("top", "맨 위 (레인 이름 아래)")]
        for p in self.s["panels"]:
            out += [(f"above:{p['row']}", f"'{p['row']}' 위"), (f"below:{p['row']}", f"'{p['row']}' 아래")]
        return out

    def fill_numbers(self):
        t = self.num_table
        t.blockSignals(True)
        nums = self.s.setdefault("numbers", [])
        t.setRowCount(len(nums))
        sources = self.number_sources()
        for i, n in enumerate(nums):
            cb = QtWidgets.QComboBox()
            for key, label, _ in sources:
                cb.addItem(label, key)
            if cb.findData(n.get("source")) < 0:
                cb.addItem(f"(없음) {n.get('source', '')}", n.get("source"))
            cb.setCurrentIndex(cb.findData(n.get("source")))
            cb.currentIndexChanged.connect(lambda _=None, n=n, cb=cb: self._number_source(n, cb))
            t.setCellWidget(i, 0, cb)
            pc = QtWidgets.QComboBox()
            for key, label in self.positions():
                pc.addItem(label, key)
            pc.setCurrentIndex(max(0, pc.findData(n.get("pos", "bottom"))))
            pc.currentIndexChanged.connect(lambda _=None, n=n, pc=pc: self._number_set(n, "pos", pc.currentData()))
            t.setCellWidget(i, 1, pc)
            t.setItem(i, 2, QtWidgets.QTableWidgetItem(n.get("label", "")))
            sp = _spin(0, 6, int(n.get("decimals", 2)))
            sp.valueChanged.connect(lambda v, n=n: self._number_set(n, "decimals", int(v)))
            t.setCellWidget(i, 3, sp)
            rc = QtWidgets.QComboBox()
            rc.addItem("그대로", None)
            for li, lane in enumerate(self.s["lanes"]):
                rc.addItem(f"{li + 1}: {lane.get('text', '').replace('|', ' ')}", li)
            rc.setCurrentIndex(0 if n.get("rel_lane") is None else max(0, rc.findData(n.get("rel_lane"))))
            rc.currentIndexChanged.connect(lambda _=None, n=n, rc=rc: self._number_set(n, "rel_lane", rc.currentData()))
            t.setCellWidget(i, 4, rc)
        t.blockSignals(False)
        t.resizeColumnToContents(0)

    def _number_set(self, n, key, value):
        n[key] = value
        self.schedule()

    def _number_source(self, n, cb):
        old_default = n.get("_default_label")
        n["source"] = cb.currentData()
        label = cb.currentText().replace(" (Net, raw)", "").replace("[여러 이미지] ", "")
        if not n.get("label") or n.get("label") == old_default:
            n["label"] = label
            n["_default_label"] = label
            self.fill_numbers()
        self.schedule()

    def on_number_item(self, item):
        if item.column() == 2:
            self.s["numbers"][item.row()]["label"] = item.text()
            self.schedule()

    def add_number(self):
        sources = self.number_sources()
        if not sources:
            QtWidgets.QMessageBox.information(self, "정량 숫자", "먼저 정량값이 있어야 합니다 (밴드 박스와 단백질 그룹).")
            return
        key, label, _ = sources[0]
        label = label.replace(" (Net, raw)", "")
        self.s["numbers"].append({"source": key, "label": label, "_default_label": label, "pos": "bottom",
                                  "decimals": 2, "rel_lane": None})
        self.fill_numbers()
        self.schedule()

    def delete_number(self):
        for r in sorted({i.row() for i in self.num_table.selectedIndexes()}, reverse=True):
            del self.s["numbers"][r]
        self.fill_numbers()
        self.schedule()

    def resolve_numbers(self):
        """Number rows with their texts for the renderer."""
        values = {key: v for key, _, v in self.number_sources()}
        out = []
        n_lanes = len(self.s["lanes"])
        for n in self.s.get("numbers", []):
            v = values.get(n.get("source"))
            if v is None:
                continue
            v = np.concatenate([v, np.full(max(0, n_lanes - v.size), np.nan)])[:n_lanes]
            texts, suffix = figure.format_numbers(v, int(n.get("decimals", 2)), n.get("rel_lane"))
            out.append({"label": n.get("label", "") + suffix, "pos": n.get("pos", "bottom"), "texts": texts})
        return out

    # ------------------------------------------------------------ reference view
    def refresh_reference(self):
        if not self.ready():
            return
        sc = self.ref.scene()
        sc.clear()
        self.ref.cursor_line = None
        d = self.doc
        H, W = d.raw.shape
        gap = 20
        x_blot = 0
        if d.marker is not None:
            m = figure._stretch(d.marker, 1, 99.5, 1.0)
            if self.s.get("marker_invert", True):
                m = 1 - m
            sc.addPixmap(QtGui.QPixmap.fromImage(_gray_qimage(m)))
            x_blot = W + gap
            pen = QtGui.QPen(QtGui.QColor("#1e88ff"), 2)
            pen.setCosmetic(True)
            sc.addRect(self.s["strip_x0"], 0, max(1, self.s["strip_x1"] - self.s["strip_x0"]), H, pen)
            t = sc.addSimpleText("마커")
            t.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            t.setPos(4, -30)
        blot = figure._stretch(self._display_source(), 0.5, 99.8, 0.6)
        item = sc.addPixmap(QtGui.QPixmap.fromImage(_gray_qimage(1 - blot)))
        item.setPos(x_blot, 0)
        t = sc.addSimpleText("블롯")
        t.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        t.setPos(x_blot + 4, -30)
        pen = QtGui.QPen(QtGui.QColor("#e65100"), 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        x0, x1 = figure.crop_x(d.analysis, self.s, W)
        for p, _, y0, y1 in figure.panel_rows(d.analysis, self.s, H):
            sc.addRect(x_blot + x0, y0, x1 - x0, y1 - y0, pen)
        total_w = x_blot + W
        kpen = QtGui.QPen(QtGui.QColor("#c62828"), 1)
        kpen.setCosmetic(True)
        f = QtGui.QFont()
        f.setPointSize(10)
        for k in self.s["kda"]:
            sc.addLine(0, k["y"], total_w, k["y"], kpen)
            t = sc.addSimpleText(f"{float(k['kda']):g} kDa", f)
            t.setBrush(QtGui.QColor("#c62828"))
            t.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            t.setPos(total_w + 6, k["y"] - 8)
        cpen = QtGui.QPen(QtGui.QColor(0, 150, 0, 160), 1)
        cpen.setCosmetic(True)
        self.ref.cursor_line = sc.addLine(0, 0, total_w, 0, cpen)
        self.ref.width_px = total_w
        sc.setSceneRect(QtCore.QRectF(-10, -0.06 * H, total_w * 1.1 + 40, H * 1.1))

    def _display_source(self) -> np.ndarray:
        d = self.doc
        if self.s.get("use_corrected"):
            c = self.main.get_corrected(d)
            if c is not None:
                return c
        return d.signal

    # ------------------------------------------------------------ preview / export
    def ready(self) -> bool:
        d = self.doc
        return d is not None and d.raw is not None and bool(d.analysis.bands)

    def render(self):
        if not self.ready():
            return
        try:
            layout = figure.render(self.fig, self._display_source(), self.doc.analysis, self.s, self.doc.marker,
                                   self.resolve_numbers())
            self.preview.set_layout(layout)
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", dpi=PREVIEW_DPI, facecolor="white")
            pm = QtGui.QPixmap()
            pm.loadFromData(buf.getvalue(), "PNG")
            self._png = pm
            w, h = self.fig.get_size_inches()
            self.size_label.setText(f"그림 크기: {w:.2f} × {h:.2f} in ({w * 2.54:.1f} × {h * 2.54:.1f} cm)")
            self._show_preview()
        except Exception as e:  # keep the window usable while settings are half-edited
            self.size_label.setText(f"그림을 그릴 수 없습니다: {e}")

    def _show_preview(self):
        if self._png is None:
            return
        avail = self.preview_area.viewport().width() - 10
        pm = self._png
        if pm.width() > avail > 50:
            pm = pm.scaledToWidth(avail, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)

    def showEvent(self, e):
        super().showEvent(e)
        if self.dirty or self.doc is not self.main.doc:
            self.set_doc(self.main.doc)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QtCore.QTimer.singleShot(0, self._show_preview)

    def export(self):
        if not self.ready():
            return
        base = os.path.splitext(self.doc.image_path or "figure")[0] + "_figure.png"
        p, flt = QtWidgets.QFileDialog.getSaveFileName(self, "그림 저장", base,
                                                       "PNG (*.png);;TIFF (*.tif *.tiff);;PDF (*.pdf);;SVG (*.svg)")
        if not p:
            return
        if not os.path.splitext(p)[1]:
            p += {"PNG": ".png", "TIFF": ".tif", "PDF": ".pdf", "SVG": ".svg"}[flt.split()[0]]
        self.render()
        kw = {}
        if p.lower().endswith((".tif", ".tiff")):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        self.fig.savefig(p, dpi=self.w_dpi.value(), facecolor="white", **kw)
        self.status(f"저장했습니다: {p}")
