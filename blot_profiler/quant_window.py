"""정량·그래프 창 (spec §6, §9): one window for band values, the quantification graph and
several images. Toolbar: the three views as a segmented control, zoom, 측정 다시 하기 and
내보내기. The pages themselves are built by the main window (they share its state)."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from .ui import controls


class QuantWindow(QtWidgets.QMainWindow):
    PAGES = (("bands", "밴드 값"), ("quant", "정량"), ("multi", "여러 이미지"))

    def __init__(self, main):
        super().__init__(main, Qt.Window)
        self.main = main
        self.setWindowTitle("정량·그래프")
        self.resize(1440, 900)
        tb = QtWidgets.QToolBar("정량")
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.sub = QtWidgets.QLabel()
        self.sub.setStyleSheet("color: rgba(60,60,67,0.6); padding-left: 6px;")
        tb.addWidget(self.sub)
        sp = QtWidgets.QWidget()
        sp.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(sp)
        self.seg = controls.Segmented(self.PAGES)
        self.seg.changed.connect(self._show)
        tb.addWidget(self.seg)
        sp = QtWidgets.QWidget()
        sp.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(sp)
        tb.addAction("−", lambda: self.zoom(0.8)).setToolTip("축소 ( ] )")
        self.zoom_label = QtWidgets.QLabel("100%")
        self.zoom_label.setFixedWidth(46)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        tb.addWidget(self.zoom_label)
        tb.addAction("+", lambda: self.zoom(1.25)).setToolTip("확대 ( [ )")
        tb.addSeparator()
        tb.addAction("측정 다시 하기", self.remeasure).setToolTip("열린 모든 이미지의 박스를 다시 측정합니다")
        self.export_btn = QtWidgets.QToolButton()
        self.export_btn.setText("내보내기")
        self.export_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        m = QtWidgets.QMenu(self.export_btn)
        m.addAction("이 그래프의 데이터 (CSV)…", main.export_graph_csv)
        im = m.addMenu("이 그래프 이미지")
        for ext, name in (("png", "PNG"), ("pdf", "PDF"), ("svg", "SVG")):
            im.addAction(f"{name}…", lambda ext=ext: main.export_plot(fmt=ext))
        m.addAction("데이터 + 이미지 함께…", main.export_graph_both)
        m.addSeparator()
        act = m.addAction("여러 개 골라서 내보내기…", main.open_export_dialog)
        act.setShortcut("Ctrl+E")
        self.export_btn.setMenu(m)
        tb.addWidget(self.export_btn)

        self.stack = QtWidgets.QStackedWidget()
        self.pages = {
            "bands": main._scroll(main._build_bands_tab()),
            "quant": main._build_results_tab(),
            "multi": main._scroll(main._build_multi_tab()),
        }
        for key, _ in self.PAGES:
            self.stack.addWidget(self.pages[key])
        self.setCentralWidget(self.stack)
        self._zoom = 1.0
        self.seg.set_value("quant")
        self.stack.setCurrentWidget(self.pages["quant"])
        main.plot.canvas.mpl_connect("button_press_event", self._press)
        main.plot.canvas.mpl_connect("motion_notify_event", self._motion)
        main.plot.canvas.mpl_connect("button_release_event", self._release)
        self._drag = None

    def page(self) -> str:
        return self.seg.value()

    def _show(self, key: str):
        self.stack.setCurrentWidget(self.pages[key])
        if key == "multi":
            self.main.refresh_multi()

    def show_page(self, key: Optional[str] = None):
        if key:
            self.seg.set_value(key)
            self._show(key)
        n = len(self.main.loaded_docs())
        self.sub.setText(f"이미지 {n}개")
        self.main.recompute_results()
        self.show()
        self.raise_()
        self.activateWindow()

    def remeasure(self):
        for d in self.main.loaded_docs():
            self.main.measure_doc(d)
        self.main.recompute_results()
        self.main.recompute_multi()
        self.main.statusBar().showMessage("열린 모든 이미지를 다시 측정했습니다")

    # ---- zoom of the graph ([ ] and the toolbar)
    def zoom(self, k: float):
        self._zoom = max(0.4, min(4.0, self._zoom * k))
        self.zoom_label.setText(f"{self._zoom * 100:.0f}%")
        sc = self.main.plot_scroll
        if abs(self._zoom - 1.0) < 1e-3:
            sc.setWidgetResizable(True)
            self.main.plot.setMinimumSize(0, 0)
        else:
            sc.setWidgetResizable(False)
            vp = sc.viewport().size()
            self.main.plot.resize(int(vp.width() * self._zoom), int(vp.height() * self._zoom))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_BracketLeft:
            self.zoom(1.25)
            return
        if e.key() == Qt.Key_BracketRight:
            self.zoom(0.8)
            return
        super().keyPressEvent(e)

    # ---- legend / text boxes: drag to move, double-click to edit (spec §6)
    def _artist_at(self, ev):
        ax = getattr(self.main, "_graph_ax", None)
        if ax is None or ev.inaxes is None and ev.x is None:
            return None, None
        for t in ax.texts:
            gid = t.get_gid() or ""
            if gid.startswith("text:") and t.contains(ev)[0]:
                return "text", t
        leg = ax.get_legend()
        if leg is not None and leg.contains(ev)[0]:
            return "legend", leg
        if ax.title.get_text() and ax.title.contains(ev)[0]:
            return "title", ax.title
        if ax.yaxis.label.contains(ev)[0]:
            return "ylabel", ax.yaxis.label
        return None, None

    def _press(self, ev):
        kind, art = self._artist_at(ev)
        if kind is None:
            return
        o = self.main.graph_opts
        if ev.dblclick:
            if kind in ("title", "ylabel"):
                key = "title" if kind == "title" else "y_title"
                text, ok = QtWidgets.QInputDialog.getText(self, "글자 편집", "글자:", text=art.get_text())
                if ok:
                    self.main.set_graph_opt(key, text)
            elif kind == "text":
                i = int(art.get_gid().split(":")[1])
                text, ok = QtWidgets.QInputDialog.getText(self, "글자 상자", "글자:", text=art.get_text())
                if ok:
                    o["texts"][i]["text"] = text
                    self.main.fill_text_boxes()
                    self.main.replot()
            elif kind == "legend":
                self.main.graph_tabs.setCurrentIndex(1)
            return
        if kind in ("legend", "text"):
            ax = self.main._graph_ax
            inv = ax.transAxes.inverted()
            p = inv.transform((ev.x, ev.y))
            if kind == "legend":
                bb = art.get_window_extent()
                x0, y1 = inv.transform((bb.x0, bb.y1))
                self._drag = ("legend", art, p, (x0, y1))
            else:
                self._drag = ("text", art, p, art.get_position())

    def _motion(self, ev):
        if not self._drag or ev.x is None:
            return
        kind, art, p0, start = self._drag
        ax = self.main._graph_ax
        p = ax.transAxes.inverted().transform((ev.x, ev.y))
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        if kind == "legend":
            art.set_bbox_to_anchor((start[0] + dx, start[1] + dy), transform=ax.transAxes)
            art._loc = 2  # upper left of the box at the anchor
        else:
            art.set_position((start[0] + dx, start[1] + dy))
        self.main.plot.canvas.draw_idle()

    def _release(self, ev):
        if not self._drag:
            return
        kind, art, p0, start = self._drag
        self._drag = None
        ax = self.main._graph_ax
        if ev.x is None:
            return
        p = ax.transAxes.inverted().transform((ev.x, ev.y))
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        o = self.main.graph_opts
        if kind == "legend":
            o["legend_pos"] = [float(start[0] + dx), float(start[1] + dy)]
        else:
            i = int(art.get_gid().split(":")[1])
            o["texts"][i]["x"], o["texts"][i]["y"] = float(start[0] + dx), float(start[1] + dy)
        self.main.replot()
