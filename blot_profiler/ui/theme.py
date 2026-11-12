"""Apple HIG design system (Desktop profile) mapped onto Qt (from Cytosol Viewer, adapted
for Blot Profiler 1.0: accent #0066cc, white sidebar / inspector, #f5f5f7 canvas).

The owner's "Apple HIG" design system describes web tokens (``--label``,
``--hig-ctl``, ``hig-text-*``, ...) and says native code must not write
values but use the system's components, semantic colours and text styles.
Qt widgets are native on macOS (the ``macos`` style draws AppKit controls),
so this module maps the design system onto Qt's own system roles:

==========================  =================================================
design system               Qt
==========================  =================================================
``label`` / ``background``  ``QPalette`` roles (WindowText, Window, Base ...)
``accent``                  ``QPalette.Highlight`` / ``Accent``
``hig-text-*`` styles       :func:`text_style` (role sizes relative to the
                            platform's body font; macOS body = 13 pt)
``label-secondary``         :func:`text_style` ``secondary=True``
``StatusIndicator``         :class:`StatusLabel` (dot + text, ``status-*-tint``)
``Button --destructive``    :func:`destructive` + :func:`confirm_destructive`
``SectionHeader``           :func:`section_header`
``FormRow`` (desktop)       trailing labels in wide forms (:class:`_Polisher`)
``Button``/``Select``/...   ``QPushButton``, ``QComboBox``, ``QSpinBox``
                            (native NumberInput + Stepper), ``QCheckBox``
==========================  =================================================

On macOS nothing is overridden: the native style and the system palette
already are the HIG (accent colour, dark mode, Increase Contrast).  On
Windows / Linux the app uses the Fusion style with a palette built from
the design system's colour tokens (light and dark), so the app looks the
same family on every platform.

Measurement plots (matplotlib graphs, Figure window, the image canvas) are
excluded by the design system's ``measurement-plots.md`` and are not touched
here.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QApplication,
    QFormLayout,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyleFactory,
    QWidget,
)

# Colour tokens of the design system (project/tokens.json), light and dark.
# Only the semantic tokens the app needs; values are Apple's system colours.
TOKENS = {
    'light': {
        'red': '#d70015', 'orange': '#ff8d28', 'green': '#248a3d', 'blue': '#0066cc',
        'gray': '#8e8e93', 'gray-3': '#c7c7cc', 'gray-5': '#e5e5ea',
        'background': '#ffffff', 'background-secondary': '#f2f2f7',
        'region-base': '#e3e3e8', 'region-canvas': '#f2f2f7', 'region-box': '#ffffff', 'region-inspector': '#f2f2f7',
        'background-tertiary': '#ffffff',
        'label': '#000000', 'label-secondary': '#3c3c4399', 'label-tertiary': '#3c3c434d',
        'placeholder': '#3c3c434d', 'separator': '#3c3c434a', 'separator-opaque': '#c6c6c8',
        'fill': '#78788033', 'fill-secondary': '#78788029', 'fill-tertiary': '#7676801f',
        'fill-quaternary': '#74748014', 'link': '#0066cc', 'on-accent': '#ffffff',
        'grouped-background': '#f2f2f7', 'grouped-background-secondary': '#ffffff',
        'gray-2': '#aeaeb2', 'gray-4': '#d1d1d6', 'gray-6': '#f2f2f7',
        'status-success-tint': '#34c75929', 'status-warning-tint': '#ff8d2829',
        'status-error-tint': '#ff383c29', 'status-info-tint': '#0088ff29',
        'status-neutral-tint': '#8e8e9329',
    },
    'dark': {
        'red': '#ff4245', 'orange': '#ff9230', 'green': '#30d158', 'blue': '#0091ff',
        'gray': '#8e8e93', 'gray-3': '#48484a', 'gray-5': '#2c2c2e',
        'background': '#000000', 'background-secondary': '#1c1c1e',
        'region-base': '#262628', 'region-canvas': '#303032', 'region-box': '#1f1f22', 'region-inspector': '#3a3a3c',
        'background-tertiary': '#2c2c2e',
        'label': '#ffffff', 'label-secondary': '#ebebf599', 'label-tertiary': '#ebebf54d',
        'placeholder': '#ebebf54d', 'separator': '#54545899', 'separator-opaque': '#38383a',
        'fill': '#7878805c', 'fill-secondary': '#78788052', 'fill-tertiary': '#7676803d',
        'fill-quaternary': '#7676802e', 'link': '#0984ff', 'on-accent': '#ffffff',
        'grouped-background': '#000000', 'grouped-background-secondary': '#1c1c1e',
        'gray-2': '#636366', 'gray-4': '#3a3a3c', 'gray-6': '#1c1c1e',
        'status-success-tint': '#30d1583d', 'status-warning-tint': '#ff92303d',
        'status-error-tint': '#ff42453d', 'status-info-tint': '#0091ff3d',
        'status-neutral-tint': '#8e8e933d',
    },
}

# status kind → (dot colour token, tint token); StatusIndicator README
STATUS = {
    'success': ('green', 'status-success-tint'),
    'warning': ('orange', 'status-warning-tint'),
    'error': ('red', 'status-error-tint'),
    'info': ('blue', 'status-info-tint'),
    'neutral': ('gray', 'status-neutral-tint'),
}

# hig-text-* roles on the Desktop (macOS) scale: size relative to body 13,
# weight.  typography-desktop.md.  Sizes follow the platform's body font, so
# they are exact on macOS (13 pt) and keep the same steps on Windows.
TEXT_ROLES = {
    'title-2': (17, QFont.Weight.Normal),
    'title-3': (15, QFont.Weight.Normal),
    'headline': (13, QFont.Weight.Bold),
    'body': (13, QFont.Weight.Normal),
    'callout': (12, QFont.Weight.Normal),
    'subhead': (11, QFont.Weight.Normal),
    'footnote': (10, QFont.Weight.Normal),
}
BODY_PT = 13.0
NUMBER_MIN_W, SELECT_MIN_W = 80, 96  # FormRow control widths (desktop)
NUMBER_MAX_W = 150  # a number field never spans a wide form row
NUMBER_FIT_W = 118  # at least this wide when its text needs it (suffix, special text)
BODY_PX = 13  # Desktop body text (typography-desktop.md)

_ROLE = 'hig_role'
_SECONDARY = 'hig_secondary'
_DESTRUCTIVE = 'hig_destructive'

_body_font: QFont | None = None


def _hex(value: str) -> QColor:
    """'#rrggbbaa' (token form) → QColor."""
    v = value.lstrip('#')
    c = QColor('#' + v[:6])
    if len(v) == 8:
        c.setAlpha(int(v[6:], 16))
    return c


def scheme() -> str:
    """'light' or 'dark', from the system (Qt ≥ 6.5) or the palette."""
    app = QGuiApplication.instance()
    if app is not None:
        try:
            cs = app.styleHints().colorScheme()
            if cs == Qt.ColorScheme.Dark:
                return 'dark'
            if cs == Qt.ColorScheme.Light:
                return 'light'
        except AttributeError:
            pass
        if app.palette().color(QPalette.ColorRole.Window).lightness() < 128:
            return 'dark'
    return 'light'


# The three regions of the window (Sidebar | Canvas | Inspector), owner
# request 2026-11-04 (v0.58, after the mockups): the sidebar is the lowest
# layer, the window's own surface (the canvas lies on it, no shade between);
# canvas and inspector float on it as two large rounded boxes (radius
# BOX_RADIUS, a hairline edge, BOX_GAP around them).  The canvas box is a
# shade darker than the inspector box.  Each region pads its content with
# the desktop margin (20).
REGION_MARGIN = 20
SIDEBAR_MARGINS = (REGION_MARGIN,) * 4
INSPECTOR_MARGINS = (REGION_MARGIN,) * 4
CONTENT_MARGINS = (REGION_MARGIN,) * 4
SHADE_W = 96  # width of the (unused since v0.58) sidebar shade on the canvas
REGION_GAP = 24  # gap-section: between a region's header and its content
BOX_RADIUS = 16  # the floating canvas / inspector boxes
BOX_GAP = 10  # space around them (window edge, between the boxes)
# token name, or (white, alpha) for the see-through (glass) variant
_REGIONS = {
    # inspector in the light grey of the pages (owner choice 2026-11-10: no white box)
    # Blot Profiler 1.0.1 (owner, after the Cytosol Viewer screenshot): the sidebar is the window's
    # base; canvas and inspector float on it as rounded boxes
    ('light', False): {'sidebar': 'region-base', 'canvas': 'region-canvas',
                       'inspector': 'region-inspector', 'shade': 0.0},
    ('dark', False): {'sidebar': 'region-base', 'canvas': 'region-canvas',
                      'inspector': 'region-inspector', 'shade': 0.0},
    # glass: the sidebar adds no tint of its own, so it is the very material
    # the title bar shows (one colour from the window top down, v0.58.1)
    ('light', True): {'sidebar': 0.0, 'canvas': 0.55, 'inspector': 0.80, 'shade': 0.0},
    ('dark', True): {'sidebar': 0.0, 'canvas': 0.06, 'inspector': 0.12, 'shade': 0.0},
}


def box_edge() -> QColor:
    """Hairline around the floating boxes."""
    c = QColor(0, 0, 0) if scheme() == 'light' else QColor(255, 255, 255)
    c.setAlphaF(0.07 if scheme() == 'light' else 0.08)
    return c


def _glass_on() -> bool:
    from . import glass  # noqa: PLC0415

    return glass.active()


def region_color(role: str, scheme_: str | None = None, glass_: bool | None = None) -> QColor:
    """Surface of a region: 'sidebar', 'canvas' or 'inspector'."""
    s = scheme_ or scheme()
    g = _glass_on() if glass_ is None else glass_
    v = _REGIONS[(s, bool(g))][role]
    if isinstance(v, str):
        return _hex(TOKENS[s][v])
    c = QColor(255, 255, 255)
    c.setAlphaF(v)
    return c


def region_shade(scheme_: str | None = None, glass_: bool | None = None) -> float:
    """Alpha `a` of the sidebar's shade on the canvas (0 = none)."""
    s = scheme_ or scheme()
    g = _glass_on() if glass_ is None else glass_
    return _REGIONS[(s, bool(g))]['shade']


def shade_gradient(x0: float, a: float):
    """The sidebar's shade as a horizontal gradient starting at `x0` (the
    region edge): black a -> 0.45a at 28 % -> 0.12a at 62 % -> 0, crisp at
    the edge, fading softly inwards."""
    from PySide6.QtGui import QLinearGradient  # noqa: PLC0415

    g = QLinearGradient(x0, 0, x0 + SHADE_W, 0)
    for at, k in ((0.0, 1.0), (0.28, 0.45), (0.62, 0.12), (1.0, 0.0)):
        c = QColor(0, 0, 0)
        c.setAlphaF(a * k)
        g.setColorAt(at, c)
    return g


def color(name: str) -> QColor:
    """A colour token for the current light / dark scheme."""
    return _hex(TOKENS[scheme()][name])


def _blend(over: QColor, under: QColor) -> QColor:
    """A token with alpha drawn over an opaque colour (for palette roles that
    must be opaque)."""
    a = over.alphaF()
    return QColor.fromRgbF(over.redF() * a + under.redF() * (1 - a),
                           over.greenF() * a + under.greenF() * (1 - a),
                           over.blueF() * a + under.blueF() * (1 - a))


def palette(name: str | None = None) -> QPalette:
    """QPalette from the design system tokens (Windows / Linux)."""
    t = {k: _hex(v) for k, v in TOKENS[name or scheme()].items()}
    R, G = QPalette.ColorRole, QPalette.ColorGroup
    win = t['background-secondary']  # level 0 of the desktop window
    base = t['background-tertiary']  # fields, lists, tables (raised)
    p = QPalette()
    for group in (G.Active, G.Inactive, G.Disabled):
        p.setColor(group, R.Window, win)
        p.setColor(group, R.Base, base)
        p.setColor(group, R.AlternateBase, _blend(t['fill-quaternary'], base))
        p.setColor(group, R.Button, base)
        p.setColor(group, R.ToolTipBase, base)
        p.setColor(group, R.Light, base)
        p.setColor(group, R.Midlight, _blend(t['separator'], base).lighter(110))
        p.setColor(group, R.Mid, t['separator-opaque'])
        p.setColor(group, R.Dark, t['gray'])
        p.setColor(group, R.Shadow, t['gray'].darker(160))
        p.setColor(group, R.Highlight, t['blue'])  # accent
        p.setColor(group, R.HighlightedText, t['on-accent'])
        p.setColor(group, R.Link, t['link'])
        p.setColor(group, R.LinkVisited, t['link'])
        p.setColor(group, R.BrightText, t['red'])
        p.setColor(group, R.PlaceholderText, t['placeholder'])
        if hasattr(R, 'Accent'):
            p.setColor(group, R.Accent, t['blue'])
        text = t['label'] if group != G.Disabled else _blend(t['label-tertiary'], win)
        for role in (R.WindowText, R.Text, R.ButtonText, R.ToolTipText):
            p.setColor(group, role, text)
    return p


def body_font() -> QFont:
    global _body_font
    if _body_font is None:
        _body_font = QFont(QApplication.font())
    return _body_font


def _role_font(role: str, base: QFont | None = None) -> QFont:
    size, weight = TEXT_ROLES[role]
    f = QFont(base or body_font())
    body = body_font()
    if body.pixelSize() > 0:
        f.setPixelSize(max(1, round(body.pixelSize() * size / BODY_PT)))
    elif body.pointSizeF() > 0:
        f.setPointSizeF(body.pointSizeF() * size / BODY_PT)
    f.setWeight(weight)
    return f


def text_style(widget: QWidget, role: str = 'body', secondary: bool = False) -> QWidget:
    """Give a label the font of a hig-text-* role and, with ``secondary``, the
    ``label-secondary`` colour.  Re-applied when the system switches between
    light and dark."""
    widget.setProperty(_ROLE, role)
    widget.setProperty(_SECONDARY, bool(secondary))
    _apply_text(widget)
    return widget


def hint(label: QWidget) -> QWidget:
    """Help / hint text under a control: Callout in label-secondary (FormRow hint).
    In a settings card it belongs to the row above (no hairline between)."""
    label.setProperty('hig_hint', True)
    return text_style(label, 'callout', secondary=True)


def secondary(label: QWidget) -> QWidget:
    """Secondary information at body size: Body in label-secondary."""
    return text_style(label, 'body', secondary=True)


def _apply_text(w: QWidget):
    role = w.property(_ROLE)
    if role:
        w.setFont(_role_font(role, w.font()))
    if w.property(_SECONDARY):
        p = w.palette()
        p.setColor(QPalette.ColorRole.WindowText, color('label-secondary'))
        p.setColor(QPalette.ColorRole.Text, color('label-secondary'))
        w.setPalette(p)


def section_header(text: str) -> QLabel:
    """SectionHeader (desktop): 11/14, weight 600, label-secondary, above a group."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    text_style(lab, 'subhead', secondary=True)
    f = lab.font()
    f.setWeight(QFont.Weight.DemiBold)
    lab.setFont(f)
    lab.setProperty(_ROLE, '')  # keep the weight on refresh; colour still follows
    lab.setProperty('hig_section', True)  # starts on the inset row grid line (style sheet)
    return lab


# ---- selection controls (owner rule 2026-11-03) -----------------------------
# check box: one on / off option (the default QCheckBox look)
# switch:    only the 자동 / 수동 choice of a value (``switch``; check boxes
#            whose text starts with '자동' become switches automatically)
# chip:      several short values picked together (multi-select chips)
# radio:     one of several (QRadioButton); sub-tabs inside a panel are
#            choice chips (style sheet)

def _restyle(w: QWidget):
    st = w.style()
    st.unpolish(w)
    st.polish(w)
    w.updateGeometry()


def switch(box: QWidget) -> QWidget:
    """Toggle switch: reserved for the 자동 / 수동 choice.  Toggle component:
    the label sits before the switch, the switch at the end of the row."""
    box.setProperty('hig_switch', True)
    box.setProperty('hig_chip', False)
    box.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    _restyle(box)
    return box


def chip(box: QWidget) -> QWidget:
    """Multi-select chip (capsule with a check mark when on)."""
    box.setProperty('hig_chip', True)
    box.setProperty('hig_switch', False)
    _restyle(box)
    return box


def is_auto_choice(box) -> bool:
    if box.property('hig_auto_choice'):  # a switch titled by its row (검출 자동화)
        return True
    t = box.text().strip()
    return t == '자동' or t.startswith('자동 ') or t.startswith('자동(')


# ---- one layout system for every inspector page -------------------------------

PAGE_SPACING = 12  # between groups (plus the 24 px title band of the next group)
GROUP_MARGINS = (12, 10, 12, 12)  # inset row padding 12: titles, labels, controls align
GROUP_SPACING = 8
FORM_H_SPACING = 12
FORM_V_SPACING = 12  # settings rows: 28 + 12 = desktop-settings-row 40, a hairline between


def unify(page: QWidget):
    """Give an inspector page the one layout system of the design system:
    the page itself without margins (the pane pads it), every group the same
    inner padding (12: the grid line that group titles, section headers, form
    labels and first controls share), the same gaps, and holder widgets
    without the style's default margins so nothing steps off the grid."""
    from PySide6.QtWidgets import QGroupBox, QLayout, QStyle, QTabWidget  # noqa: PLC0415

    pm = page.style().pixelMetric(QStyle.PixelMetric.PM_LayoutLeftMargin)

    def default_margins(lay: QLayout) -> bool:
        m = lay.contentsMargins()
        return {m.left(), m.top(), m.right(), m.bottom()} <= {pm, 9, 11}

    def forms(lay):
        if lay is None:
            return
        if isinstance(lay, QFormLayout):
            lay.setHorizontalSpacing(FORM_H_SPACING)
            lay.setVerticalSpacing(FORM_V_SPACING)
            settings_rows(lay)
        for i in range(lay.count()):
            forms(lay.itemAt(i).layout())

    if not page.objectName():
        page.setObjectName('hig_page')  # text outside the groups starts on the grid line too
    # QScrollArea.setWidget turns autoFillBackground on: the page would paint
    # an opaque sheet over the region's surface (a box inside the inspector)
    page.setAutoFillBackground(False)
    root = page.layout()
    if root is not None:
        root.setContentsMargins(0, 0, 4, 12)
        root.setSpacing(PAGE_SPACING)
    for w in page.findChildren(QWidget):
        lay = w.layout()
        if lay is None:
            continue
        if isinstance(w, QGroupBox):
            lay.setContentsMargins(*GROUP_MARGINS)
            lay.setSpacing(GROUP_SPACING)
            RowLines.attach(w)
        elif not isinstance(w, QTabWidget) and default_margins(lay):
            lay.setContentsMargins(0, 0, 0, 0)
        forms(lay)
    forms(root)


SETTINGS_ROW_H = 30  # every settings row of a card the same height (owner rule 2026-11-04)
PLAIN_NUM_W = 76  # number field beside a slider
PLAIN_NUM_MAX_W = 104


def plain_number(spin: QAbstractSpinBox) -> QAbstractSpinBox:
    """A number input without stepper arrows (beside a slider)."""
    spin.setProperty('hig_plain', True)  # the sheet hides the stepper buttons
    spin.setProperty('hig_num', True)  # the polisher leaves its width alone
    spin.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    spin.setMinimumWidth(PLAIN_NUM_W)
    spin.setMaximumWidth(PLAIN_NUM_MAX_W)
    _restyle(spin)
    return spin


def slider_field(slider: QWidget, spin: QAbstractSpinBox, *extra: QWidget) -> QWidget:
    """Slider input (owner rule 2026-11-04): title – slider – plain number
    field on one row; `extra` (a 자동 switch …) follows at the row end."""
    from PySide6.QtWidgets import QHBoxLayout, QSizePolicy  # noqa: PLC0415

    holder = QWidget()
    holder.setProperty('hig_no_reflow', True)
    holder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    lay.addWidget(slider, 1)
    lay.addWidget(plain_number(spin))
    for w in extra:
        lay.addWidget(w)
    return holder


def auto_value(spin: QAbstractSpinBox, manual: float | None = None,
               tip: str = '') -> QWidget:
    """A value whose minimum means 자동 (special value text): the choice is a
    자동 switch (owner rule 2026-11-04); while it is on the field is hidden
    (there is no value of its own to show), off brings the last manual
    value back (or `manual`)."""
    from PySide6.QtWidgets import QHBoxLayout  # noqa: PLC0415

    holder = QWidget()
    holder.setProperty('hig_no_reflow', True)
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    box = QCheckBox('자동')
    if tip:
        box.setToolTip(tip)
    lay.addWidget(spin)
    lay.addWidget(box)
    holder.auto_box = box
    spin.auto_box = box
    state = {'last': manual}

    def from_spin(v):
        on = v <= spin.minimum()
        if not on:
            state['last'] = v
        box.blockSignals(True)
        box.setChecked(on)
        box.blockSignals(False)
        spin.setVisible(not on)

    def from_box(on):
        if on:
            spin.setValue(spin.minimum())
        else:
            v = state['last']
            if v is None or v <= spin.minimum():
                v = spin.minimum() + spin.singleStep()
            spin.setValue(v)
            spin.setFocus()

    spin.valueChanged.connect(from_spin)
    box.toggled.connect(from_box)
    from_spin(spin.value())
    return holder


def show_rows(form: QFormLayout, widgets, visible: bool):
    """Show only the inputs of the chosen 자동 / 수동 mode (owner rule
    2026-11-04): hide the form rows of the other one."""
    for w in widgets:
        try:
            form.setRowVisible(w, visible)
        except (RuntimeError, TypeError):
            w.setVisible(visible)


SETTINGS_FIELD_W = 140  # hig-settings__control--field: every input / stepper of a card


def _field_width(w):
    """Inputs and steppers of a settings card share one width (left and
    right edges line up); plain fields beside a slider keep theirs."""
    from PySide6.QtWidgets import QLineEdit  # noqa: PLC0415

    if isinstance(w, (QAbstractSpinBox, QLineEdit)) and not w.property('hig_plain'):
        from PySide6.QtWidgets import QSizePolicy  # noqa: PLC0415

        w.setFixedWidth(SETTINGS_FIELD_W)
        w.setSizePolicy(QSizePolicy.Policy.Fixed, w.sizePolicy().verticalPolicy())


def settings_rows(form: QFormLayout):
    """Settings card rows (owner rule 2026-11-04): the title at the leading
    edge, the control (field, pop-up, [value][자동], swatch …) at the end of
    the row.  Fields that stretch (sliders, text fields) still fill the row."""
    from PySide6.QtWidgets import QBoxLayout  # noqa: PLC0415

    lead = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    # absolute: a switch is laid out right-to-left, which would mirror a plain AlignRight
    end = Qt.AlignmentFlag.AlignAbsolute | Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    form.setLabelAlignment(lead)
    form.setProperty('hig_settings', True)
    for r in range(form.rowCount()):
        for role in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.SpanningRole):
            it = form.itemAt(r, role)
            w = it.widget() if it is not None else None
            if w is not None and not w.property('hig_hint') \
                    and w.minimumHeight() < SETTINGS_ROW_H and w.sizeHint().height() < SETTINGS_ROW_H:
                w.setMinimumHeight(SETTINGS_ROW_H)  # one row height in the card
    for r in range(form.rowCount()):
        item = form.itemAt(r, QFormLayout.ItemRole.FieldRole)
        if item is None or form.itemAt(r, QFormLayout.ItemRole.SpanningRole) is not None:
            continue  # a spanning row (check box, note, buttons) keeps its place
        lab = form.itemAt(r, QFormLayout.ItemRole.LabelRole)
        if lab is None or (isinstance(lab.widget(), QLabel) and not lab.widget().text().strip()):
            continue  # a slider row under its field follows that field
        sub = item.layout()
        if isinstance(sub, QBoxLayout) and sub.direction() in (
                QBoxLayout.Direction.LeftToRight, QBoxLayout.Direction.RightToLeft):
            n = sub.count()
            nxt = form.itemAt(r + 1, QFormLayout.ItemRole.FieldRole) if r + 1 < form.rowCount() else None
            if nxt is not None and nxt.layout() is not None and nxt.layout().count() > 1:
                continue  # [min – max] above a pair of sliders: keep them lined up
            fields = [sub.itemAt(i).widget() for i in range(n)
                      if isinstance(sub.itemAt(i).widget(), QAbstractSpinBox)]
            for i in range(n):  # a value field keeps its own width
                if isinstance(sub.itemAt(i).widget(), (QAbstractSpinBox, QComboBox)):
                    sub.setStretch(i, 0)
            if len(fields) == 1:  # [value][자동]: the card's field width
                _field_width(fields[0])
            if n and sub.itemAt(n - 1).spacerItem() is not None:
                sub.insertItem(0, sub.takeAt(n - 1))  # a trailing stretch moves to the front
            elif not sub.expandingDirections() & Qt.Orientation.Horizontal:
                for i in range(n):
                    sub.setStretch(i, 0)
                sub.insertStretch(0, 1)
        if item.widget() is not None:
            _field_width(item.widget())
        if sub is None and item.widget() is not None and not (
                item.widget().sizePolicy().expandingDirections() & Qt.Orientation.Horizontal):
            item.setAlignment(end)


def _slider_only(item) -> bool:
    from PySide6.QtWidgets import QAbstractSlider  # noqa: PLC0415

    w = item.widget()
    if w is not None:
        return isinstance(w, QAbstractSlider)
    lay = item.layout()
    if lay is None:
        return False
    kids = [lay.itemAt(i) for i in range(lay.count())]
    return any(k.widget() is not None for k in kids) and all(
        k.spacerItem() is not None or isinstance(k.widget(), (QAbstractSlider, QLabel)) for k in kids)


_GROUP_SEQ = [0]


def group_rows(*widgets, name: str | None = None) -> str:
    """Rows that belong together (owner rule 2026-11-04): no hairline between
    them, only around the group.  Pass any widget of each row (its title
    label, field, holder or a widget inside the field's layout)."""
    if name is None:
        _GROUP_SEQ[0] += 1
        name = f'g{_GROUP_SEQ[0]}'
    for w in widgets:
        if w is not None:
            w.setProperty('hig_group', name)
    return name


def _item_group(item):
    if item is None:
        return None
    w = item.widget()
    if w is not None:
        g = w.property('hig_group')
        if g:
            return g
        for c in w.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            g = c.property('hig_group')
            if g:
                return g
        return None
    lay = item.layout()
    if lay is not None:
        for i in range(lay.count()):
            g = _item_group(lay.itemAt(i))
            if g:
                return g
    return None


def _row_group(form, r):
    for role in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole,
                 QFormLayout.ItemRole.SpanningRole):
        g = _item_group(form.itemAt(r, role))
        if g:
            return g
    return None


class RowLines(QWidget):
    """Hairlines between the rows of a settings card (Settings component):
    drawn over the group box in the gaps between consecutive form rows, from
    the text start to the card's inner edge.  A row with an empty title (a
    slider under its field) continues the row above it: no line."""

    @classmethod
    def attach(cls, group: QWidget):
        if group.findChild(RowLines, options=Qt.FindChildOption.FindDirectChildrenOnly) is None:
            cls(group)

    def __init__(self, group: QWidget):
        super().__init__(group)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setGeometry(group.rect())
        group.installEventFilter(self)
        self.raise_()

    def eventFilter(self, obj, ev):
        if obj is self.parentWidget() and ev.type() in (
                QEvent.Type.Resize, QEvent.Type.LayoutRequest, QEvent.Type.Show):
            self.setGeometry(obj.rect())
            self.raise_()
            self.update()
        return False

    def lines(self) -> list[tuple[int, int, int]]:
        """(y, x0, x1) of every hairline, in the group's coordinates."""
        from PySide6.QtCore import QPoint  # noqa: PLC0415

        group = self.parentWidget()
        out = []
        # SettingsPane: one 1 px line between settings, the card's full width
        x0, x1 = 0, group.width()
        # rows of every settings form in the card (pages such as 자동 / 수동 sit
        # in forms of their own), in the card's coordinates, top to bottom
        rows = []
        for form in group.findChildren(QFormLayout):
            if not form.property('hig_settings'):
                continue
            host = form.parentWidget()
            if host is None or not host.isVisibleTo(group) or not host.isVisible():
                continue
            off = host.mapTo(group, QPoint(0, 0))
            for r in range(form.rowCount()):
                rect, cont, shown, nested = None, False, False, False
                for role in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole,
                             QFormLayout.ItemRole.SpanningRole):
                    it = form.itemAt(r, role)
                    if it is None or it.isEmpty():
                        continue
                    w = it.widget()
                    if w is not None and not w.isVisible():
                        continue
                    if w is not None and any(f.property('hig_settings') for f in w.findChildren(QFormLayout)):
                        nested = True  # a page holding its own rows: those count instead
                        break
                    if isinstance(w, QLabel) and not w.text().strip() and w.pixmap().isNull():
                        cont = cont or role == QFormLayout.ItemRole.LabelRole
                        continue  # an empty title / an empty status line
                    if role == QFormLayout.ItemRole.SpanningRole and (
                            _slider_only(it) or (w is not None and w.property('hig_hint'))):
                        cont = True  # a slider / the hint under the field above
                    g = it.geometry()
                    if g.height() <= 0:
                        continue
                    shown = True
                    rect = g if rect is None else rect.united(g)
                if nested or not shown:
                    continue
                lab = form.itemAt(r, QFormLayout.ItemRole.LabelRole)
                lw = lab.widget() if lab is not None else None
                if isinstance(lw, QLabel) and lw.text().startswith(' '):
                    cont = True  # an indented title ('  최소') belongs to the row above
                if form.itemAt(r, QFormLayout.ItemRole.LabelRole) is None \
                        and form.itemAt(r, QFormLayout.ItemRole.SpanningRole) is None:
                    cont = True  # addRow('', slider): a field without a title follows the row above
                rows.append((rect.translated(off), cont, _row_group(form, r)))
        rows.sort(key=lambda t: t[0].top())
        for (a, _c, ga), (b, cont, gb) in zip(rows, rows[1:]):
            if cont or (ga and ga == gb) or b.top() <= a.bottom():
                continue  # rows grouped by meaning: no line inside the group
            out.append(((a.bottom() + 1 + b.top()) // 2, x0, x1))
        return out

    def paintEvent(self, _ev):
        from PySide6.QtGui import QPen  # noqa: PLC0415

        p = QPainter(self)
        p.setPen(QPen(color('separator'), 0))  # hairline: one device pixel
        for y, a, b in self.lines():
            p.drawLine(a, y, b, y)
        p.end()


def separator() -> QFrame:
    """1 px separator line (used to set destructive actions apart)."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setProperty('hig_separator', True)
    _apply_separator(line)
    return line


def _apply_separator(line: QFrame):
    p = line.palette()
    p.setColor(QPalette.ColorRole.WindowText, color('separator-opaque'))
    line.setPalette(p)


# ---- destructive actions --------------------------------------------------

def destructive(button: QAbstractButton) -> QAbstractButton:
    """Button --plain --destructive: no fill, red label (states-and-interaction.md)."""
    button.setProperty(_DESTRUCTIVE, True)
    if isinstance(button, QPushButton):
        button.setFlat(True)
        button.setAutoDefault(False)
    _apply_destructive(button)
    return button


def _apply_destructive(button: QAbstractButton):
    p = button.palette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        p.setColor(group, QPalette.ColorRole.ButtonText, color('red'))
    button.setPalette(p)


def confirm_destructive(parent, title: str, text: str, action: str,
                        info: str = '') -> bool:
    """Alert for an action that cannot be undone: the title names the object,
    the default (Return) and Escape button is 취소, the action is red."""
    box, act = destructive_alert(parent, title, text, action, info)
    box.exec()
    return box.clickedButton() is act


def destructive_alert(parent, title: str, text: str, action: str, info: str = ''):
    """The confirmation alert of :func:`confirm_destructive` (not shown)."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    box.setInformativeText(info or '이 작업은 되돌릴 수 없습니다.')
    cancel = box.addButton('취소', QMessageBox.ButtonRole.RejectRole)
    act = box.addButton(action, QMessageBox.ButtonRole.DestructiveRole)
    _apply_destructive(act)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    return box, act


# ---- status ---------------------------------------------------------------

def status_icon(kind: str, size: int = 10) -> QIcon:
    """The StatusIndicator dot as an icon (e.g. for a button that has a state)."""
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.GlobalColor.transparent)
    qp = QPainter(pm)
    qp.setRenderHint(QPainter.RenderHint.Antialiasing)
    qp.setPen(Qt.PenStyle.NoPen)
    qp.setBrush(color(STATUS[kind][0]))
    qp.drawEllipse(QRectF(1, 1, size - 2, size - 2))
    qp.end()
    return QIcon(pm)


class StatusLabel(QLabel):
    """StatusIndicator: a status dot and its text in ``label`` colour on the
    ``status-*-tint`` background, never colour alone.  ``kind=None`` shows
    plain secondary hint text (no dot, no tint)."""

    DOT = 8
    GAP = 6
    PAD_X = 8
    PAD_Y = 2

    def __init__(self, text: str = '', kind: str | None = None, parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self._kind = None
        self.setProperty(_ROLE, 'callout')
        _apply_text(self)
        self.set_kind(kind)

    def kind(self):
        return self._kind

    def set_kind(self, kind: str | None):
        self._kind = kind if kind in STATUS else None
        self.setProperty(_SECONDARY, self._kind is None)
        p = self.palette()
        p.setColor(QPalette.ColorRole.WindowText,
                   color('label' if self._kind else 'label-secondary'))
        self.setPalette(p)
        if self._kind:
            self.setContentsMargins(self.PAD_X + self.DOT + self.GAP, self.PAD_Y,
                                    self.PAD_X, self.PAD_Y)
        else:
            self.setContentsMargins(0, 0, 0, 0)
        self.update()

    def paintEvent(self, ev):
        if self._kind and self.text():
            dot, tint = STATUS[self._kind]
            qp = QPainter(self)
            qp.setRenderHint(QPainter.RenderHint.Antialiasing)
            qp.setPen(Qt.PenStyle.NoPen)
            r = QRectF(self.rect())
            qp.setBrush(color(tint))
            qp.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)  # radius-control
            qp.setBrush(color(dot))
            line = self.fontMetrics().height()
            y = self.PAD_Y + (line - self.DOT) / 2
            qp.drawEllipse(QRectF(self.PAD_X, y, self.DOT, self.DOT))
            qp.end()
        super().paintEvent(ev)


# ---- application ------------------------------------------------------------

class _Polisher(QObject):
    """Window-level touches when a window is first shown: wide forms get the
    desktop FormRow layout (trailing labels).  Forms that `flow.reflow` lets
    wrap (narrow side panes, < 480 px) keep labels above long fields, which is
    the FormRow rule for narrow containers."""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Polish and isinstance(obj, QAbstractSpinBox) \
                and not obj.property('hig_num'):
            # stepper: the value starts at the leading edge (owner rule
            # 2026-11-04), the arrows sit at the end
            obj.setProperty('hig_num', True)
            obj.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            # NumberInput width: digits + 2 (72–96 px); the field still grows
            # to its size hint when the pane is wide enough
            if not obj.minimumWidth():
                # wide enough for its suffix / special text ('80.0 µm²', '제한 없음')
                need = obj.sizeHint().width()
                obj.setMinimumWidth(max(NUMBER_MIN_W, min(need, NUMBER_FIT_W)))
            # ... but not across a whole form row: a number field is short
            if obj.maximumWidth() >= 16777215:
                obj.setMaximumWidth(NUMBER_MAX_W)
        if ev.type() == QEvent.Type.Polish and isinstance(obj, QPushButton) \
                and obj.focusPolicy() == Qt.FocusPolicy.StrongFocus:
            obj.setFocusPolicy(Qt.FocusPolicy.TabFocus)  # macOS: a click gives no focus ring
        if ev.type() == QEvent.Type.Polish and isinstance(obj, QCheckBox) \
                and obj.property('hig_switch') is None and obj.property('hig_chip') is None:
            if is_auto_choice(obj):
                switch(obj)
            else:
                obj.setProperty('hig_switch', False)
        if ev.type() == QEvent.Type.Polish and isinstance(obj, QComboBox) \
                and not obj.property('hig_sel') and not obj.minimumWidth():
            obj.setProperty('hig_sel', True)
            obj.setMinimumWidth(SELECT_MIN_W)  # can shrink in a narrow pane
            if not obj.isEditable():
                # Select component: a borderless pop-up as wide as its longest
                # choice, never stretched like a field
                from PySide6.QtWidgets import QSizePolicy  # noqa: PLC0415

                obj.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
                obj.setSizePolicy(QSizePolicy.Policy.Maximum, obj.sizePolicy().verticalPolicy())
        if ev.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow() \
                and not obj.property('hig_polished'):
            obj.setProperty('hig_polished', True)
            align_forms(obj)
            if obj.property('hig_glass'):
                from . import glass  # noqa: PLC0415

                glass.attach(obj)
                # see-through panes, or opaque fallback; widgets that picked a
                # region colour before the glass came up (the canvas view)
                # take the glass one now, so a region shows one colour
                refresh_regions()
        return False


def refresh_regions():
    app = QApplication.instance()
    if app is None:
        return
    _apply_sheet(app)
    for w in app.allWidgets():
        if hasattr(w, 'refresh_theme') and not w.property('hig_icon'):
            w.refresh_theme()


def align_forms(root: QWidget):
    # dialogs too: titles at the leading edge, controls at the row end
    for form in root.findChildren(QFormLayout):
        if not form.property('hig_settings'):
            settings_rows(form)


def refresh():
    """Re-apply token colours after the system switched light / dark."""
    app = QApplication.instance()
    if app is None:
        return
    if _custom_palette:
        app.setPalette(palette())
        _apply_sheet(app)
    for w in app.allWidgets():
        if isinstance(w, StatusLabel):
            w.set_kind(w.kind())
        elif w.property(_SECONDARY):
            _apply_text(w)
        if w.property(_DESTRUCTIVE):
            _apply_destructive(w)
        if w.property('hig_separator'):
            _apply_separator(w)
        if w.property('hig_icon') and hasattr(w, 'refresh_icon'):
            w.refresh_icon()
        elif hasattr(w, 'refresh_theme'):
            w.refresh_theme()


_custom_palette = False
_polisher: _Polisher | None = None


def native() -> bool:
    """True where Qt draws the platform's own (AppKit) controls."""
    return sys.platform == 'darwin'


def apply(app: QApplication):
    """Apply the design system to the whole application (call once, before
    the first window is built): Fusion as the neutral base, the token
    palette, the body font (13 px, typography-desktop.md) and the component
    style sheet of `hig_style` on every platform, so macOS and Windows show
    the same design system components."""
    global _custom_palette, _polisher, _body_font
    if not app.property('hig_applied'):
        app.setProperty('hig_applied', True)
        f = QFont(app.font())
        f.setPixelSize(BODY_PX)
        app.setFont(f)
        _body_font = QFont(f)
    fusion = QStyleFactory.create('Fusion')
    if fusion is not None:
        app.setStyle(fusion)
    app.setPalette(palette())
    _custom_palette = True
    _apply_sheet(app)
    if _polisher is None:
        _polisher = _Polisher(app)
        app.installEventFilter(_polisher)
        try:
            app.styleHints().colorSchemeChanged.connect(lambda *_: refresh())
        except AttributeError:
            pass


def _apply_sheet(app: QApplication):
    from . import hig_style  # noqa: PLC0415

    app.setStyleSheet(hig_style.stylesheet())


def prominent(button: QPushButton) -> QPushButton:
    """Button --prominent: the accent-filled main action (one, at most two, a screen)."""
    button.setProperty('hig_prominent', True)
    return button
