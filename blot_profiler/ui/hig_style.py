"""The app-wide style sheet of the Apple HIG design system (Desktop profile).

Every value comes from the design system: colours from the tokens in
`theme.TOKENS` (light / dark), sizes from `proportions.md` (Desktop column)
and the component READMEs.  The sheet is built once per colour scheme and
re-applied when the system switches between light and dark.

Since v0.50 the macOS **tahoe** style layer (components/MacOSStyle):
concentric radii (panel 18 → row 10, inset group 12, field 8, value 6,
menu 12), capsule buttons and chips, borderless filled fields, inset groups
(translucent background, 1 px glass line, title above at the row padding),
accent-pill selection in menus and the sidebar, no dividers.

Component mapping (design system component → Qt widget):

==================  ==========================================================
Button              QPushButton: capsule 28, ``fill``, no border; prominent =
                    accent fill; destructive = plain red; checked = accent
TextField / Select  QLineEdit / QComboBox: ``fill-tertiary``, radius 8,
NumberInput         accent border on focus; spin boxes with stacked chevrons
Toggle              QCheckBox = check box; ``hig_switch`` = switch (only the
                    자동 / 수동 choice); ``hig_chip`` = multi-select chip;
                    QRadioButton = radio
TabView             sub-tabs (QTabBar) = choice chips; the inspector uses
                    ``hig_widgets.ChipTabs`` (painted capsule chips)
Inset group         QGroupBox
List / Table        item views radius 12; sidebar rows = accent pill
Menu / Tooltip      QMenu radius 12, items accent pill on hover
Progress / Slider   6 px capsule, accent fill; slider knob 18
Surfaces            RegionLayout: raised sidebar / inspector, recessed
                    canvas (theme.region_color), no lines; the sidebar's
                    shade on the canvas edge (hig_widgets.RegionShade)
==================  ==========================================================

Not styled here (design system exclusion ``measurement-plots.md``): the image
canvas, matplotlib graphs and the Figure / graph previews.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from . import theme

# Desktop proportions (proportions.md, px)
CTL = 28  # control-md
CTL_SM = 24
PAD = 12  # control-pad
FIELD_PAD = 8
R_CTL = 6
R_GROUP = 10
R_WINDOW = 12
ROW = 28
SW_W, SW_H = 40, 22
SELECT_CAP = 18  # desktop-select-cap
CHECK = 16
STEPPER_W = 20
GAP_GROUP = 12
TH_H = 24
PROGRESS_H = 6


def rgba(name: str, scheme: str) -> str:
    c = theme._hex(theme.TOKENS[scheme][name])
    return f'rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})'


def _qrgba(c) -> str:
    return f'rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha()})'


def _tint(name: str, alpha: float, scheme: str) -> str:
    """A colour token at a fixed share (states-and-interaction.md: hover 6 %,
    selection 16 %, chip 26 %)."""
    c = theme._hex(theme.TOKENS[scheme][name])
    return f'rgba({c.red()}, {c.green()}, {c.blue()}, {round(alpha * 255)})'


# ---- SF Symbols-style glyphs, written as SVG files for the style sheet ----

def _svg(body: str, w: int, h: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">{body}</svg>')


def _icons(scheme: str) -> dict[str, str]:
    t = theme.TOKENS[scheme]
    accent, white = t['blue'][:7], '#ffffff'
    fill = theme._hex(t['fill'])
    track = f'rgb({fill.red()},{fill.green()},{fill.blue()})'
    track_op = f'{fill.alphaF():.2f}'
    border = t['separator-opaque'][:7]
    sec = theme._hex(t['label-secondary'])
    chev = f'rgb({sec.red()},{sec.green()},{sec.blue()})'
    chev_op = f'{sec.alphaF():.2f}'
    # macOS switch (Toggle component, settings look): knob with a soft
    # hairline and a small shadow, off track `fill` with an inner hairline
    knob_shadow = ('<rect x="{x}" y="2.6" width="24" height="18" rx="9" fill="#000" fill-opacity="0.10"/>'
                   '<rect x="{x}" y="2.2" width="24" height="18.4" rx="9.2" fill="#000" fill-opacity="0.05"/>')
    base = t['background-tertiary'][:7]

    def switch(on: bool, disabled: bool = False):
        # macOS 26 settings switch: a track a little wider than before and a
        # pill-shaped knob stretched sideways (not a circle)
        op = '0.4' if disabled else '1'
        x = 14 if on else 2
        bg = (f'<rect x="0" y="0" width="40" height="22" rx="11" fill="{accent}"/>' if on else
              f'<rect x="0" y="0" width="40" height="22" rx="11" fill="{track}" '
              f'fill-opacity="{track_op}"/>'
              '<rect x="0.25" y="0.25" width="39.5" height="21.5" rx="10.75" fill="none" '
              'stroke="#000" stroke-opacity="0.10" stroke-width="0.5"/>')
        knob = (knob_shadow.format(x=x)
                + f'<rect x="{x}" y="2" width="24" height="18" rx="9" fill="{white}" stroke="#000" '
                  'stroke-opacity="0.12" stroke-width="0.5"/>')
        return _svg(f'<g opacity="{op}">{bg}{knob}</g>', SW_W, SW_H)

    def check(state: str, disabled: bool = False):
        op = '0.4' if disabled else '1'
        if state == 'off':
            body = (f'<rect x="0.5" y="0.5" width="15" height="15" rx="4" fill="{base}" '
                    f'stroke="{border}"/>')
        else:
            mark = ('<path d="M4 8.2 L6.8 11 L12 5" fill="none" stroke="#fff" '
                    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                    if state == 'on' else
                    '<path d="M4.5 8 H11.5" stroke="#fff" stroke-width="2" '
                    'stroke-linecap="round"/>')
            body = f'<rect x="0" y="0" width="16" height="16" rx="4" fill="{accent}"/>{mark}'
        return _svg(f'<g opacity="{op}">{body}</g>', CHECK, CHECK)

    def radio(on: bool, disabled: bool = False):
        op = '0.4' if disabled else '1'
        body = (f'<circle cx="8" cy="8" r="8" fill="{accent}"/><circle cx="8" cy="8" r="3" '
                f'fill="#fff"/>' if on else
                f'<circle cx="8" cy="8" r="7.5" fill="{base}" stroke="{border}"/>')
        return _svg(f'<g opacity="{op}">{body}</g>', CHECK, CHECK)

    stroke = (f'fill="none" stroke="{chev}" stroke-opacity="{chev_op}" stroke-width="1.6" '
              'stroke-linecap="round" stroke-linejoin="round"')

    def select_cap(darker: float, op: float = 1.0):
        a = min(1.0, fill.alphaF() + darker)
        n = SELECT_CAP
        k = n / 24  # the 24 px chevron.up.chevron.down glyph scaled into the cap
        g = (f'<g transform="translate({n * 0.12:.2f} {n * 0.12:.2f}) scale({k * 0.76:.4f})">'
             f'<path d="m7.5 9.5 4.5-4.5 4.5 4.5M7.5 14.5l4.5 4.5 4.5-4.5" fill="none" '
             f'stroke="{chev}" stroke-opacity="{chev_op}" stroke-width="2.4" '
             'stroke-linecap="round" stroke-linejoin="round"/></g>')
        return _svg(f'<g opacity="{op}"><circle cx="{n / 2}" cy="{n / 2}" r="{n / 2}" '
                    f'fill="{track}" fill-opacity="{a:.2f}"/>{g}</g>', n, n)
    return {
        'switch_on': switch(True), 'switch_off': switch(False),
        'switch_on_dis': switch(True, True), 'switch_off_dis': switch(False, True),
        'check_on': check('on'), 'check_off': check('off'), 'check_mixed': check('mixed'),
        'check_on_dis': check('on', True), 'check_off_dis': check('off', True),
        'radio_on': radio(True), 'radio_off': radio(False),
        'radio_on_dis': radio(True, True), 'radio_off_dis': radio(False, True),
        'menu_check': _svg(f'<path d="M2.5 7.5 L5.5 10.5 L11.5 3.5" fill="none" '
                           f'stroke="{t["label"]}" stroke-width="1.8" stroke-linecap="round" '
                           'stroke-linejoin="round"/>', 14, 14),
        'menu_check_sel': _svg('<path d="M2.5 7.5 L5.5 10.5 L11.5 3.5" fill="none" '
                               'stroke="#ffffff" stroke-width="1.8" stroke-linecap="round" '
                               'stroke-linejoin="round"/>', 14, 14),
        'chip_check': _svg(f'<path d="M2 6.3 L4.8 9 L10 3.2" fill="none" stroke="{accent}" '
                           'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
                           12, 12),
        'xmark': _svg(f'<path d="M3 3 L9 9 M9 3 L3 9" {stroke}/>', 12, 12),
        # chevron.up.chevron.down (Select)
        'updown': _svg(f'<path d="M3 5.5 L6 2.5 L9 5.5 M3 8.5 L6 11.5 L9 8.5" {stroke}/>', 12, 14),
        # Select cap: round `fill` cap (18) with the chevron pair, a step
        # darker on hover, faded when disabled
        'select_cap': select_cap(0.0), 'select_cap_hover': select_cap(0.10),
        'select_cap_dis': select_cap(0.0, 0.4),
        # Stepper halves
        'up': _svg(f'<path d="M2.5 5 L5 2.5 L7.5 5" {stroke}/>', 10, 7),
        'down': _svg(f'<path d="M2.5 2 L5 4.5 L7.5 2" {stroke}/>', 10, 7),
        # disclosure triangles (closed: pointing to the start → end; open: down)
        'tri_closed': _svg(f'<path d="M3.5 2 L7.5 5 L3.5 8 Z" fill="{chev}" '
                           f'fill-opacity="{chev_op}"/>', 10, 10),
        'tri_open': _svg(f'<path d="M2 3.5 L8 3.5 L5 7.5 Z" fill="{chev}" '
                         f'fill-opacity="{chev_op}"/>', 10, 10),
    }


def icon_dir(scheme: str) -> Path:
    d = Path(tempfile.gettempdir()) / f'blotprofiler-hig-{scheme}'
    d.mkdir(parents=True, exist_ok=True)
    for name, svg in _icons(scheme).items():
        p = d / f'{name}.svg'
        if not p.exists() or p.read_text(encoding='utf-8') != svg:
            p.write_text(svg, encoding='utf-8')
    return d


# ---- the style sheet ----------------------------------------------------------

# macOS tahoe style (components/MacOSStyle): concentric radii
R_CAPSULE = CTL // 2  # buttons, chips, segmented controls
R_FIELD = 8
R_VALUE = 6
R_ROW = 10  # selected row in the sidebar list
R_INSET_GROUP = 12  # property cards (macOS tahoe group radius 12)
R_MENU = 12
INSET = 8
ROW_PAD = 12  # inset row padding = title / section header start (one grid line)
CHIP_H = 24


def _more(name: str, add: float, scheme: str) -> str:
    """A fill token made `add` more opaque (hover / pressed of a filled control)."""
    c = theme._hex(theme.TOKENS[scheme][name])
    a = min(1.0, c.alphaF() + add)
    return f'rgba({c.red()}, {c.green()}, {c.blue()}, {round(a * 255)})'


def stylesheet(scheme: str | None = None, glass: bool | None = None) -> str:
    from . import glass as _glass  # noqa: PLC0415

    s = scheme or theme.scheme()
    glass = _glass.active() if glass is None else glass
    c = lambda n: rgba(n, s)  # noqa: E731
    tint = lambda n, a: _tint(n, a, s)  # noqa: E731
    more = lambda n, a: _more(n, a, s)  # noqa: E731
    d = icon_dir(s).as_posix()
    url = lambda n: f'url("{d}/{n}.svg")'  # noqa: E731
    accent = c('blue')
    line = tint('label', 0.12)  # hig-glass-line
    edge = tint('label', 0.07 if s == 'light' else 0.10)
    seg_on = '#ffffff' if s == 'light' else '#636366'
    card = theme.TOKENS[s]['region-card']  # white cards on the inspector box (mockup 1.0.1)
    card_edge = tint('label', 0.06 if s == 'light' else 0.08)
    region = lambda r: _qrgba(theme.region_color(r, s, glass))  # noqa: E731
    if glass:  # the regions are see-through over the window's glass
        surfaces = """
QMainWindow, QSplitter, QStatusBar { background: transparent; }
"""
    else:
        surfaces = ''
    return f"""
/* ---- surfaces: the sidebar is the window's own layer; canvas and
   inspector float on it as rounded boxes with a hairline edge (v0.58) ---- */
QMainWindow, QDialog {{ background: {c('background-secondary')}; }}
QMainWindow, QMainWindow > QSplitter {{ background: {region('sidebar')}; }}
QWidget#hig_sidebar {{ background: {region('sidebar')}; border: none; }}
QWidget#hig_floor {{ background: {region('sidebar')}; border: none; }}
QWidget#hig_content {{ background: {region('canvas')}; border-radius: {theme.BOX_RADIUS}px;
    border: 1px solid {edge}; }}
QWidget#hig_inspector {{ background: {region('inspector')}; border-radius: {theme.BOX_RADIUS}px;
    border: 1px solid {edge}; }}
QToolBar {{ background: {region('sidebar')}; border: none; spacing: 6px; padding: 4px 8px; }}
/* segmented control (macOS tahoe): one capsule track, the chosen segment a raised pill */
QWidget#hig_segment {{ background: {tint('label', 0.07 if s == 'light' else 0.14)}; border-radius: 8px; }}
QWidget#hig_segment QPushButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px;
    min-height: {CTL - 6}px; padding: 0 12px; font-weight: 500; color: {c('label')}; }}
QWidget#hig_segment QPushButton:hover {{ background: {tint('label', 0.05)}; }}
QWidget#hig_segment QPushButton:checked {{ background: {seg_on}; color: {c('label')}; font-weight: 600;
    border: 1px solid {edge}; }}
/* zoom chip over a canvas: magnifier + percent and its buttons on one capsule */
QFrame#zoom_chip {{ background: {card}; border: 1px solid {edge}; border-radius: 12px; }}
QFrame#zoom_chip QLabel {{ background: transparent; color: {c('label')}; padding: 0; }}
QScrollArea, QScrollArea > QWidget > QWidget#qt_scrollarea_viewport,
QScrollArea > QWidget#qt_scrollarea_viewport {{ background: transparent; border: none; }}
QSplitter::handle {{ background: transparent; }}
QStatusBar {{ background: {region('sidebar')}; color: {c('label-secondary')}; }}
QStatusBar QLabel {{ color: {c('label-secondary')}; }}
QStatusBar::item {{ border: none; }}

/* ---- inset group (hig-inset-group): rounded translucent card, title above ---- */
QGroupBox {{
    background: {card}; border: 1px solid {card_edge};
    border-radius: {R_INSET_GROUP}px; margin-top: 26px; padding: 0;
}}
QGroupBox[hig_untitled="true"] {{ margin-top: 2px; }}
QGroupBox {{ font-weight: 700; }}
QGroupBox QLabel, QGroupBox QCheckBox, QGroupBox QRadioButton, QGroupBox QComboBox,
QGroupBox QAbstractSpinBox, QGroupBox QLineEdit, QGroupBox QListView, QGroupBox QTableView,
QGroupBox QHeaderView, QGroupBox QToolButton {{ font-weight: 400; }}
QGroupBox QPushButton {{ font-weight: 500; }}
QWidget#hig_sidebar QGroupBox {{ background: transparent; border: none; }}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 2px; top: 4px; padding: 0;
    color: {c('label')}; font-weight: 700;
}}
QGroupBox[checkable="true"]::title {{ color: {c('label')}; }}
QGroupBox[checkable="true"]:!checked::title {{ color: {c('label-secondary')}; }}
QGroupBox::indicator {{ width: {CHECK}px; height: {CHECK}px; margin-right: 2px; }}
QGroupBox::indicator:checked {{ image: {url('check_on')}; }}
QGroupBox::indicator:unchecked {{ image: {url('check_off')}; }}
QLabel[hig_section="true"] {{ padding: 12px 0 2px {ROW_PAD}px; }}
QGroupBox QLabel[hig_section="true"] {{ padding: 6px 0 2px 0; }}
/* inspector pages: text and options outside a group start on the same grid
   line as the group titles and rows (ROW_PAD) */
QWidget#hig_page QLabel {{ padding-left: {ROW_PAD}px; padding-right: 4px; }}
QWidget#hig_page QCheckBox, QWidget#hig_page QRadioButton {{ margin-left: {ROW_PAD}px; }}
QWidget#hig_page QGroupBox QLabel {{ padding-left: 0; padding-right: 0; }}
QWidget#hig_page QGroupBox QCheckBox, QWidget#hig_page QGroupBox QRadioButton {{
    margin-left: 0; }}
QWidget#hig_page QGroupBox QLabel[hig_section="true"] {{ padding: 6px 0 2px 0; }}

/* ---- Button: capsule, gray fill, no border ---- */
QPushButton {{
    min-height: {CTL - 4}px; padding: 0 {PAD + 2}px;
    border: 2px solid transparent; border-radius: {R_CAPSULE}px;
    background: {c('fill')}; color: {c('label')}; font-weight: 500;
}}
QPushButton:hover {{ background: {more('fill', 0.08)}; }}
QPushButton:pressed {{ background: {more('fill', 0.18)}; }}
QPushButton:checked {{ background: {accent}; color: {c('on-accent')}; font-weight: 600; }}
QPushButton:disabled {{ color: {c('label-tertiary')}; background: {c('fill-quaternary')}; }}
QPushButton:focus {{ border-color: {tint('blue', 0.45)}; }}
QPushButton[hig_prominent="true"] {{
    background: {accent}; color: {c('on-accent')}; font-weight: 600;
    border-bottom: 2px solid {tint('blue', 0.55)};
}}
QPushButton[hig_prominent="true"]:hover {{ background: {tint('blue', 0.88)}; }}
QPushButton[hig_prominent="true"]:pressed {{ background: {tint('blue', 0.65)}; }}
QPushButton[hig_prominent="true"]:disabled {{ background: {tint('blue', 0.40)}; }}
QPushButton[hig_destructive="true"] {{ background: transparent; color: {c('red')}; }}
QPushButton[hig_destructive="true"]:hover {{ background: {tint('red', 0.10)}; }}
QPushButton[hig_destructive="true"]:pressed {{ background: {tint('red', 0.18)}; }}
QPushButton[hig_destructive="true"]:disabled {{ color: {tint('red', 0.40)}; }}
QDialogButtonBox QPushButton {{ min-width: 64px; }}
QToolButton {{
    min-height: {CTL - 4}px; min-width: {CTL - 4}px; border-radius: {R_CAPSULE}px;
    border: 2px solid transparent; background: transparent; color: {c('label')};
}}
QToolButton:hover {{ background: {c('fill-tertiary')}; }}
QToolButton:pressed {{ background: {c('fill')}; }}
QToolButton:checked {{ background: {accent}; }}
QToolButton:focus {{ border-color: {tint('blue', 0.45)}; }}
/* toolbar capsule (hig-glass-capsule): icon buttons float on one capsule */
QWidget#hig_capsule {{
    background: {tint('background', 0.62)}; border: 1px solid {line};
    border-radius: {(CTL + 6) // 2}px;
}}

/* ---- TextField / Select / NumberInput: radius 8, filled, no border ---- */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QAbstractSpinBox {{
    background: {c('fill-tertiary')}; color: {c('label')};
    border: 1px solid transparent; border-radius: {R_FIELD}px;
    selection-background-color: {tint('blue', 0.30)}; selection-color: {c('label')};
}}
QLineEdit, QComboBox {{ min-height: {CTL - 4}px; padding: 0 {FIELD_PAD}px; }}
QPlainTextEdit, QTextEdit {{ padding: 4px {FIELD_PAD - 2}px; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus,
QAbstractSpinBox:focus {{ border: 1px solid {accent}; background: {c('background-tertiary')}; }}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
    color: {c('label-tertiary')}; background: {c('fill-quaternary')};
}}
/* Select (pop-up button): NO box: the value text and a round chevron cap at
   its end, as wide as the longest choice (theme: size policy Maximum) */
QComboBox {{
    background: transparent; border: 1px solid transparent; border-radius: {R_VALUE}px;
    min-height: {CTL_SM}px; padding: 0 {SELECT_CAP + 6}px 0 2px; color: {c('label')};
}}
QComboBox:hover, QComboBox:on {{ background: transparent; }}
QComboBox:focus {{ border: 1px solid {accent}; background: transparent; }}
QComboBox:disabled {{ background: transparent; color: {c('label-tertiary')}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    width: {SELECT_CAP}px; border: none; background: transparent;
}}
QComboBox::down-arrow {{ image: {url('select_cap')}; width: {SELECT_CAP}px; height: {SELECT_CAP}px; }}
QComboBox::down-arrow:hover, QComboBox::down-arrow:on {{ image: {url('select_cap_hover')}; }}
QComboBox::down-arrow:disabled {{ image: {url('select_cap_dis')}; }}
/* an editable combo is a text field with a pop-up: keep the field */
QComboBox:editable {{
    background: {c('fill-tertiary')}; border-radius: {R_FIELD}px; padding-left: {FIELD_PAD}px;
}}
QComboBox QAbstractItemView {{
    background: {c('background-tertiary')}; border: 1px solid {line};
    border-radius: {R_MENU}px; padding: 6px; outline: none;
    selection-background-color: {accent}; selection-color: {c('on-accent')};
}}
QAbstractSpinBox {{
    min-height: {CTL - 4}px; padding: 0 {STEPPER_W + 2}px 0 {FIELD_PAD}px;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: {STEPPER_W}px; border: none; background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right;
    border-top-right-radius: {R_FIELD}px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right;
    border-bottom-right-radius: {R_FIELD}px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {c('fill-tertiary')}; }}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{
    background: {c('fill')}; }}
/* number field next to a slider: a plain input (no stepper arrows) */
/* (stepper buttons hidden by the sheet: NoButtons collapses the edit field
   under a style sheet) */
QAbstractSpinBox[hig_plain="true"] {{ padding: 0 {FIELD_PAD}px; }}
QAbstractSpinBox[hig_plain="true"]::up-button, QAbstractSpinBox[hig_plain="true"]::down-button {{
    width: 0px; border: none; }}
QAbstractSpinBox[hig_plain="true"]::up-arrow, QAbstractSpinBox[hig_plain="true"]::down-arrow {{
    image: none; width: 0px; height: 0px; }}
QAbstractSpinBox::up-arrow {{ image: {url('up')}; width: 10px; height: 7px; }}
QAbstractSpinBox::down-arrow {{ image: {url('down')}; width: 10px; height: 7px; }}

/* ---- selection controls (owner rule 2026-11-03):
   check box = on / off option; switch = only the 자동 / 수동 choice;
   multi-select chip = several short values picked together; radio = one of
   several ---- */
QCheckBox {{ spacing: 6px; min-height: {CHECK + 2}px; }}
QCheckBox::indicator {{ width: {CHECK}px; height: {CHECK}px; }}
QCheckBox::indicator:checked {{ image: {url('check_on')}; }}
QCheckBox::indicator:unchecked {{ image: {url('check_off')}; }}
QCheckBox::indicator:indeterminate {{ image: {url('check_mixed')}; }}
QCheckBox::indicator:checked:disabled {{ image: {url('check_on_dis')}; }}
QCheckBox::indicator:unchecked:disabled {{ image: {url('check_off_dis')}; }}
QCheckBox[hig_switch="true"] {{ spacing: 6px; min-height: {SW_H}px; }}
QCheckBox[hig_switch="true"]::indicator {{ width: {SW_W}px; height: {SW_H}px; }}
QCheckBox[hig_switch="true"]::indicator:checked {{ image: {url('switch_on')}; }}
QCheckBox[hig_switch="true"]::indicator:unchecked {{ image: {url('switch_off')}; }}
QCheckBox[hig_switch="true"]::indicator:checked:disabled {{ image: {url('switch_on_dis')}; }}
QCheckBox[hig_switch="true"]::indicator:unchecked:disabled {{
    image: {url('switch_off_dis')}; }}
QCheckBox[hig_chip="true"] {{
    min-height: {CHIP_H - 2}px; padding: 0 10px; spacing: 0;
    border: 1px solid {c('separator-opaque')}; border-radius: {CHIP_H // 2}px;
    background: transparent; color: {c('label')};
}}
QCheckBox[hig_chip="true"]:hover {{ background: {c('fill-quaternary')}; }}
QCheckBox[hig_chip="true"]:checked {{
    background: {tint('blue', 0.16)}; border-color: {accent}; font-weight: 600;
}}
QCheckBox[hig_chip="true"]:disabled {{ color: {c('label-tertiary')}; }}
QCheckBox[hig_chip="true"]::indicator {{ width: 0; height: 0; image: none; }}
QCheckBox[hig_chip="true"]::indicator:checked {{
    width: 12px; height: 12px; margin-right: 4px; image: {url('chip_check')}; }}
QRadioButton {{ spacing: 6px; }}
QRadioButton::indicator {{ width: {CHECK}px; height: {CHECK}px; }}
QRadioButton::indicator:checked {{ image: {url('radio_on')}; }}
QRadioButton::indicator:unchecked {{ image: {url('radio_off')}; }}
QRadioButton::indicator:checked:disabled {{ image: {url('radio_on_dis')}; }}
QRadioButton::indicator:unchecked:disabled {{ image: {url('radio_off_dis')}; }}

/* ---- sub-tabs inside a panel: choice chips (one of several) ---- */
QTabWidget::pane {{ border: none; background: transparent; top: 0; }}
QTabWidget::tab-bar {{ alignment: left; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    min-height: {CHIP_H}px; padding: 0 {PAD}px; margin: 2px 6px 10px 0;
    border: none; border-radius: {CHIP_H // 2}px;
    background: {tint('blue', 0.12)}; color: {c('label')}; font-weight: 500;
}}
QTabBar::tab:hover {{ background: {tint('blue', 0.20)}; }}
QTabBar::tab:selected {{ background: {accent}; color: {c('on-accent')}; font-weight: 600; }}
QTabBar::close-button {{ image: {url('xmark')}; width: 12px; height: 12px;
    subcontrol-position: right; border-radius: 6px; margin-left: 4px; }}
QTabBar::close-button:hover {{ background: {c('fill')}; }}
QTabBar QToolButton {{ background: {c('background-secondary')}; }}

/* ---- List / Table / Tree ---- */
QAbstractItemView {{
    background: {c('background-tertiary')}; color: {c('label')};
    alternate-background-color: {c('fill-quaternary')};
    border: 1px solid {line}; border-radius: {R_INSET_GROUP}px;
    outline: none; padding: 2px;
    selection-background-color: {tint('blue', 0.16)}; selection-color: {c('label')};
}}
QListView::item, QTreeView::item {{
    min-height: {ROW - 4}px; padding: 2px {FIELD_PAD}px; border-radius: {R_VALUE}px;
}}
QListView::item:hover, QTreeView::item:hover, QTableView::item:hover {{
    background: {tint('blue', 0.06)}; }}
QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {{
    background: {tint('blue', 0.16)}; color: {c('label')};
}}
QTreeView::branch {{ background: transparent; }}
QTreeView::branch:has-children:closed {{ image: {url('tri_closed')}; }}
QTreeView::branch:has-children:open {{ image: {url('tri_open')}; }}
QTableView {{ gridline-color: {c('separator')}; }}
QTableView QTableCornerButton::section {{ background: transparent; border: none; }}
QHeaderView {{ background: transparent; border: none; }}
QHeaderView::section {{
    min-height: {TH_H}px; padding: 0 {FIELD_PAD + 2}px;
    background: {c('background-tertiary')}; color: {c('label-secondary')};
    font-weight: 600; border: none; border-bottom: 1px solid {c('separator')};
}}
QHeaderView::section:vertical {{ border-bottom: none; border-right: 1px solid {c('separator')}; }}
/* sidebar list: rows on the panel itself, selection = accent pill + white text */
QWidget#hig_sidebar QAbstractItemView {{ background: transparent; border: none; padding: 0; }}
QWidget#hig_sidebar QTreeView::item, QWidget#hig_sidebar QTreeView::item:hover,
QWidget#hig_sidebar QTreeView::item:selected {{
    background: transparent; color: {c('label')}; padding: 3px 6px; }}
QWidget#hig_sidebar QTreeView {{ show-decoration-selected: 0; }}
QTreeView::branch:selected, QTreeView::branch:hover, QTreeView::branch:selected:active,
QTreeView::branch:selected:!active, QTreeView::branch:!has-children {{
    background: transparent; border: none; }}

/* ---- Menu / Tooltip: accent pill on hover ---- */
QMenuBar {{ background: transparent; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: {R_CAPSULE}px; background: transparent; }}
QMenuBar::item:selected {{ background: {c('fill')}; }}
QMenu {{
    background: {c('background-tertiary')}; border: 1px solid {line};
    border-radius: {R_MENU}px; padding: 6px;
}}
QMenu::item {{ min-height: {CTL_SM}px; padding: 0 24px 0 24px; border-radius: {R_VALUE}px; }}
QMenu::item:selected {{ background: {accent}; color: {c('on-accent')}; }}
QMenu::item:disabled {{ color: {c('label-tertiary')}; }}
QMenu::separator {{ height: 1px; background: {c('separator')}; margin: 4px 10px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 6px; }}
QMenu::indicator:checked {{ image: {url('menu_check')}; }}
QMenu::indicator:checked:selected {{ image: {url('menu_check_sel')}; }}
QAbstractItemView::indicator {{ width: {CHECK}px; height: {CHECK}px; }}
QAbstractItemView::indicator:checked {{ image: {url('check_on')}; }}
QAbstractItemView::indicator:unchecked {{ image: {url('check_off')}; }}
QAbstractItemView::indicator:indeterminate {{ image: {url('check_mixed')}; }}
QToolTip {{
    background: {c('background-tertiary')}; color: {c('label')};
    border: 1px solid {line}; border-radius: 0; padding: 3px 8px;
}}  /* a tool tip is its own window: rounded corners would leave square
       white corners behind them (seen on macOS) */

/* ---- Progress / Slider / Scroll bars ---- */
QProgressBar {{
    max-height: {PROGRESS_H}px; min-height: {PROGRESS_H}px; border: none;
    border-radius: {PROGRESS_H // 2}px; background: {c('fill')}; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: {PROGRESS_H // 2}px; }}
QSlider::groove:horizontal {{ height: 4px; background: {c('fill')}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::groove:vertical {{ width: 4px; background: {c('fill')}; border-radius: 2px; }}
QSlider::add-page:vertical {{ background: {accent}; border-radius: 2px; }}
QSlider::handle {{
    background: {c('on-accent')}; border: 1px solid {c('separator-opaque')};
    width: 16px; height: 16px; border-radius: 9px;
}}
QSlider::handle:horizontal {{ margin: -7px 0; }}
QSlider::handle:vertical {{ margin: 0 -7px; }}
QSlider::handle:hover {{ border-color: {accent}; }}
QScrollBar:vertical {{ width: 10px; background: transparent; margin: 2px; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle {{ background: {c('fill')}; border-radius: 3px; min-height: 24px;
    min-width: 24px; }}
QScrollBar::handle:hover {{ background: {c('gray')}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- materials ---- */
{surfaces}
/* ---- excluded: measurement plots and the image canvas keep their look ---- */
QGraphicsView {{ border: none; border-radius: 0; padding: 0; background: transparent; }}
"""
