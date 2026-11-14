"""Window chrome on macOS (1.3): no line under the title bar, and the see-through
background of the design system's 'floating boxes · glass' region style.

Same approach as Cytosol Viewer (cytosol/glass.py):

* Title bar: the window's content goes up under the title bar (full-size content
  view), the title bar itself is transparent with no separator and no title, so the
  sidebar floor runs from the window's top edge down in one colour; the traffic
  lights sit at the top of the sidebar. The Qt content is pushed down by the title
  bar height so nothing is hidden under the lights. Done for every window that has
  the ``hig_titlebar`` property.
* Glass: an AppKit ``NSVisualEffectView`` (material *sidebar*, blending *behind
  window*) goes under the Qt view of a window with the ``hig_glass`` property; the
  floor (sidebar) paints nothing, the canvas box 55 % and the inspector box 80 %
  white (dark: 6 % / 12 %), see theme._REGIONS. Images, cards and tables stay
  opaque.

Both are skipped on other platforms, with the offscreen / minimal Qt plugins
(tests), when pyobjc is missing; the glass also when the user turned on Reduce
Transparency or switched it off in the View menu (opaque region tokens then).
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QSettings, Qt

_active = False
TITLEBAR_H = 28  # macOS title bar height (traffic lights row)


def _cocoa() -> bool:
    if sys.platform != "darwin":
        return False
    from PySide6.QtGui import QGuiApplication  # noqa: PLC0415

    if QGuiApplication.platformName() != "cocoa":
        return False
    try:
        import AppKit  # noqa: F401, PLC0415
        import objc  # noqa: F401, PLC0415
    except Exception:
        return False
    return True


def enabled_setting() -> bool:
    """View menu › 반투명 배경 (on by default)."""
    v = QSettings().value("view/glass", True)
    return v not in (False, "false", "0", 0)


def set_enabled_setting(on: bool):
    QSettings().setValue("view/glass", bool(on))


def reduce_transparency() -> bool:
    try:
        from AppKit import NSWorkspace  # noqa: PLC0415

        return bool(NSWorkspace.sharedWorkspace().accessibilityDisplayShouldReduceTransparency())
    except Exception:
        return False


def available() -> bool:
    return _cocoa() and enabled_setting() and not reduce_transparency()


def active() -> bool:
    return _active


def prepare(window) -> bool:
    """Before the window is first shown: let it be see-through."""
    if not available():
        return False
    window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    return True


def _ns_window(window):
    import objc  # noqa: PLC0415

    view = objc.objc_object(c_void_p=int(window.winId()))
    return view.window()


def unify_titlebar(window) -> bool:
    """No title bar line: content under a transparent title bar (after show)."""
    if not _cocoa():
        return False
    try:
        nsw = _ns_window(window)
        if nsw is None:
            return False
        full_size = 1 << 15  # NSWindowStyleMaskFullSizeContentView
        nsw.setStyleMask_(nsw.styleMask() | full_size)
        nsw.setTitlebarAppearsTransparent_(True)
        nsw.setTitleVisibility_(1)  # NSWindowTitleHidden
        if hasattr(nsw, "setTitlebarSeparatorStyle_"):
            nsw.setTitlebarSeparatorStyle_(1)  # NSTitlebarSeparatorStyleNone (macOS 11+)
        m = window.contentsMargins()
        window.setContentsMargins(m.left(), TITLEBAR_H, m.right(), m.bottom())
        return True
    except Exception:
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        return False


def attach(window) -> bool:
    """After the window is shown: put the sidebar material under the Qt view."""
    global _active
    if not available():
        return False
    try:
        from AppKit import (  # noqa: PLC0415
            NSColor,
            NSViewHeightSizable,
            NSViewWidthSizable,
            NSVisualEffectBlendingModeBehindWindow,
            NSVisualEffectMaterialSidebar,
            NSVisualEffectStateFollowsWindowActiveState,
            NSVisualEffectView,
            NSWindowBelow,
        )

        nsw = _ns_window(window)
        if nsw is None:
            return False
        content = nsw.contentView()
        frame = content.superview() or content
        fx = NSVisualEffectView.alloc().initWithFrame_(frame.bounds())
        fx.setMaterial_(NSVisualEffectMaterialSidebar)
        fx.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        fx.setState_(NSVisualEffectStateFollowsWindowActiveState)
        fx.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        if frame is content:
            content.addSubview_positioned_relativeTo_(fx, NSWindowBelow, None)
        else:
            frame.addSubview_positioned_relativeTo_(fx, NSWindowBelow, content)
        nsw.setOpaque_(False)
        nsw.setBackgroundColor_(NSColor.clearColor())
        _active = True
    except Exception:
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        _active = False
    return _active
