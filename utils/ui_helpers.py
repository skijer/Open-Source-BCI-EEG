# utils/ui_helpers.py
"""
Shared UI helpers.
Adds a right-aligned navigation bar that can include an optional
'Reconnect' (plug) button.
"""
from PyQt5 import QtWidgets, QtCore
from utils.theme_manager import ThemeManager
_tm   = ThemeManager.instance()
_ICON = QtCore.QSize(32, 32)

# ────────────────────────────────────────────────────────────────
def nav_bar(parent,
            on_first,               # callback for the left-most button
            on_settings,            # callback for the settings button
            *,
            first_icon="icons/return.svg",
            first_tooltip="Main menu",
            on_reconnect=None       # callback for the 🔌 button; omit to hide
           ) -> QtWidgets.QWidget:
    """Return a QWidget that holds   [ first ][ theme ][ plug? ][ settings ]  """
    bar = QtWidgets.QWidget(parent)
    lay = QtWidgets.QHBoxLayout(bar); lay.setContentsMargins(0, 0, 0, 0)
    lay.addStretch()                # right-align everything that follows

    def mk(svg, tip, cb):
        b = QtWidgets.QToolButton(bar)
        b.setFixedSize(_ICON + QtCore.QSize(8, 8)); b.setIconSize(_ICON)
        b.setToolTip(tip); b.clicked.connect(cb); b.setProperty("svg_path", svg)
        return b

    btn_first = mk(first_icon, first_tooltip, on_first)
    btn_theme = mk("icons/theme.svg", "Toggle theme", _tm.toggle)
    btn_plug  = mk("icons/plug.svg",  "Reconnect serial", on_reconnect) if on_reconnect else None
    btn_set   = mk("icons/settings.svg", "Settings", on_settings)

    for b in (btn_first, btn_theme, btn_plug, btn_set):
        if b: lay.addWidget(b)

    # tint all icons whenever the palette flips
    def _retint(_):
        for b in (btn_first, btn_theme, btn_plug, btn_set):
            if b:
                svg = b.property("svg_path")
                b.setIcon(_tm.tinted_icon(svg, _ICON))
    _tm.themeChanged.connect(_retint); _retint(_tm.is_dark)
    return bar
