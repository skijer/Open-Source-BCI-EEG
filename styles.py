NEON_BLUE   = "#00d8ff"
NEON_ORANGE = "#ff9a00"
BG_DARK     = "#0d1117"
BG_LIGHT    = "#fafbfc"
PORTAL_FONT = "Consolas, 'Courier New', monospace"

DARK = f"""
QMainWindow, QTabWidget::pane, QWidget {{
    background: {BG_DARK};  color: #e6edf3;  font-family: {PORTAL_FONT};
}}
QTabBar::tab {{
    background: transparent;  padding: 8px 14px;  margin: 0 4px;
    border: 2px solid {NEON_BLUE};  border-bottom: none;
    border-radius: 8px 8px 0 0;
}}
QTabBar::tab:selected {{
    background: {NEON_BLUE};  color: {BG_DARK};  font-weight: bold;
}}
QToolButton {{
    background: {NEON_ORANGE};  color:{BG_DARK};  border-radius: 6px;  padding:4px 10px;
}}
QPushButton, QComboBox, QCheckBox, QLineEdit {{
    background: #161b22;  color:#e6edf3;  border:1px solid {NEON_BLUE};
    border-radius:6px; padding:4px 8px;
}}
QPushButton:hover {{ background:{NEON_BLUE}; color:{BG_DARK}; }}
QFrame {{
    border: 2px solid {NEON_BLUE}; border-radius:10px;
}}
"""
LIGHT = (
    DARK.replace(BG_DARK, BG_LIGHT)
        .replace("#e6edf3", "#0d1117")
        .replace("#161b22", "#eaeef2")
)
