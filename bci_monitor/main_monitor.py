# bci_monitor/main_monitor.py
import sys, traceback, json
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtSvg import QSvgRenderer
import styles
import utils.config_manager as cfg
import utils.serial_backend   as sb
from bci_trainer.panels.plot_panel import PlotPanel

class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trainer Settings")
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)
        self.fields = {}
        try:
            settings = json.load(open("config.json"))
        except:
            settings = {}
        for k, v in settings.items():
            e = QtWidgets.QLineEdit(str(v))
            self.fields[k] = e
            layout.addRow(k, e)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def accept(self):
        new = {}
        for k, e in self.fields.items():
            t = e.text().strip()
            new[k] = int(t) if t.isdigit() else (float(t) if '.' in t else t)
        json.dump(new, open("config.json", "w"), indent=2)
        cfg.reload(new)
        super().accept()

def tinted_icon(svg_path: str, size: QtCore.QSize, color: QtGui.QColor) -> QtGui.QIcon:
    """Render an SVG at `size` and tint it to `color`."""
    renderer = QSvgRenderer(svg_path)
    pix = QtGui.QPixmap(size)
    pix.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pix)
    renderer.render(painter)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
    painter.fillRect(pix.rect(), color)
    painter.end()
    return QtGui.QIcon(pix)

class MonitorWindow(QtWidgets.QMainWindow):
    ICON_SIZE = QtCore.QSize(32, 32)

    def __init__(self, launcher=None, serial=None):
        super().__init__()
        self.launcher   = launcher
        self.serial     = serial or sb.DummySerial()
        self.dark_theme = True
        self.panels     = []
        self.setWindowTitle("EEG Monitor")
        self._build_ui()
        self._start_timer()
        self.apply_theme()

    def _build_ui(self):
        tb = QtWidgets.QToolBar(movable=False)
        tb.setIconSize(self.ICON_SIZE)
        self.addToolBar(tb)

        def mk(svg, tip, cb):
            btn = QtWidgets.QToolButton()
            color = QtGui.QColor("#e6edf3") if self.dark_theme else QtGui.QColor("#0d1117")
            icon = tinted_icon(svg, self.ICON_SIZE, color)
            btn.setIcon(icon)
            btn.setIconSize(self.ICON_SIZE)
            btn.setFixedSize(self.ICON_SIZE + QtCore.QSize(8,8))
            btn.setToolTip(tip)
            btn.clicked.connect(cb)
            btn.setProperty("svg_path", svg)
            return btn

        # Add panel on the left
        tb.addWidget(mk("icons/add.svg", "Add Panel", self._add_panel))
        # spacer
        sp = QtWidgets.QWidget()
        sp.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(sp)
        # right‑side
        tb.addWidget(mk("icons/return.svg",   "Main Menu", lambda: self.launcher.stack.setCurrentIndex(0)))
        tb.addWidget(mk("icons/theme.svg",    "Toggle Theme", self._toggle_theme))
        tb.addWidget(mk("icons/settings.svg", "Settings", self._open_prefs))

        # two initial panels
        self._create_panel("Waveform", QtCore.Qt.LeftDockWidgetArea)
        self._create_panel("Spectrum", QtCore.Qt.RightDockWidgetArea)

    def _create_panel(self, kind, area):
        dock = QtWidgets.QDockWidget(kind, self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )
        panel = PlotPanel(self.serial, kind)
        dock.setWidget(panel)
        self.addDockWidget(area, dock)
        self.panels.append(dock)

    def _add_panel(self):
        if len(self.panels) >= 4:
            QtWidgets.QMessageBox.warning(self, "Limit", "Max 4 panels")
            return
        kinds = ["Waveform","Spectrum","8 Graphs","BandPower"]
        kind, ok = QtWidgets.QInputDialog.getItem(self, "Add Panel", "Choose type:", kinds, 0, False)
        if not ok: return
        area = [
            QtCore.Qt.LeftDockWidgetArea,
            QtCore.Qt.RightDockWidgetArea,
            QtCore.Qt.TopDockWidgetArea,
            QtCore.Qt.BottomDockWidgetArea
        ][len(self.panels)]
        self._create_panel(kind, area)

    def _start_timer(self):
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self):
        try:
            for dock in self.panels:
                panel = dock.widget()
                if hasattr(panel, "update_panel"):
                    panel.update_panel()
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)

    def _toggle_theme(self):
        self.dark_theme = not self.dark_theme
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(styles.DARK if self.dark_theme else styles.LIGHT)
        # recolor toolbar icons
        for btn in self.findChildren(QtWidgets.QToolButton):
            svg = btn.property("svg_path")
            if not svg: continue
            color = QtGui.QColor("#e6edf3") if self.dark_theme else QtGui.QColor("#0d1117")
            btn.setIcon(tinted_icon(svg, self.ICON_SIZE, color))

        # recolor plot backgrounds
        for dock in self.panels:
            panel = dock.widget()
            if hasattr(panel, "set_plot_theme"):
                panel.set_plot_theme(self.dark_theme)

    def _open_prefs(self):
        dlg = ConfigDialog(self)
        if dlg.exec_():
            self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))

    def closeEvent(self, e):
        if isinstance(self.serial, sb.SerialThread):
            self.serial.stop()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MonitorWindow()
    w.show()
    sys.exit(app.exec_())
