# bci_monitor/main_monitor.py
import sys, traceback, json
from PyQt5 import QtWidgets, QtCore

import styles, utils.config_manager as cfg, utils.serial_backend as sb
from bci_trainer.panels.plot_panel import PlotPanel
from utils.theme_manager import ThemeManager
from utils.ui_helpers   import nav_bar
_tm = ThemeManager.instance()

# ────────────────────────────────────────────────────────── helpers
def _svg_btn(svg, tip, cb, size):
    btn = QtWidgets.QToolButton()
    btn.setFixedSize(size + QtCore.QSize(8, 8)); btn.setIconSize(size)
    btn.setToolTip(tip); btn.clicked.connect(cb)
    btn.setProperty("svg_path", svg)
    btn.setIcon(_tm.tinted_icon(svg, size))
    return btn

# ────────────────────────────────────────────────────────── settings dlg
class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Monitor Settings")
        form = QtWidgets.QFormLayout(self); self.fields = {}
        try: settings = json.load(open("config.json"))
        except Exception: settings = {}
        for k, v in settings.items():
            e = QtWidgets.QLineEdit(str(v)); self.fields[k] = e; form.addRow(k, e)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addWidget(bb)

    def accept(self):
        new = {}
        for k, e in self.fields.items():
            t = e.text().strip()
            new[k] = int(t) if t.isdigit() else float(t) if "." in t else t
        json.dump(new, open("config.json", "w"), indent=2); cfg.reload(new)
        super().accept()

# ────────────────────────────────────────────────────────── main window
class MonitorWindow(QtWidgets.QMainWindow):
    ICON_SIZE = QtCore.QSize(32, 32)

    def __init__(self, launcher=None, serial=None):
        super().__init__()
        self.launcher = launcher
        self.serial   = serial or sb.DummySerial()
        self.panels   = []

        self.setWindowTitle("EEG Monitor")
        self._build_ui(); self._start_timer()
        _tm.themeChanged.connect(self._after_theme_flip)
        self._after_theme_flip(_tm.is_dark)

    # ---------------------------- hot-swap entry ----------------------
    def set_serial(self, ser):
        self.serial = ser
        for dock in self.panels:
            panel = dock.widget()
            if hasattr(panel, "set_serial"):
                panel.set_serial(ser)

    # ---------------------------- UI ---------------------------------
    def _build_ui(self):
        tb = QtWidgets.QToolBar(movable=False); tb.setIconSize(self.ICON_SIZE)
        self.addToolBar(tb)
        tb.addWidget(_svg_btn("icons/add.svg", "Add panel", self._add_panel,
                              self.ICON_SIZE))

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)

        tb.addWidget(nav_bar(self,
                             lambda: self.launcher.stack.setCurrentIndex(0),
                             self._open_prefs,
                             on_reconnect=self.launcher.reconnect_serial))

        self._create_panel("Waveform", QtCore.Qt.LeftDockWidgetArea)
        self._create_panel("Spectrum", QtCore.Qt.RightDockWidgetArea)

    def _create_panel(self, kind, area):
        dock = QtWidgets.QDockWidget(kind, self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable   |
                         QtWidgets.QDockWidget.DockWidgetFloatable)
        panel = PlotPanel(self.serial, kind)
        dock.setWidget(panel); self.addDockWidget(area, dock)
        self.panels.append(dock)

    # ---------------------------- runtime -----------------------------
    def _add_panel(self):
        if len(self.panels) >= 4:
            QtWidgets.QMessageBox.warning(self, "Limit", "Max 4 panels"); return
        kinds = ["Waveform", "Spectrum", "8 Graphs", "BandPower"]
        kind, ok = QtWidgets.QInputDialog.getItem(self, "Add Panel", "Choose type:",
                                                  kinds, 0, False)
        if not ok: return
        area = [QtCore.Qt.LeftDockWidgetArea, QtCore.Qt.RightDockWidgetArea,
                QtCore.Qt.TopDockWidgetArea,  QtCore.Qt.BottomDockWidgetArea][len(self.panels)]
        self._create_panel(kind, area)

    def _start_timer(self):
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        self._timer.timeout.connect(self._refresh); self._timer.start()

    def _refresh(self):
        try:
            for dock in self.panels:
                panel = dock.widget()
                if hasattr(panel, "update_panel"): panel.update_panel()
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)

    # ---------------------------- theme tweaks ------------------------
    def _after_theme_flip(self, dark):
        for btn in self.findChildren(QtWidgets.QToolButton):
            svg = btn.property("svg_path")
            if svg: btn.setIcon(_tm.tinted_icon(svg, self.ICON_SIZE))
        for dock in self.panels:
            panel = dock.widget()
            if hasattr(panel, "set_plot_theme"): panel.set_plot_theme(dark)

    # ---------------------------- misc --------------------------------
    def _open_prefs(self):
        dlg = ConfigDialog(self)
        if dlg.exec_(): self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))

    def closeEvent(self, e):
        if isinstance(self.serial, sb.SerialThread): self.serial.stop()
        super().closeEvent(e)
