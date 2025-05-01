# bci_trainer/main_trainer.py
import json, traceback
from PyQt5 import QtWidgets, QtCore

import styles as st, utils.config_manager as cfg, utils.serial_backend as sb
from bci_trainer.panels.plot_panel          import PlotPanel
from bci_trainer.panels.class_manager       import ClassManager
from bci_trainer.panels.eeg_recorder_widget import EEGRecorderWidget
from bci_trainer.panels.ai_training_widget  import AITrainingWidget
from bci_trainer.panels.ai_inference_widget import AIInferenceWidget
from utils.theme_manager import ThemeManager
from utils.ui_helpers   import nav_bar
_tm = ThemeManager.instance()

# ────────────────────────────────────────────────────────── settings dlg
class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Trainer Settings")
        form = QtWidgets.QFormLayout(self); self.fields = {}
        try: settings = json.load(open("utils/config.json"))
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
class TrainerWindow(QtWidgets.QMainWindow):
    ICON_SIZE = QtCore.QSize(32, 32)

    def __init__(self, launcher=None, serial=None):
        super().__init__()
        self.launcher = launcher
        self.serial   = serial or sb.DummySerial()

        self.setWindowTitle("BCI Trainer")
        self._build_ui(); self._start_timer()
        _tm.themeChanged.connect(self._after_theme_flip)
        self._after_theme_flip(_tm.is_dark)

    # ---------------------------- hot-swap entry ----------------------
    def set_serial(self, ser):
        self.serial = ser
        self.plot.set_serial(ser)
        self.recorder.set_serial(ser)
        self.infer.set_serial(ser)

    # ---------------------------- UI ---------------------------------
    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.plot     = PlotPanel(self.serial, "Waveform")
        self.cls_mgr  = ClassManager()
        self.recorder = EEGRecorderWidget(self.serial)
        self.trainer  = AITrainingWidget()
        self.infer    = AIInferenceWidget(self.serial)
        for w, name in [(self.plot, "Waveform"), (self.cls_mgr, "Classes"),
                        (self.recorder, "Recorder"), (self.trainer, "Trainer"),
                        (self.infer, "Inference")]:
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(self._on_tab_change)

        self.tabs.setCornerWidget(
            nav_bar(self,
                    lambda: self.launcher.stack.setCurrentIndex(0),
                    self._open_config,
                    on_reconnect=self.launcher.reconnect_serial),
            QtCore.Qt.TopRightCorner)
        self.setCentralWidget(self.tabs)

    # ---------------------------- runtime -----------------------------
    def _on_tab_change(self, idx):
        if self.tabs.widget(idx) is self.recorder:
            self.recorder.load_classes_from_file()

    def _start_timer(self):
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        self._timer.timeout.connect(lambda: self.plot.update_panel())
        self._timer.start()

    # ---------------------------- theme -------------------------------
    def _after_theme_flip(self, dark):
        corner = self.tabs.cornerWidget(QtCore.Qt.TopRightCorner)
        if corner:
            for btn in corner.findChildren(QtWidgets.QToolButton):
                svg = btn.property("svg_path")
                if svg: btn.setIcon(_tm.tinted_icon(svg, self.ICON_SIZE))

    # ---------------------------- misc --------------------------------
    def _open_config(self):
        dlg = ConfigDialog(self)
        if dlg.exec_():
            self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
            for p in (self.plot, self.cls_mgr, self.recorder,
                      self.trainer, self.infer):
                getattr(p, "update_panel", lambda: None)()

    def closeEvent(self, e):
        if isinstance(self.serial, sb.SerialThread): self.serial.stop()
        super().closeEvent(e)
