# bci_trainer/main_trainer.py
import sys, traceback, json
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtSvg import QSvgRenderer
import styles as st
import utils.config_manager as cfg
import utils.serial_backend   as sb

from bci_trainer.panels.plot_panel          import PlotPanel
from bci_trainer.panels.class_manager       import ClassManager
from bci_trainer.panels.eeg_recorder_widget import EEGRecorderWidget
from bci_trainer.panels.ai_training_widget  import AITrainingWidget
from bci_trainer.panels.ai_inference_widget import AIInferenceWidget

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

class TrainerWindow(QtWidgets.QMainWindow):
    ICON_SIZE = QtCore.QSize(32, 32)

    def __init__(self, launcher=None, serial=None):
        super().__init__()
        self.launcher   = launcher
        self.serial     = serial or sb.DummySerial()
        self.dark_theme = True
        self.setWindowTitle("BCI Trainer")
        self._build_ui()
        self._start_timer()
        self.apply_theme()

    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.plot     = PlotPanel(self.serial, "Waveform")
        self.cls_mgr  = ClassManager()
        self.recorder = EEGRecorderWidget(self.serial)
        self.trainer  = AITrainingWidget()
        self.infer    = AIInferenceWidget(self.serial)

        for w, name in [
            (self.plot,     "Waveform"),
            (self.cls_mgr,  "Classes"),
            (self.recorder, "Recorder"),
            (self.trainer,  "Trainer"),
            (self.infer,    "Inference")
        ]:
            self.tabs.addTab(w, name)
        self.tabs.currentChanged.connect(self._on_tab_change)

        corner = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(corner)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        def mk(svg, tip, cb):
            btn = QtWidgets.QToolButton()
            color = QtGui.QColor("#e6edf3") if self.dark_theme else QtGui.QColor("#0d1117")
            icon = tinted_icon(svg, self.ICON_SIZE, color)
            btn.setIcon(icon)
            btn.setIconSize(self.ICON_SIZE)
            btn.setFixedSize(self.ICON_SIZE + QtCore.QSize(8, 8))
            btn.setToolTip(tip)
            btn.clicked.connect(cb)
            btn.setProperty("svg_path", svg)
            return btn

        h.addWidget(mk("icons/return.svg",   "Main Menu",        lambda: self.launcher.stack.setCurrentIndex(0)))
        h.addWidget(mk("icons/theme.svg",    "Toggle Theme",     self._toggle_theme))
        h.addWidget(mk("icons/settings.svg", "Settings",         self._open_config))

        self.tabs.setCornerWidget(corner, QtCore.Qt.TopRightCorner)
        self.setCentralWidget(self.tabs)

    def _on_tab_change(self, idx):
        if self.tabs.widget(idx) is self.recorder:
            self.recorder.load_classes_from_file()

    def _start_timer(self):
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        self._timer.timeout.connect(lambda: self.plot.update_panel())
        self._timer.start()

    def _toggle_theme(self):
        self.dark_theme = not self.dark_theme
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(st.DARK if self.dark_theme else st.LIGHT)
        # recolor corner icons
        corner = self.tabs.cornerWidget(QtCore.Qt.TopRightCorner)
        if corner:
            for btn in corner.findChildren(QtWidgets.QToolButton):
                svg = btn.property("svg_path")
                color = QtGui.QColor("#e6edf3") if self.dark_theme else QtGui.QColor("#0d1117")
                btn.setIcon(tinted_icon(svg, self.ICON_SIZE, color))

    def _open_config(self):
        dlg = ConfigDialog(self)
        if dlg.exec_():
            self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
            for p in (self.plot, self.cls_mgr, self.recorder, self.trainer, self.infer):
                getattr(p, "update_panel", lambda: None)()

    def closeEvent(self, e):
        if isinstance(self.serial, sb.SerialThread):
            self.serial.stop()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = TrainerWindow()
    w.show()
    sys.exit(app.exec_())
