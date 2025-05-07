# main.py

import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtSvg import QSvgWidget

import styles, utils.serial_backend as sb
from utils.theme_manager import ThemeManager
from utils.ui_helpers   import nav_bar
from bci_trainer.main_trainer       import TrainerWindow
from bci_monitor.main_monitor       import MonitorWindow
from utils.camera_widget import CameraWidget
from utils.control_panel_widget     import ControlPanelWidget

_tm = ThemeManager.instance()

# ─────────────────────────────────────────────────────── main menu
class MainMenu(QtWidgets.QWidget):
    def __init__(self, launcher):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(20)

        # nav bar
        v.addWidget(
            nav_bar(self,
                    QtWidgets.QApplication.quit,
                    lambda: QtWidgets.QMessageBox.information(self, "Launcher", "No settings here"),
                    first_icon="icons/close.svg", first_tooltip="Exit",
                    on_reconnect=launcher.reconnect_serial),
            alignment=QtCore.Qt.AlignTop
        )

        # logo
        logo = QSvgWidget("icons/logo.svg")
        logo.setFixedSize(180, 180)
        v.addWidget(logo, alignment=QtCore.Qt.AlignHCenter)

        # title
        title = QtWidgets.QLabel("LEO BCI")
        title.setAlignment(QtCore.Qt.AlignHCenter)
        title.setStyleSheet(f"font-size:32px; font-weight:bold; color:{styles.NEON_BLUE};")
        v.addWidget(title)

        v.addStretch()

        # buttons
        btn_css = (
            "QPushButton { font-size:24px; color:%s; background:%s; "
            "border-radius:8px; min-height:60px; } "
            "QPushButton:hover { background:%s; }"
            % (styles.BG_DARK, styles.NEON_BLUE, styles.NEON_ORANGE)
        )
        for text, idx in [
            ("▶️ Trainer",    1),
            ("▶️ Monitor",    2),
            ("▶️ Eye Tracker",3),
            ("▶️ Control",    4)
        ]:
            b = QtWidgets.QPushButton(text)
            b.setStyleSheet(btn_css)
            b.clicked.connect(lambda _, i=idx: launcher.stack.setCurrentIndex(i))
            v.addWidget(b)

        v.addStretch()


# ─────────────────────────────────────────────────────── launcher
class Launcher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LEO BCI Launcher")
        self.resize(800, 600)
        self._open_serial(initial=True)

        self.stack   = QtWidgets.QStackedWidget()
        self.menu    = MainMenu(self)
        self.trainer = TrainerWindow(self, self.serial)
        self.monitor = MonitorWindow(self, self.serial)
        self.camera  = CameraWidget(self)
        self.control = ControlPanelWidget(self, self.serial)

        for w in (self.menu, self.trainer, self.monitor, self.camera, self.control):
            self.stack.addWidget(w)

        self.setCentralWidget(self.stack)
        self.stack.setCurrentIndex(0)

    # ------------------- serial open helper -------------------------
    def _open_serial(self, initial=False):
        port = sb.find_usb()
        if port:
            ser = sb.SerialThread(port)
            if ser.ok:
                ser.start()
                QtWidgets.QMessageBox.information(self, "Serial", f"Connected to {port}")
            else:
                QtWidgets.QMessageBox.critical(self, "Serial", f"Failed to open {port}")
                ser = sb.DummySerial()
        else:
            if initial:
                QtWidgets.QMessageBox.warning(self, "Serial",
                                              "No USB device found – using dummy EEG")
            ser = sb.DummySerial()
        self.serial = ser

    # ------------------- reconnect slot -----------------------------
    def reconnect_serial(self):
        if isinstance(self.serial, sb.SerialThread) and self.serial.ok:
            QtWidgets.QMessageBox.information(self, "Serial", "Already connected.")
            return
        if isinstance(self.serial, sb.SerialThread):
            self.serial.stop()
        self._open_serial()
        for w in (self.trainer, self.monitor):
            w.set_serial(self.serial)


# ─────────────────────────────────────────────────────── entry point
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Launcher().show()
    sys.exit(app.exec_())
