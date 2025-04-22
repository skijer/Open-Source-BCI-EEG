# bci_launcher.py
import sys, json
from PyQt5 import QtWidgets, QtCore
import styles
import utils.config_manager as cfg
import utils.serial_backend   as sb
from bci_trainer.main_trainer import TrainerWindow
from bci_monitor.main_monitor import MonitorWindow

class MainMenu(QtWidgets.QWidget):
    def __init__(self, launcher):
        super().__init__()
        self.launcher = launcher
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtCore.Qt.black)
        self.setPalette(pal)

        v = QtWidgets.QVBoxLayout(self)
        v.addStretch()
        btn_style = """
            QPushButton {
                font-size: 24px;
                color: white;
                background: #00d8ff;
                border-radius: 8px;
                min-height: 60px;
            }
            QPushButton:hover { background: #ff9a00; }
        """
        b1 = QtWidgets.QPushButton("▶️  Trainer")
        b1.setStyleSheet(btn_style)
        b1.clicked.connect(lambda: launcher.stack.setCurrentIndex(1))
        b2 = QtWidgets.QPushButton("▶️  Monitor")
        b2.setStyleSheet(btn_style)
        b2.clicked.connect(lambda: launcher.stack.setCurrentIndex(2))
        v.addWidget(b1)
        v.addWidget(b2)
        v.addStretch()

class Launcher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Launcher")
        self.resize(800, 600)

        # — open serial once —
        port = sb.find_usb()
        if port:
            self.serial = sb.SerialThread(port)
            self.serial.start()
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "No USB port found")
            self.serial = sb.DummySerial()

        # — stack of scenes —
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.menu    = MainMenu(self)
        self.trainer = TrainerWindow(launcher=self, serial=self.serial)
        self.monitor = MonitorWindow(launcher=self, serial=self.serial)

        self.stack.addWidget(self.menu)    # index 0
        self.stack.addWidget(self.trainer) # index 1
        self.stack.addWidget(self.monitor) # index 2

        self.apply_theme()
        self.stack.setCurrentIndex(0)

    def open_settings(self):
        dlg = self.trainer.ConfigDialog(self)
        if dlg.exec_():
            newcfg = json.load(open("config.json"))
            cfg.reload(newcfg)
            # update intervals
            self.trainer._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
            self.monitor._timer.setInterval(cfg.get("UPDATE_INTERVAL"))

    def apply_theme(self):
        self.setStyleSheet(styles.DARK)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Launcher()
    w.show()
    sys.exit(app.exec_())
