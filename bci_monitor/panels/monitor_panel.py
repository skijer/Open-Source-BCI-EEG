# bci_monitor/panels/monitor_panel.py
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import utils.config_manager as cfg

class MonitorPanel(QtWidgets.QWidget):
    def __init__(self, serial, parent=None):
        super().__init__(parent)
        self.serial = serial

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.plot1 = pg.PlotWidget(title="Channel 1")
        self.plot2 = pg.PlotWidget(title="Channel 2–9 sum")
        splitter.addWidget(self.plot1)
        splitter.addWidget(self.plot2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)

    def update_panel(self):
        x, d = self.serial.get_plot_data()
        if d.size==0: return
        # example: show channel 1 vs sum of others
        self.plot1.setData(x, d[0])
        combo = d[1:].sum(axis=0)
        self.plot2.setData(x, combo)
