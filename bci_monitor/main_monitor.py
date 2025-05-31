# bci_monitor/main_monitor.py

import json
import re
import traceback

import numpy as np
import pyqtgraph as pg
from scipy.signal import welch

from PyQt5 import QtWidgets, QtCore

import utils.config_manager as cfg
import utils.serial_backend as sb
from utils.theme_manager import ThemeManager
from utils.ui_helpers   import nav_bar

# ─────────────────────────────────────────────────────────────────────────────
# Constants for channel names and colors
CHANNEL_NAMES  = [f"CH{i}" for i in range(1, 10)]
CHANNEL_COLORS = ['#F00', '#0F0', '#00F', '#0FF', '#F0F', '#FF0', '#FA0', '#0AF', '#A0F']

# ─────────────────────────────────────────────────────────────────────────────
# Compute bandpower for a given frequency band
def compute_bandpower(data, sf, band, window_sec=None, relative=False):
    band = np.asarray(band)
    nperseg = int(window_sec * sf) if window_sec is not None else min(256, len(data))
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = (freqs >= band[0]) & (freqs <= band[1])
    bp = np.trapz(psd[idx_band], dx=freq_res)
    if relative:
        bp /= np.trapz(psd, dx=freq_res)
    return bp

# ─────────────────────────────────────────────────────────────────────────────
# Inline PlotPanel
class PlotPanel(QtWidgets.QWidget):
    """
    kind = Waveform | Spectrum | Eight | BandPower
    """
    def __init__(self, serial, kind="Waveform", parent=None):
        super().__init__(parent)
        self.serial = serial
        self.kind   = kind

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ─── Controls ─────────────────────────────────────────────
        top = QtWidgets.QHBoxLayout()
        lay.addLayout(top)

        self.kind_combo = QtWidgets.QComboBox()
        self.kind_combo.addItems(["Waveform", "Spectrum", "Eight", "BandPower"])
        self.kind_combo.setCurrentText(self.kind)
        self.kind_combo.currentTextChanged.connect(self._on_kind_change)
        top.addWidget(self.kind_combo)

        self.all_btn = QtWidgets.QPushButton("All")
        self.all_btn.setCheckable(True)
        self.all_btn.setChecked(True)
        self.all_btn.toggled.connect(self._all_toggle)
        top.addWidget(self.all_btn)

        self.chk = []
        for i,name in enumerate(CHANNEL_NAMES):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet(f"color:{CHANNEL_COLORS[i]};")
            self.chk.append(cb)
            top.addWidget(cb)

        # ─── Plot area ───────────────────────────────────────────
        self.plot_container = QtWidgets.QWidget()
        self.plot_layout    = QtWidgets.QVBoxLayout(self.plot_container)
        lay.addWidget(self.plot_container)

        self._init_plot_area()

    def _clear_plot_area(self):
        def clear(lay):
            while lay.count():
                item = lay.takeAt(0)
                if w := item.widget():
                    w.setParent(None); w.deleteLater()
                if sub := item.layout():
                    clear(sub)
        clear(self.plot_layout)

    def _init_plot_area(self):
        if self.kind in ("Waveform","Spectrum","BandPower"):
            self.pw = pg.PlotWidget()
            self.plot_layout.addWidget(self.pw)
            if self.kind != "BandPower":
                self.curves = [self.pw.plot(pen=CHANNEL_COLORS[i]) for i in range(9)]
        else:  # Eight panels
            grid = QtWidgets.QGridLayout()
            self.plot_layout.addLayout(grid)
            self.subplots = []
            for i in range(8):
                pw = pg.PlotWidget(title=CHANNEL_NAMES[i+1])
                grid.addWidget(pw, i, 0)
                self.subplots.append(pw)

    def _on_kind_change(self, k):
        self.kind = k
        self._clear_plot_area()
        self._init_plot_area()
        self.update_panel()

    def _actives(self):
        return [i for i,cb in enumerate(self.chk) if cb.isChecked()]

    def _all_toggle(self, c):
        for cb in self.chk:
            cb.setChecked(c)
        self.all_btn.setText("All" if c else "None")

    def update_panel(self):
        if   self.kind=="Waveform": self._waveform()
        elif self.kind=="Spectrum": self._spectrum()
        elif self.kind=="Eight":    self._eight()
        else:                       self._band()

    def _waveform(self):
        x,d = self.serial.get_plot_data()
        if d.size==0: return
        act = self._actives()
        for i,curve in enumerate(self.curves):
            curve.setData(x if i in act else [], d[i] if i in act else [])
        self.pw.setXRange(max(0,x[-1]-cfg.get("PLOT_LENGTH")), x[-1])

    def _spectrum(self):
        # 1) FFT size from config
        nfft = cfg.get("FFT_LENGTH")
        # 2) Pull exactly nfft samples
        _, d = self.serial.get_fft_data(length=nfft)
        if d.size == 0:
            return

        # 3) Build frequency axis
        fs = cfg.get("SAMPLE_RATE")
        freqs = np.fft.rfftfreq(nfft, 1/fs)

        # 4) Read fmin/fmax from your updated config
        fmin = cfg.get("FFT_FREQ_MIN")
        fmax = cfg.get("FFT_FREQ_MAX")
        # ensure min ≤ max
        if fmin > fmax:
            fmin, fmax = fmax, fmin

        mask = (freqs >= fmin) & (freqs <= fmax)

        # 5) Clear and replot
        self.pw.clear()
        self.curves = [self.pw.plot(pen=c) for c in CHANNEL_COLORS]

        for i in self._actives():
            fft = np.abs(np.fft.rfft(d[i], n=nfft))
            self.curves[i].setData(freqs[mask], fft[mask])

        # 6) Zoom the x-axis
        self.pw.setXRange(fmin, fmax)



    def _eight(self):
        x,d = self.serial.get_plot_data()
        if d.size==0: return
        for idx,pw in enumerate(self.subplots):
            ch = idx+1
            pw.clear()
            if self.chk[ch].isChecked():
                pw.plot(x, d[ch], pen=CHANNEL_COLORS[ch])
                pw.setXRange(max(0,x[-1]-cfg.get("PLOT_LENGTH")), x[-1])

    def _band(self):
        x,d = self.serial.get_plot_data(length=cfg.get("FFT_LENGTH"))
        if d.size==0: return
        act = self._actives()
        if not act:
            self.pw.clear()
            t = pg.TextItem("No channels selected", color='w', anchor=(0.5,0.5))
            t.setPos(0,0); self.pw.addItem(t)
            return
        comb = d[act,:].sum(axis=0)
        sf = cfg.get("SAMPLE_RATE")
        bands = {
          'Delta': (0.5,4), 'Theta':(4,8), 'Alpha':(8,12),
          'Beta1':(12,18),'Beta2':(18,30),'Gamma':(30,45)
        }
        pows = [ compute_bandpower(comb, sf, b) for b in bands.values() ]
        self.pw.clear()
        bg = pg.BarGraphItem(x=list(range(len(bands))), height=pows, width=0.6)
        self.pw.addItem(bg)
        ax = self.pw.getAxis('bottom')
        ax.setTicks([ [(i,n) for i,n in enumerate(bands.keys())] ])
        if pows: self.pw.setYRange(0, max(pows)*1.1)


# ─────────────────────────────────────────────────────────────────────────────
class MonitorWindow(QtWidgets.QMainWindow):
    ICON_SIZE = QtCore.QSize(32,32)

    def __init__(self, launcher=None, serial=None):
        super().__init__()
        self.launcher = launcher
        self.serial   = serial or sb.DummySerial()
        self.panels   = []

        self.setWindowTitle("EEG Monitor")
        self._build_ui()
        self._start_timer()

        _tm = ThemeManager.instance()
        _tm.themeChanged.connect(self._after_theme_flip)
        self._after_theme_flip(_tm.is_dark)

    def _build_ui(self):
        tb = QtWidgets.QToolBar(movable=False)
        tb.setIconSize(self.ICON_SIZE)
        self.addToolBar(tb)

        def _svg_btn(svg, tip, cb):
            btn = QtWidgets.QToolButton()
            btn.setFixedSize(self.ICON_SIZE + QtCore.QSize(8,8))
            btn.setIconSize(self.ICON_SIZE)
            btn.setToolTip(tip)
            btn.clicked.connect(cb)
            btn.setProperty("svg_path", svg)
            btn.setIcon(ThemeManager.instance().tinted_icon(svg, self.ICON_SIZE))
            return btn

        tb.addWidget(_svg_btn("icons/add.svg","Add panel",self._add_panel))
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addWidget(nav_bar(self,
                             lambda: self.launcher.stack.setCurrentIndex(0),
                             self._open_prefs,
                             on_reconnect=self.launcher.reconnect_serial))

        # ─ default layout ────────────────────────────────────────────
        left_dock   = self._create_panel("Eight",     QtCore.Qt.LeftDockWidgetArea)
        spec_dock   = self._create_panel("Spectrum",  QtCore.Qt.RightDockWidgetArea)
        bandp_dock  = self._create_panel("BandPower", QtCore.Qt.RightDockWidgetArea)

        # stack spectrum over bandpower
        self.splitDockWidget(spec_dock, bandp_dock, QtCore.Qt.Vertical)

    def _create_panel(self, kind, area):
        dock = QtWidgets.QDockWidget(kind, self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetMovable   |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )
        panel = PlotPanel(self.serial, kind)
        dock.setWidget(panel)
        self.addDockWidget(area, dock)
        self.panels.append(dock)
        return dock

    def _add_panel(self):
        if len(self.panels) >= 4:
            QtWidgets.QMessageBox.warning(self,"Limit","Max 4 panels")
            return
        kinds = ["Waveform","Spectrum","Eight","BandPower"]
        kind,ok = QtWidgets.QInputDialog.getItem(self,
                   "Add Panel","Choose type:",kinds,0,False)
        if not ok: return
        area = [QtCore.Qt.LeftDockWidgetArea,
                QtCore.Qt.RightDockWidgetArea,
                QtCore.Qt.TopDockWidgetArea,
                QtCore.Qt.BottomDockWidgetArea][len(self.panels)]
        dock = self._create_panel(kind, area)
        # if new panel is BandPower under Spectrum, stack them
        if kind=="BandPower":
            # find the spectrum dock
            for d in self.panels:
                if d.windowTitle()=="Spectrum":
                    self.splitDockWidget(d, dock, QtCore.Qt.Vertical)
                    break

    def _start_timer(self):
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self):
        try:
            for d in self.panels:
                p = d.widget()
                if hasattr(p,"update_panel"):
                    p.update_panel()
        except Exception as e:
            traceback.print_exception(type(e),e,e.__traceback__)

    def _after_theme_flip(self, dark):
        for btn in self.findChildren(QtWidgets.QToolButton):
            if svg:=btn.property("svg_path"):
                btn.setIcon(ThemeManager.instance().tinted_icon(svg,self.ICON_SIZE))
        for d in self.panels:
            p = d.widget()
            if hasattr(p,"set_plot_theme"):
                p.set_plot_theme(dark)

    def _open_prefs(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Monitor Settings")
        form = QtWidgets.QFormLayout(dlg)
        fields = {}
        try: settings = json.load(open("utils/config.json"))
        except: settings = {}
        for k,v in settings.items():
            e = QtWidgets.QLineEdit(str(v))
            fields[k]=e; form.addRow(k,e)
        bb = QtWidgets.QDialogButtonBox(
               QtWidgets.QDialogButtonBox.Save|
               QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(lambda:self._save_prefs(fields,dlg))
        bb.rejected.connect(dlg.reject)
        form.addWidget(bb)
        dlg.exec_()

    def _save_prefs(self, fields, dlg):
        new = {}
        for k,e in fields.items():
            t = e.text().strip()
            new[k] = int(t) if t.isdigit() else float(t) if "." in t else t
        json.dump(new,open("utils/config.json","w"),indent=2)
        cfg.reload(new)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        dlg.accept()

    def set_serial(self, ser):
        self.serial = ser
        for d in self.panels:
            w = d.widget()
            if hasattr(w,"set_serial"):
                w.set_serial(ser)

    def closeEvent(self, e):
        if isinstance(self.serial, sb.SerialThread):
            self.serial.stop()
        super().closeEvent(e)
