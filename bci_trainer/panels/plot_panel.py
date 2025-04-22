# panels/plot_panel.py
from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np
import utils.config_manager as cfg
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch, welch

CHANNEL_NAMES  = [f"CH{i}" for i in range(1, 10)]
CHANNEL_COLORS = ['#F00', '#0F0', '#00F', '#0FF', '#F0F', '#FF0', '#FA0', '#0AF', '#A0F']

def compute_bandpower(data, sf, band, window_sec=None, relative=False):
    band = np.asarray(band)
    nperseg = int(window_sec * sf) if window_sec is not None else min(256, len(data))
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    bp = np.trapz(psd[idx_band], dx=freq_res)
    if relative:
        bp /= np.trapz(psd, dx=freq_res)
    return bp

class PlotPanel(QtWidgets.QWidget):
    """
    kind = Waveform | Spectrum | Eight | BandPower
    """
    def __init__(self, serial, kind="Waveform", parent=None):
        super().__init__(parent)
        self.serial = serial
        self.kind   = kind

        # Layout principal
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ─── barra de controles ───────────────────────────────────────
        top = QtWidgets.QHBoxLayout()
        lay.addLayout(top)

        # ➤ ComboBox de tipo de gráfico
        self.kind_combo = QtWidgets.QComboBox()
        self.kind_combo.addItems(["Waveform", "Spectrum", "Eight", "BandPower"])
        self.kind_combo.setCurrentText(self.kind)
        self.kind_combo.currentTextChanged.connect(self._on_kind_change)
        top.addWidget(self.kind_combo)

        # ➤ Botón All/None + checkboxes de canales
        self.all_btn = QtWidgets.QPushButton("All")
        self.all_btn.setCheckable(True)
        self.all_btn.setChecked(True)
        self.all_btn.toggled.connect(self._all_toggle)
        top.addWidget(self.all_btn)

        self.chk = []
        for i, name in enumerate(CHANNEL_NAMES):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {CHANNEL_COLORS[i]};")
            self.chk.append(cb)
            top.addWidget(cb)

        # ─── contenedor del área de gráficos ───────────────────────────
        self.plot_container = QtWidgets.QWidget()
        self.plot_layout    = QtWidgets.QVBoxLayout(self.plot_container)
        lay.addWidget(self.plot_container)

        # Primera vez: init del área de gráficos
        self._init_plot_area()

    def _clear_plot_area(self):
        """Elimina recursivamente widgets y layouts dentro de plot_layout."""
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()
                sublayout = item.layout()
                if sublayout:
                    clear_layout(sublayout)
        clear_layout(self.plot_layout)

    def _init_plot_area(self):
        """Construye el PlotWidget(s) según self.kind."""
        k = self.kind
        if k in ("Waveform", "Spectrum", "BandPower"):
            self.pw = pg.PlotWidget()
            self.plot_layout.addWidget(self.pw)
            if k != "BandPower":
                # Una curva por canal
                self.curves = [
                    self.pw.plot(pen=CHANNEL_COLORS[i]) for i in range(9)
                ]
        else:  # Eight
            grid = QtWidgets.QGridLayout()
            self.plot_layout.addLayout(grid)
            self.subplots = []
            idx = 1
            for r in range(2):
                for c in range(4):
                    pw = pg.PlotWidget()
                    pw.setTitle(CHANNEL_NAMES[idx])
                    grid.addWidget(pw, r, c)
                    self.subplots.append(pw)
                    idx += 1

    def _on_kind_change(self, new_kind):
        """Cuando cambia la ComboBox, reconstruye el área de gráficos y repinta."""
        self.kind = new_kind
        self._clear_plot_area()
        self._init_plot_area()
        self.update_panel()  # fuerza un primer dibujo inmediato

    def _actives(self):
        return [i for i, cb in enumerate(self.chk) if cb.isChecked()]

    def _all_toggle(self, check):
        for cb in self.chk:
            cb.setChecked(check)
        self.all_btn.setText("All" if check else "None")

    def update_panel(self):
        if   self.kind == "Waveform":  self._waveform()
        elif self.kind == "Spectrum":  self._spectrum()
        elif self.kind == "Eight":     self._eight()
        else:                          self._band()

    # ─── Waveform ───────────────────────────────────────────────────
    def _waveform(self):
        x, d = self.serial.get_plot_data()
        if d.size == 0:
            return
        act = self._actives()
        for i in range(9):
            if i in act:
                self.curves[i].setData(x, d[i])
            else:
                self.curves[i].setData([], [])
        self.pw.setXRange(max(0, x[-1] - cfg.get("PLOT_LENGTH")), x[-1])

    # ─── FFT / Spectrum ─────────────────────────────────────────────
    def _spectrum(self):
        x, d = self.serial.get_fft_data()
        fs = cfg.get("SAMPLE_RATE")
        if d.size == 0:
            return
        self.pw.clear()
        self.curves = []
        for col in CHANNEL_COLORS:
            self.curves.append(self.pw.plot(pen=col))
        act = self._actives()
        for i in act:
            sig   = d[i]
            fft   = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(len(sig), 1/fs)
            mask  = (freqs >= 3) & (freqs <= 50)
            self.curves[i].setData(freqs[mask], fft[mask])

    # ─── 8 gráficas ────────────────────────────────────────────────
    def _eight(self):
        x, d = self.serial.get_plot_data()
        if d.size == 0:
            return
        for i, p in enumerate(self.subplots, start=1):
            p.clear()
            if self.chk[i].isChecked():
                p.plot(x, d[i], pen=CHANNEL_COLORS[i])
                p.setXRange(max(0, x[-1] - cfg.get("PLOT_LENGTH")), x[-1])

    # ─── Potencia en bandas ─────────────────────────────────────────
    def _band(self):
        x, data = self.serial.get_plot_data(length=1024)
        if data.size == 0:
            return
        active = self._actives()
        if not active:
            self.pw.clear()
            txt = pg.TextItem("No channels selected", color='w', anchor=(0.5, 0.5))
            txt.setPos(0, 0)
            self.pw.addItem(txt)
            return
        combined = np.sum(data[active, :], axis=0)
        sf = cfg.get("SAMPLE_RATE")
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta1': (12, 18),
            'Beta2': (18, 30),
            'Gamma': (30, 45)
        }
        band_powers = {
            name: compute_bandpower(combined, sf, frange)
            for name, frange in bands.items()
        }
        self.pw.clear()
        names  = list(band_powers.keys())
        powers = [band_powers[n] for n in names]
        x_axis = np.arange(len(names))
        bg = pg.BarGraphItem(x=x_axis, height=powers, width=0.6)
        self.pw.addItem(bg)
        ax = self.pw.getAxis('bottom')
        ax.setTicks([[(i, names[i]) for i in range(len(names))]])
        if powers:
            self.pw.setYRange(0, max(powers) * 1.1)
