"""
main_monitor.py  —  Monitor de EEG con PyQt5 + PyQtGraph
========================================================
• Eje X en segundos reales (no índices de muestra)
• Reglas verticales (rulers) cada 1 s, dibujadas eficientemente
• Y-axis oculto; potencia Vrms mostrada esquina inferior-derecha
• Soporta hasta 4 paneles (Waveform, Spectrum, Eight, BandPower)
"""

from __future__ import annotations

import json
import traceback
from typing import List, Tuple

import numpy as np
import pyqtgraph as pg
from scipy.signal import welch

from PyQt5 import QtCore, QtWidgets

import utils.config_manager as cfg
import utils.serial_backend as sb
from utils.theme_manager import ThemeManager
from utils.ui_helpers import nav_bar

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ----------------------------------------------------------------------------
CHANNEL_NAMES: List[str] = [f"CH{i}" for i in range(1, 10)]
CHANNEL_COLORS: List[str] = [
    "#F00", "#0F0", "#00F", "#0FF", "#F0F",
    "#FF0", "#FA0", "#0AF", "#A0F",
]

# ──────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ----------------------------------------------------------------------------
def compute_bandpower(
    signal: np.ndarray,
    fs: int,
    band: Tuple[float, float],
    *,
    window_sec: float | None = None,
    relative: bool = False,
) -> float:
    """Potencia de banda absoluta (o relativa) en *band* (Hz)."""
    band = np.asarray(band)
    nperseg = int(window_sec * fs) if window_sec else min(256, len(signal))
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    dx = freqs[1] - freqs[0]
    mask = (freqs >= band[0]) & (freqs <= band[1])
    val = np.trapz(psd[mask], dx=dx)
    return val / np.trapz(psd, dx=dx) if relative else val


def vrms(sig: np.ndarray) -> float:
    return np.sqrt(np.mean(sig ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Clase PlotPanel
# ----------------------------------------------------------------------------
class PlotPanel(QtWidgets.QWidget):
    """Panel capaz de mostrar Waveform, Spectrum, Eight (8 subplots) o BandPower"""

    MODES = ("Waveform", "Spectrum", "Eight", "BandPower")

    def __init__(self, serial, mode: str = "Waveform", parent=None) -> None:
        super().__init__(parent)
        self.serial = serial
        self.mode   = mode

        # Holders internos
        self.curves: List[pg.PlotDataItem] = []
        self.subplot_labels: List[pg.TextItem] = []
        self.vrms_label: pg.TextItem | None = None

        # ─ Layout raíz
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ---- Barra de controles
        ctr = QtWidgets.QHBoxLayout()
        root.addLayout(ctr)

        self.mode_cb = QtWidgets.QComboBox()
        self.mode_cb.addItems(self.MODES)
        self.mode_cb.setCurrentText(mode)
        self.mode_cb.currentTextChanged.connect(self._on_mode_change)
        ctr.addWidget(self.mode_cb)

        self.all_btn = QtWidgets.QPushButton("All")
        self.all_btn.setCheckable(True)
        self.all_btn.setChecked(True)
        self.all_btn.toggled.connect(self._toggle_all)
        ctr.addWidget(self.all_btn)

        self.chks: List[QtWidgets.QCheckBox] = []
        for i, name in enumerate(CHANNEL_NAMES):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet(f"color:{CHANNEL_COLORS[i]};")
            self.chks.append(cb)
            ctr.addWidget(cb)

        # ---- Contenedor de plots
        self.plot_container = QtWidgets.QWidget()
        root.addWidget(self.plot_container)
        self.plot_stack = QtWidgets.QVBoxLayout(self.plot_container)

        self._init_plots()

    # ─────────────────────────────────────────────
    # Helpers de UI
    # --------------------------------------------
    def _toggle_all(self, state: bool):
        for cb in self.chks:
            cb.setChecked(state)
        self.all_btn.setText("All" if state else "None")

    def _on_mode_change(self, mode: str):
        self.mode = mode
        self._clear_plots()
        self._init_plots()
        self.refresh()

    def _active_idx(self) -> List[int]:
        return [i for i, cb in enumerate(self.chks) if cb.isChecked()]

    # ---- creación / limpieza de widgets de plot
    def _clear_plots(self):
        def rec(layout: QtWidgets.QLayout):
            while layout.count():
                itm = layout.takeAt(0)
                if w := itm.widget():
                    w.setParent(None); w.deleteLater()
                if sub := itm.layout():
                    rec(sub)
        rec(self.plot_stack)
        self.curves.clear()
        self.subplot_labels.clear()
        self.vrms_label = None

    def _init_plots(self):
        """Crea los widgets según self.mode"""
        if self.mode in ("Waveform", "Spectrum", "BandPower"):
            self.plot = pg.PlotWidget()
            self.plot_stack.addWidget(self.plot)
            self.plot.hideAxis("left")

            if self.mode != "BandPower":
                self.curves = [self.plot.plot(pen=CHANNEL_COLORS[i]) for i in range(9)]

            self.vrms_label = pg.TextItem(anchor=(1, 1))
            self.plot.addItem(self.vrms_label)

        else:  # Eight
            grid = QtWidgets.QGridLayout()
            self.plot_stack.addLayout(grid)
            self.subplots = []
            for i in range(8):
                pw = pg.PlotWidget(title=CHANNEL_NAMES[i + 1])
                pw.hideAxis("left")
                lbl = pg.TextItem(anchor=(1, 1))
                pw.addItem(lbl)

                grid.addWidget(pw, i, 0)
                self.subplots.append(pw)
                self.subplot_labels.append(lbl)

    # ─────────────────────────────────────────────
    # Dispatcher de actualización
    # --------------------------------------------
    def refresh(self):
        if self.mode == "Waveform":
            self._waveform()
        elif self.mode == "Spectrum":
            self._spectrum()
        elif self.mode == "Eight":
            self._eight()
        else:
            self._bandpower()

    # ---- Waveform
    def _waveform(self):
        t, d = self.serial.get_plot_data()
        if d.size == 0:
            return
        act = self._active_idx()

        for i, curve in enumerate(self.curves):
            if i in act:
                curve.setData(t, d[i])
            else:
                curve.clear()

        # Reglas y ventana
        self._draw_rulers(self.plot, t)
        window = cfg.get("PLOT_LENGTH") / cfg.get("SAMPLE_RATE")
        self.plot.setXRange(max(0.0, t[-1] - window), t[-1])

        # Vrms
        text = "<br>".join(
            f"<span style='color:{CHANNEL_COLORS[i]}'>{CHANNEL_NAMES[i]}</span>: {vrms(d[i]):.2f} µVrms"
            for i in act
        )
        self.vrms_label.setText(text)
        self._anchor_bottom_right(self.plot, self.vrms_label)

    # ---- Spectrum
    def _spectrum(self):
        nfft = cfg.get("FFT_LENGTH")
        _, d = self.serial.get_fft_data(length=nfft)
        if d.size == 0:
            return

        fs = cfg.get("SAMPLE_RATE")
        freqs = np.fft.rfftfreq(nfft, d=1 / fs)
        fmin, fmax = cfg.get("FFT_FREQ_MIN"), cfg.get("FFT_FREQ_MAX")
        if fmin > fmax:
            fmin, fmax = fmax, fmin
        mask = (freqs >= fmin) & (freqs <= fmax)

        self.plot.clear()
        self.curves = [self.plot.plot(pen=p) for p in CHANNEL_COLORS]
        for i in self._active_idx():
            spec = np.abs(np.fft.rfft(d[i], n=nfft))
            self.curves[i].setData(freqs[mask], spec[mask])
        self.plot.setXRange(fmin, fmax)

    # ---- Eight (8 subplots)
    def _eight(self):
        t, d = self.serial.get_plot_data()
        if d.size == 0:
            return

        window = cfg.get("PLOT_LENGTH") / cfg.get("SAMPLE_RATE")
        act = self._active_idx()

        for k, pw in enumerate(self.subplots):
            ch = k + 1
            pw.clear()
            if ch in act:
                pw.plot(t, d[ch], pen=CHANNEL_COLORS[ch])
                pw.setXRange(max(0.0, t[-1] - window), t[-1])
                self._draw_rulers(pw, t)
                self.subplot_labels[k].setText(f"{vrms(d[ch]):.2f} µVrms")
                self._anchor_bottom_right(pw, self.subplot_labels[k])

    # ---- BandPower
    def _bandpower(self):
        _, d = self.serial.get_plot_data(length=cfg.get("FFT_LENGTH"))
        if d.size == 0:
            return

        act = self._active_idx()
        if not act:
            self.plot.clear()
            self.plot.addItem(pg.TextItem("No channels selected", anchor=(0.5, 0.5)))
            return

        comb = d[act, :].sum(axis=0)
        fs = cfg.get("SAMPLE_RATE")
        bands = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 12),
            "Beta1": (12, 18),
            "Beta2": (18, 30),
            "Gamma": (30, 45),
        }
        pws = [compute_bandpower(comb, fs, b) for b in bands.values()]

        self.plot.clear()
        self.plot.addItem(pg.BarGraphItem(x=list(range(len(bands))), height=pws, width=0.6))
        self.plot.getAxis("bottom").setTicks([[ (i, n) for i, n in enumerate(bands.keys()) ]])
        if pws:
            self.plot.setYRange(0, max(pws) * 1.1)

    # ─────────────────────────────────────────────
    # Utilidades gráficas
    # --------------------------------------------
    @staticmethod
    def _anchor_bottom_right(pw: pg.PlotWidget, lbl: pg.TextItem):
        vb = pw.getViewBox()
        x_max = vb.viewRange()[0][1]
        y_min = vb.viewRange()[1][0]
        lbl.setPos(x_max, y_min)

    def _draw_rulers(self, pw: pg.PlotWidget, t: np.ndarray):
        """Dibuja reglas verticales cada 1 s solo cuando la ventana visible cambia."""
        if t.size == 0:
            return

        if not hasattr(pw, "_ruler_sec_range"):
            pw._ruler_sec_range = (None, None)
            pw._ruler_lines: List[pg.InfiniteLine] = []

        window = cfg.get("PLOT_LENGTH") / cfg.get("SAMPLE_RATE")
        t_start = max(0.0, t[-1] - window)
        sec_start = int(np.floor(t_start))
        sec_end   = int(np.floor(t[-1]))

        if pw._ruler_sec_range == (sec_start, sec_end):
            return  # No cambia → no redibujar

        # borrar viejas
        for ln in pw._ruler_lines:
            pw.removeItem(ln)
        pw._ruler_lines.clear()

        # dibujar nuevas
        for s in range(sec_start + 1, sec_end + 1):
            ln = pg.InfiniteLine(pos=s, angle=90,
                                 pen=pg.mkPen((180, 180, 180, 80)))
            pw.addItem(ln)
            pw._ruler_lines.append(ln)
        pw._ruler_sec_range = (sec_start, sec_end)


# ──────────────────────────────────────────────────────────────────────────────
# Clase MonitorWindow
# ----------------------------------------------------------------------------
class MonitorWindow(QtWidgets.QMainWindow):
    ICON_SIZE = QtCore.QSize(32, 32)

    def __init__(self, launcher=None, serial=None) -> None:
        super().__init__()
        self.launcher = launcher
        self.serial   = serial or sb.DummySerial()
        self.panels: List[QtWidgets.QDockWidget] = []

        self.setWindowTitle("EEG Monitor")
        self._build_ui()
        self._start_timer()

        ThemeManager.instance().themeChanged.connect(self._after_theme_flip)

    # ---- UI -------------------------------------------------------------
    def _build_ui(self):
        toolbar = QtWidgets.QToolBar(movable=False)
        toolbar.setIconSize(self.ICON_SIZE)
        self.addToolBar(toolbar)

        def make_btn(svg, tip, cb):
            btn = QtWidgets.QToolButton()
            btn.setFixedSize(self.ICON_SIZE + QtCore.QSize(8, 8))
            btn.setIconSize(self.ICON_SIZE)
            btn.setToolTip(tip)
            btn.clicked.connect(cb)
            btn.setProperty("svg_path", svg)
            btn.setIcon(ThemeManager.instance().tinted_icon(svg, self.ICON_SIZE))
            return btn

        toolbar.addWidget(make_btn("icons/add.svg", "Add panel", self._add_panel))
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(
            nav_bar(
                self,
                lambda: self.launcher.stack.setCurrentIndex(0) if self.launcher else None,
                self._open_prefs,
                on_reconnect=self.launcher.reconnect_serial if self.launcher else None,
            )
        )

        # --- Paneles por defecto
        left   = self._create_panel("Eight",     QtCore.Qt.LeftDockWidgetArea)
        spec   = self._create_panel("Spectrum",  QtCore.Qt.RightDockWidgetArea)
        band   = self._create_panel("BandPower", QtCore.Qt.RightDockWidgetArea)
        self.splitDockWidget(spec, band, QtCore.Qt.Vertical)

    def _create_panel(self, mode: str, area: QtCore.Qt.DockWidgetArea):
        dock = QtWidgets.QDockWidget(mode, self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetMovable   |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )
        panel = PlotPanel(self.serial, mode)
        dock.setWidget(panel)
        self.addDockWidget(area, dock)
        self.panels.append(dock)
        return dock

    def _add_panel(self):
        if len(self.panels) >= 4:
            QtWidgets.QMessageBox.warning(self, "Limit", "Max 4 panels")
            return
        mode, ok = QtWidgets.QInputDialog.getItem(
            self, "Add Panel", "Choose type:",
            PlotPanel.MODES, 0, False
        )
        if not ok:
            return
        area_order = [
            QtCore.Qt.LeftDockWidgetArea,
            QtCore.Qt.RightDockWidgetArea,
            QtCore.Qt.TopDockWidgetArea,
            QtCore.Qt.BottomDockWidgetArea,
        ]
        dock = self._create_panel(mode, area_order[len(self.panels) % 4])

        # Si es BandPower y existe Spectrum, apilarlos
        if mode == "BandPower":
            for d in self.panels:
                if d.windowTitle() == "Spectrum":
                    self.splitDockWidget(d, dock, QtCore.Qt.Vertical)
                    break

    # ---- Timer ----------------------------------------------------------
    def _start_timer(self):
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self):
        try:
            for dock in self.panels:
                panel = dock.widget()
                if hasattr(panel, "refresh"):
                    panel.refresh()
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)

    # ---- Theming --------------------------------------------------------
    def _after_theme_flip(self, dark: bool):
        for btn in self.findChildren(QtWidgets.QToolButton):
            if svg_path := btn.property("svg_path"):
                btn.setIcon(ThemeManager.instance().tinted_icon(svg_path, self.ICON_SIZE))

    # ---- Preferencias ---------------------------------------------------
    def _open_prefs(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Monitor Settings")
        form = QtWidgets.QFormLayout(dlg)
        fields = {}

        try:
            settings = json.load(open("utils/config.json"))
        except Exception:  # archivo inexistente, etc.
            settings = {}

        for k, v in settings.items():
            le = QtWidgets.QLineEdit(str(v))
            fields[k] = le
            form.addRow(k, le)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(lambda: self._save_prefs(fields, dlg))
        bb.rejected.connect(dlg.reject)
        form.addWidget(bb)

        dlg.exec_()

    def _save_prefs(self, fields, dlg):
        new_cfg = {}
        for k, le in fields.items():
            txt = le.text().strip()
            new_cfg[k] = int(txt) if txt.isdigit() else float(txt) if "." in txt else txt
        json.dump(new_cfg, open("utils/config.json", "w"), indent=2)
        cfg.reload(new_cfg)
        self._timer.setInterval(cfg.get("UPDATE_INTERVAL"))
        dlg.accept()

    # ---- Serial swap / cierre ------------------------------------------
    def set_serial(self, ser):
        self.serial = ser
        for dock in self.panels:
            panel = dock.widget()
            if hasattr(panel, "serial"):
                panel.serial = ser

    def closeEvent(self, event):
        if isinstance(self.serial, sb.SerialThread):
            self.serial.stop()
        super().closeEvent(event)
