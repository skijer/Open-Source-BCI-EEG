# serial_backend.py

import re
import numpy as np
import serial, serial.tools.list_ports
from PyQt5 import QtCore
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch
import utils.config_manager as cfg


# ────────────────────────────────────────────────── dummy generator
class DummySerial(QtCore.QObject):
    """Synthetic 8 Hz sine + noise, 9 channels."""
    def __init__(self):
        super().__init__()
        self.fs = cfg.get("SAMPLE_RATE")
        self.t  = 0  # global sample index

    def _make_chunk(self, n: int):
        x = np.arange(self.t, self.t + n)
        self.t += n
        bases = [
            50 * np.sin(2 * np.pi * 8 * x / self.fs + p)
            for p in np.linspace(0, np.pi, 9, endpoint=False)
        ]
        noise = 15 * np.random.randn(9, n)
        return x, np.array(bases) + noise

    def get_plot_data(self, length=None):
        n = int(length or cfg.get("PLOT_LENGTH"))
        return self._make_chunk(n)

    def get_fft_data(self, length=None):
        n = int(length or cfg.get("FFT_LENGTH"))
        return self._make_chunk(n)

    def start_recording(self): pass
    def stop_recording(self):  return np.zeros((9, 1))


# ────────────────────────────────────────────────── live serial thread
class SerialThread(QtCore.QThread):
    """
    Abre el puerto en __init__.  Señal `data_received` emite cada línea cruda.
    """
    data_received = QtCore.pyqtSignal(str)

    def __init__(self, port: str):
        super().__init__()
        self.port    = port
        self.running = False
        self.ok      = False

        # intento de apertura sincronizada
        try:
            self.sp = serial.Serial(self.port, 115200, timeout=1)
            self.sp.flushInput()
            self.ok = True
        except Exception as e:
            print("Serial open failed:", e)
            self.sp = None

        # buffers
        self.buffer = [[] for _ in range(9)]
        self.times  = []

        # parámetros de muestreo y diseño de filtros
        fs  = cfg.get("SAMPLE_RATE")
        nyq = 0.5 * fs

        # filtro notch 50/60 Hz
        f0 = cfg.get("NOTCH_FREQ") / nyq
        self.b_notch, self.a_notch = iirnotch(f0, cfg.get("QUALITY_FACTOR"))
        self.zi_n = [lfilter_zi(self.b_notch, self.a_notch) for _ in range(9)]

        # filtro band-pass (por ejemplo 4–60 Hz)
        low  = cfg.get("BANDPASS_LO") / nyq
        high = cfg.get("BANDPASS_HI") / nyq
        self.b_band, self.a_band = butter(
            cfg.get("BUTTER_ORDER"),
            [low, high],
            btype='band'
        )
        self.zi_bp = [lfilter_zi(self.b_band, self.a_band) for _ in range(9)]

    def run(self):
        if not self.ok:
            return
        self.running = True
        self.sp.write(b'1')  # MCU: start stream
        while self.running:
            if self.sp.in_waiting:
                raw = self.sp.readline().decode(errors='ignore').strip()
                self.data_received.emit(raw)

                # línea con 9 valores: Channel:v1,v2,...,v9
                if re.match(r'^Channel:(-?\d+\.?\d*,){8}-?\d+\.?\d*$', raw):
                    vals = list(map(float, raw.split('Channel:')[1].split(',')))
                    self._push(vals)

    def stop(self):
        self.running = False
        self.wait(1000)
        if self.sp and self.sp.is_open:
            self.sp.close()

    def _push(self, vals):
        # 1) notch
        fv = []
        for i, v in enumerate(vals):
            y, self.zi_n[i] = lfilter(
                self.b_notch, self.a_notch, [v], zi=self.zi_n[i]
            )
            fv.append(y[0])

        # 2) band-pass
        for i, v in enumerate(fv):
            y, self.zi_bp[i] = lfilter(
                self.b_band, self.a_band, [v], zi=self.zi_bp[i]
            )
            fv[i] = y[0]

        # 3) almacenar
        for i in range(9):
            self.buffer[i].append(fv[i])
        self.times.append(len(self.times))

    def _slice(self, length):
        if not self.times:
            return np.array([]), np.array([])
        start = max(0, len(self.times) - length)
        x = np.arange(start, len(self.times))
        d = [np.array(ch[start:]) for ch in self.buffer]
        return x, np.array(d)

    def get_plot_data(self, length=None):
        return self._slice(length or cfg.get("PLOT_LENGTH"))

    def get_fft_data(self, length=None):
        return self._slice(length or cfg.get("FFT_LENGTH"))


# ────────────────────────────────────────────────── usb scanner
def find_usb():
    for p in serial.tools.list_ports.comports():
        if 'USB' in p.description.upper():
            return p.device
    return None
