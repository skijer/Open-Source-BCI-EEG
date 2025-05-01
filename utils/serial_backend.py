
import re, traceback, math, time
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
        self.t  = 0                      # global sample index

    # ----- helpers ---------------------------------------------------
    def _make_chunk(self, n: int):
        x = np.arange(self.t, self.t + n)
        self.t += n
        bases  = [50 * np.sin(2 * np.pi * 8 * x / self.fs + p)
                  for p in np.linspace(0, np.pi, 9, endpoint=False)]
        noise  = 15 * np.random.randn(9, n)
        return x, np.array(bases) + noise

    # ----- public API ------------------------------------------------
    def get_plot_data(self, length=None):
        n = int(length or cfg.get("PLOT_LENGTH"))
        return self._make_chunk(n)

    def get_fft_data(self, length=None):
        n = int(length or cfg.get("FFT_LENGTH"))
        return self._make_chunk(n)

    def start_recording(self):   pass
    def stop_recording(self):    return np.zeros((9, 1))


# ────────────────────────────────────────────────── live serial thread
class SerialThread(QtCore.QThread):
    """
    Open the port **synchronously** in __init__.
    self.ok is True only if the open succeeded.
    """
    data_received = QtCore.pyqtSignal(str)

    def __init__(self, port: str):
        super().__init__()
        self.port    = port
        self.running = False
        self.ok      = False        # becomes True after successful open

        # attempt open right now
        try:
            self.sp = serial.Serial(self.port, 115200, timeout=1)
            self.sp.flushInput()
            self.ok = True
        except Exception as e:
            print("Serial open failed:", e)
            self.sp = None

        # ----- filters & buffers ------------------------------------
        self.buffer  = [[] for _ in range(9)]
        self.times   = []
        nyq = 0.5 * cfg.get("SAMPLE_RATE")

        f0  = cfg.get("NOTCH_FREQ") / nyq
        self.b_notch, self.a_notch = iirnotch(f0, cfg.get("QUALITY_FACTOR"))
        self.zi_n = [lfilter_zi(self.b_notch, self.a_notch) for _ in range(9)]

        low = cfg.get("FILTER_CUTOFF") / nyq
        self.b_low, self.a_low = butter(cfg.get("BUTTER_ORDER"), low, btype='low')
        self.zi_l = [lfilter_zi(self.b_low, self.a_low) for _ in range(9)]

    # ----------------------------------------------------------------
    def run(self):
        if not self.ok:
            return
        self.running = True
        self.sp.write(b'1')                 # tell MCU to start streaming
        while self.running:
            if self.sp.in_waiting:
                raw = self.sp.readline().decode(errors='ignore').strip()
                self.data_received.emit(raw)
                if re.match(r'^Channel:(-?\d+\.?\d*,){8}-?\d+\.?\d*$', raw):
                    vals = list(map(float, raw.split('Channel:')[1].split(',')))
                    self._push(vals)

    def stop(self):
        self.running = False
        self.wait(1000)
        if self.sp and self.sp.is_open:
            self.sp.close()

    # ----------------------------------------------------------------
    def _push(self, vals):
        fv = []
        for i, v in enumerate(vals):
            y, self.zi_n[i] = lfilter(self.b_notch, self.a_notch, [v],
                                       zi=self.zi_n[i]); fv.append(y[0])
        for i, v in enumerate(fv):
            y, self.zi_l[i] = lfilter(self.b_low, self.a_low, [v],
                                       zi=self.zi_l[i]); fv[i] = y[0]
        for i in range(9): self.buffer[i].append(fv[i])
        self.times.append(len(self.times))

    # ----------------------------------------------------------------
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
