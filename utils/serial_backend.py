import re, numpy as np, serial, serial.tools.list_ports
from PyQt5 import QtCore
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch
import utils.config_manager as cfg

class DummySerial(QtCore.QObject):
    def get_plot_data(self, *a, **k): return np.array([]), np.array([])
    def get_fft_data(self,  *a, **k): return np.array([]), np.array([])
    def start_recording(self): pass
    def stop_recording(self): return np.array([])

class SerialThread(QtCore.QThread):
    data_received = QtCore.pyqtSignal(str)

    def __init__(self, port: str):
        super().__init__()
        self.port, self.running = port, False
        self.buffer  = [[] for _ in range(9)]
        self.times   = []
        nyq = .5 * cfg.get("SAMPLE_RATE")
        f0  = cfg.get("NOTCH_FREQ") / nyq
        self.b_notch, self.a_notch = iirnotch(f0, cfg.get("QUALITY_FACTOR"))
        self.zi_n = [lfilter_zi(self.b_notch, self.a_notch) for _ in range(9)]
        low_cut = cfg.get("FILTER_CUTOFF")/nyq
        self.b_low, self.a_low = butter(cfg.get("BUTTER_ORDER"), low_cut, btype='low')
        self.zi_l  = [lfilter_zi(self.b_low, self.a_low) for _ in range(9)]

    def run(self):
        self.running = True
        with serial.Serial(self.port,115200,timeout=1) as sp:
            sp.write(b'1')
            while self.running and sp.in_waiting:
                raw = sp.readline().decode('utf-8','ignore').strip()
                self.data_received.emit(raw)
                if re.match(r'^Channel:(-?\d+\.?\d*,){8}-?\d+\.?\d*$', raw):
                    vals = list(map(float, raw.split('Channel:')[1].split(',')))
                    self._push(vals)

    def stop(self):
        self.running = False
        self.wait(1000)

    def _push(self, vals):
        fv = []
        for i,v in enumerate(vals):
            y, self.zi_n[i] = lfilter(self.b_notch, self.a_notch, [v], zi=self.zi_n[i]); fv.append(y[0])
        for i,v in enumerate(fv):
            y, self.zi_l[i] = lfilter(self.b_low, self.a_low, [v], zi=self.zi_l[i]); fv[i] = y[0]
        for i in range(9):
            self.buffer[i].append(fv[i])
        self.times.append(len(self.times))

    def _slice(self, length):
        if not self.times: return np.array([]), np.array([])
        start = max(0, len(self.times) - length)
        x = np.arange(start, len(self.times))
        d = [np.array(ch[start:]) for ch in self.buffer]
        return x, np.array(d)

    def get_plot_data(self, length=None):
        return self._slice(length or cfg.get("PLOT_LENGTH"))

    def get_fft_data(self, length=None):
        return self._slice(length or cfg.get("FFT_LENGTH"))

def find_usb():
    for p in serial.tools.list_ports.comports():
        if 'USB' in p.description: return p.device
