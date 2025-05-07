# ───────────────────────────── utils/control_panel_widget.py ─────────
# Master panel – EEG, camera, UTP  →  vJoy / keyboard
# ---------------------------------------------------------------------
import os, json, time, socket, threading, numpy as np, cv2
from PyQt5 import QtCore, QtWidgets
from utils.ui_helpers    import nav_bar
from utils.theme_manager import ThemeManager
# ---------------------------------------------------------------------
try:
    import pyvjoy
    _VJOY_OK = True
except ImportError:
    _VJOY_OK = False

DEFAULT_CAM_MAP = {("up",):"up",("down",):"down",("left",):"left",
                   ("right",):"right",("adelante",):"space",("atras",):"space"}

# ───────────────────────────── EEG INFERENCE WORKER ──────────────────
class InferenceWorker(QtCore.QThread):
    """
    Now emits either vJoy *or* keyboard depending on availability.
    Mapping values:
      • "A" , "left" , "space" … → keyboard keys
      • "1" , "2" …             → vJoy button numbers
    """
    def __init__(self, serial, model_folder, use_best, repeat, action_map):
        super().__init__()
        self.serial, self.model_folder = serial, model_folder
        self.use_best, self.repeat     = use_best, repeat
        self.map = action_map
        self._stop = False

        self.vj = None
        if _VJOY_OK:
            try:  self.vj = pyvjoy.VJoyDevice(1)
            except Exception: self.vj = None

    # ···───────────────────────────────────────────────────────────────
    def run(self):
        from numpy import load; import tensorflow as tf
        f   = "best_model.keras" if self.use_best else "final_model.keras"
        mdl = tf.keras.models.load_model(os.path.join(self.model_folder,f), compile=False)
        meta = load(os.path.join(self.model_folder,"preproc_metadata.npz"), allow_pickle=True)
        mean,scale = meta["mean"], meta["scale"]
        idx2name   = {v:k for k,v in meta["classes"].item().items()}
        chans,smp  = int(meta["chans"]), int(meta["samples"])

        hist=[]
        while not self._stop:
            _, arr = self.serial.get_plot_data(length=smp)
            if arr.shape[1] < smp:
                time.sleep(0.05); continue
            x = ((arr[1:1+chans,-smp:].astype(np.float32).reshape(1,-1)-mean)/scale
                 ).reshape(1,chans,smp,1)
            cls_idx = int(np.argmax(mdl.predict(x,verbose=0), axis=1)[0])
            cls     = idx2name.get(cls_idx,f"#{cls_idx}")
            hist.append(cls)
            if len(hist) >= self.repeat and all(h==cls for h in hist[-self.repeat:]):
                act = self.map.get(cls)
            if len(hist) > self.repeat: hist.pop(0)
            time.sleep(0.15)

    def stop(self): self._stop = True

# ───────────────────────────── UDP “UTP” WORKER ──────────────────────
class UtpWorker(QtCore.QThread):
    log_send = QtCore.pyqtSignal(str); log_ack = QtCore.pyqtSignal(str)
    def __init__(self, ip, port, cmd_map):
        super().__init__(); self.ip, self.port, self.map = ip, port, cmd_map
        self._stop = False
    def run(self):
        sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM); sock.bind(('',self.port))
        threading.Thread(target=self._listen,args=(sock,),daemon=True).start()
        import keyboard
        for k,c in self.map.items():
            keyboard.on_press_key(k, lambda e,cmd=c: self._send(sock,cmd))
        while not self._stop: time.sleep(0.1)
        sock.close()
    def _listen(self,sock):
        while not self._stop:
            try: data,*_=sock.recvfrom(1024); self.log_ack.emit(data.decode())
            except: break
    def _send(self,sock,cmd):
        sock.sendto(cmd.encode(),(self.ip,self.port)); self.log_send.emit(cmd)
    def stop(self): self._stop = True

# ───────────────────────────── MAIN CONTROL PANEL ────────────────────
class ControlPanelWidget(QtWidgets.QWidget):
    def __init__(self, launcher, serial):
        super().__init__(launcher)
        self.launcher, self.serial = launcher, serial

        self.inf_map = {}
        self.cam_map = DEFAULT_CAM_MAP.copy()
        self.utp_cmd_sets = {}

        self.inf_worker = self.cam_worker = self.utp_worker = None

        self._build_ui()

    # ···───────────────────────────────────────────────────────────────
    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self); main.setContentsMargins(12,12,12,12)
        main.addWidget(nav_bar(self, self._return, lambda: None,
                               on_reconnect=self.launcher.reconnect_serial))

        # EEG -----------------------------------------------------------
        gb = QtWidgets.QGroupBox("EEG Inference"); lay = QtWidgets.QGridLayout(gb)
        self.cb_inf = QtWidgets.QCheckBox("Enable inference")
        self.cmb_models = QtWidgets.QComboBox(); self._reload_models()
        self.rb_best  = QtWidgets.QRadioButton("Best");  self.rb_best.setChecked(True)
        self.rb_final = QtWidgets.QRadioButton("Final")
        self.sp_thr = QtWidgets.QSpinBox(); self.sp_thr.setRange(1,20); self.sp_thr.setValue(3)
        self.btn_map_inf = QtWidgets.QPushButton("Configure mapping…", clicked=self._map_inf)
        self.lb_map_inf  = QtWidgets.QLabel("Mapped: –")
        lay.addWidget(self.cb_inf,0,0,1,3)
        lay.addWidget(QtWidgets.QLabel("Model / Version:"),1,0)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.cmb_models); h.addWidget(self.rb_best); h.addWidget(self.rb_final)
        lay.addLayout(h,1,1,1,2)
        lay.addWidget(QtWidgets.QLabel("Repeat thr:"),2,0); lay.addWidget(self.sp_thr,2,1,1,2)
        lay.addWidget(self.btn_map_inf,3,1); lay.addWidget(self.lb_map_inf,3,2)
        main.addWidget(gb)


        # UTP -----------------------------------------------------------
        gb = QtWidgets.QGroupBox("UTP Connection"); lay = QtWidgets.QGridLayout(gb)
        self.cb_utp = QtWidgets.QCheckBox("Enable UTP")
        self.ip = QtWidgets.QLineEdit(); self.ip.setPlaceholderText("IP address")
        self.port = QtWidgets.QSpinBox(); self.port.setRange(1,65535)
        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.cb_utp,0,0,1,2)
        lay.addWidget(QtWidgets.QLabel("IP:"),1,0); lay.addWidget(self.ip,1,1)
        lay.addWidget(QtWidgets.QLabel("Port:"),2,0); lay.addWidget(self.port,2,1)
        lay.addWidget(QtWidgets.QLabel("Log:"),3,0); lay.addWidget(self.log,3,1)
        main.addWidget(gb)

        # Run/Stop ------------------------------------------------------
        self.btn_run = QtWidgets.QPushButton("▶ START ALL", clicked=self._toggle)
        self.btn_run.setMinimumHeight(40); main.addWidget(self.btn_run, alignment=QtCore.Qt.AlignRight)

    # ··· helpers
    def _reload_models(self):
        self.cmb_models.clear()
        if os.path.isdir("models"):
            for d in sorted(os.listdir("models")):
                if os.path.isdir(os.path.join("models",d)): self.cmb_models.addItem(d)

    # ─────────────── mapping dialogs (unchanged visual) ──────────────
    def _map_inf(self):
        sel = self.cmb_models.currentText()
        meta = os.path.join("models", sel, "preproc_metadata.npz")
        if not os.path.isfile(meta):
            QtWidgets.QMessageBox.warning(self,"Error","Load a model first"); return
        data = np.load(meta, allow_pickle=True); classes = list(data["classes"].item().keys())
        self.inf_map = _mapping_dialog(self, classes, self.inf_map, title="Map EEG classes → actions")
        self.lb_map_inf.setText("Mapped: " + (" | ".join(f"{k}->{v}" for k,v in self.inf_map.items()) or "–"))
        if sel: json.dump(self.inf_map, open(os.path.join("models",sel,"action_map.json"),"w"), indent=2)


    # ─────────────── run / stop all threads ───────────────────────────
    def _toggle(self):
        # ---- STOP ----------------------------------------------------
        if any((self.inf_worker, self.cam_worker, self.utp_worker)):
            for w in (self.inf_worker, self.cam_worker, self.utp_worker):
                if w: w.stop(); w.wait()
            self.inf_worker = self.cam_worker = self.utp_worker = None
            self.btn_run.setText("▶ START ALL"); return

        # ---- START EEG ----------------------------------------------
        if self.cb_inf.isChecked():
            self.inf_worker = InferenceWorker(self.serial,
                                              os.path.join("models", self.cmb_models.currentText()),
                                              self.rb_best.isChecked(),
                                              self.sp_thr.value(),
                                              self.inf_map)
            self.inf_worker.start()
        # ---- START UTP ----------------------------------------------
        if self.cb_utp.isChecked():
            self.utp_worker = UtpWorker(self.ip.text().strip(),
                                        self.port.value(),
                                        self.utp_cmd_sets.get("default", {}))
            self.utp_worker.log_send.connect(lambda c: self.log.append(f"Sent: {c}"))
            self.utp_worker.log_ack .connect(lambda c: self.log.append(f"Ack : {c}"))
            self.utp_worker.start()

        self.btn_run.setText("■ STOP ALL")

    def _return(self):
        self._toggle()
        if hasattr(self.launcher,"stack"):
            self.launcher.stack.setCurrentIndex(0)

# ───────────────────────────── tiny helper dlg ───────────────────────
def _mapping_dialog(parent, left_values, current_map, *, title="Mapping"):
    dlg = QtWidgets.QDialog(parent); dlg.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(dlg)
    tbl = QtWidgets.QTableWidget(len(left_values),2)
    tbl.setHorizontalHeaderLabels(["Input","Action"]); tbl.verticalHeader().setVisible(False)
    for r,val in enumerate(left_values):
        left = "+".join(val) if isinstance(val,(list,tuple)) else val
        tbl.setItem(r,0,QtWidgets.QTableWidgetItem(left)); tbl.item(r,0).setFlags(QtCore.Qt.ItemIsEnabled)
        tbl.setItem(r,1,QtWidgets.QTableWidgetItem(current_map.get(tuple([val]) if not isinstance(val,(list,tuple)) else tuple(val), "")))
    lay.addWidget(tbl); lay.addWidget(QtWidgets.QPushButton("Save", clicked=dlg.accept))
    if not dlg.exec_(): return current_map
    out = {}
    for r in range(tbl.rowCount()):
        left  = tuple(tbl.item(r,0).text().split("+"))
        right = tbl.item(r,1).text().strip().lower()
        if right: out[left] = right
    return out
