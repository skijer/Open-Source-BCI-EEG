import os
import json
import time
import socket
import threading
import numpy as np
import keyboard
import cv2
import pyautogui
from PyQt5 import QtWidgets, QtCore
from utils.theme_manager import ThemeManager
from utils.ui_helpers import nav_bar
from utils.OrloskyPupilDetectorRaspberryPi4BCI import crop_to_aspect_ratio, get_darkest_area

class InferenceWorker(QtCore.QThread):
    def __init__(self, serial, model_folder, use_best, repeat, action_map):
        super().__init__()
        self.serial = serial
        self.model_folder = model_folder
        self.use_best = use_best
        self.repeat = repeat
        self.action_map = action_map
        self.requested_stop = False

    def run(self):
        import tensorflow as tf
        from numpy import load

        fname = "best_model.keras" if self.use_best else "final_model.keras"
        mpath = os.path.join(self.model_folder, fname)
        meta  = os.path.join(self.model_folder, "preproc_metadata.npz")
        model = tf.keras.models.load_model(mpath, compile=False)
        data  = load(meta, allow_pickle=True)
        mean, scale = data["mean"], data["scale"]
        cls_map     = data["classes"].item()
        idx2name    = {v: k for k, v in cls_map.items()}
        chans, samples = int(data["chans"]), int(data["samples"])
        hist = []

        while not self.requested_stop:
            _, arr = self.serial.get_plot_data(length=samples)
            if arr.shape[1] < samples:
                time.sleep(0.1)
                continue

            chunk = arr[1:1+chans, -samples:].astype(np.float32)
            flat  = (chunk.reshape(1, -1) - mean) / scale
            x     = flat.reshape(1, chans, samples, 1)
            idx   = int(np.argmax(model.predict(x, verbose=0), axis=1)[0])
            name  = idx2name.get(idx, f"#{idx}")
            hist.append(name)

            if len(hist) >= self.repeat and all(p == name for p in hist[-self.repeat:]):
                act = self.action_map.get(name)
                if act:
                    pyautogui.press(act)
            if len(hist) > self.repeat:
                hist.pop(0)

            time.sleep(0.2)

    def stop(self):
        self.requested_stop = True

class CameraWorker(QtCore.QThread):
    def __init__(self, cam_index, sens_x, sens_y, mapping, include_center, include_diag):
        super().__init__()
        self.cam_index      = cam_index
        self.sens_x         = sens_x
        self.sens_y         = sens_y
        self.mapping        = mapping
        self.include_center = include_center
        self.include_diag   = include_diag
        self.requested_stop = False

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            return
        ret, frm = cap.read()
        if not ret:
            cap.release()
            return

        im          = crop_to_aspect_ratio(frm)
        base_center = get_darkest_area(im) or (im.shape[1]//2, im.shape[0]//2)

        while not self.requested_stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            im = crop_to_aspect_ratio(frame)
            pt = get_darkest_area(im)
            if not pt:
                time.sleep(0.1)
                continue

            dx, dy = pt[0] - base_center[0], pt[1] - base_center[1]
            dirs = []
            if dy < -self.sens_y: dirs.append("up")
            if dy >  self.sens_y: dirs.append("down")
            if dx < -self.sens_x: dirs.append("left")
            if dx >  self.sens_x: dirs.append("right")
            if self.include_center and abs(dx) < self.sens_x and abs(dy) < self.sens_y:
                dirs = ["center"]
            if not self.include_diag and len(dirs) == 2:
                dirs = [dirs[0]]

            key = self.mapping.get(tuple(dirs))
            if key:
                if "+" in key:
                    keyboard.send(key)
                else:
                    keyboard.press_and_release(key)

            time.sleep(0.05)

        cap.release()

    def stop(self):
        self.requested_stop = True

class UtpWorker(QtCore.QThread):
    log_send = QtCore.pyqtSignal(str)
    log_ack  = QtCore.pyqtSignal(str)

    def __init__(self, ip, port, cmd_map):
        super().__init__()
        self.ip, self.port = ip, port
        self.cmd_map       = cmd_map
        self.requested_stop= False

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.port))

        def listen_ack():
            while not self.requested_stop:
                try:
                    data, addr = sock.recvfrom(1024)
                    msg = data.decode()
                    self.log_ack.emit(msg)
                except:
                    break

        threading.Thread(target=listen_ack, daemon=True).start()

        for key, cmd in self.cmd_map.items():
            keyboard.on_press_key(key, lambda e, c=cmd: self._send_cmd(c, sock))

        while not self.requested_stop:
            time.sleep(0.1)

        sock.close()

    def _send_cmd(self, cmd, sock):
        sock.sendto(cmd.encode(), (self.ip, self.port))
        self.log_send.emit(cmd)

    def stop(self):
        self.requested_stop = True

class ControlPanelWidget(QtWidgets.QWidget):
    def __init__(self, launcher, serial):
        super().__init__(launcher)
        self.launcher = launcher
        self.serial   = serial
        self.inf_map  = {}
        self.cam_map  = {}
        self.utp_cmd_sets = {}

        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(12,12,12,12)
        main.setSpacing(10)

        nav = nav_bar(self, self._on_return, lambda: None,
                      on_reconnect=launcher.reconnect_serial)
        main.addWidget(nav)

        gb_inf = QtWidgets.QGroupBox("EEG Inference")
        lay_inf = QtWidgets.QGridLayout(gb_inf)
        self.cb_inf = QtWidgets.QCheckBox("Enable inference")
        self.cmb_models = QtWidgets.QComboBox()
        self._reload_model_list()
        self.rb_best = QtWidgets.QRadioButton("Best")
        self.rb_final = QtWidgets.QRadioButton("Final")
        self.rb_best.setChecked(True)
        self.spn_thr = QtWidgets.QSpinBox()
        self.spn_thr.setRange(1,20)
        self.spn_thr.setValue(3)
        self.btn_map_inf = QtWidgets.QPushButton("Configure mapping…")
        self.lb_map_inf = QtWidgets.QLabel("Mapped: –")
        self.btn_map_inf.clicked.connect(self._configure_inf_mapping)
        lay_inf.addWidget(self.cb_inf,0,0,1,3)
        lay_inf.addWidget(QtWidgets.QLabel("Model / Version:"),1,0)
        row_mod = QtWidgets.QHBoxLayout()
        row_mod.addWidget(self.cmb_models)
        row_mod.addWidget(self.rb_best)
        row_mod.addWidget(self.rb_final)
        lay_inf.addLayout(row_mod,1,1,1,2)
        lay_inf.addWidget(QtWidgets.QLabel("Repeat threshold:"),2,0)
        lay_inf.addWidget(self.spn_thr,2,1,1,2)
        lay_inf.addWidget(self.btn_map_inf,3,1)
        lay_inf.addWidget(self.lb_map_inf,3,2)
        main.addWidget(gb_inf)

        gb_cam = QtWidgets.QGroupBox("Camera Position")
        lay_cam = QtWidgets.QGridLayout(gb_cam)
        self.cb_cam = QtWidgets.QCheckBox("Enable camera")
        self.cmb_port = QtWidgets.QComboBox()
        self._scan_cams()
        self.btn_map_cam = QtWidgets.QPushButton("Configure mapping…")
        self.lb_map_cam = QtWidgets.QLabel("Mapped: –")
        self.btn_map_cam.clicked.connect(self._configure_cam_mapping)
        lay_cam.addWidget(self.cb_cam,0,0,1,2)
        lay_cam.addWidget(QtWidgets.QLabel("Port:"),1,0)
        lay_cam.addWidget(self.cmb_port,1,1)
        lay_cam.addWidget(self.btn_map_cam,2,0)
        lay_cam.addWidget(self.lb_map_cam,2,1)
        main.addWidget(gb_cam)

        gb_utp = QtWidgets.QGroupBox("UTP Connection")
        lay_utp = QtWidgets.QGridLayout(gb_utp)
        self.cb_utp = QtWidgets.QCheckBox("Enable UTP")
        self.edit_ip = QtWidgets.QLineEdit()
        self.edit_ip.setPlaceholderText("IP address")
        self.edit_port = QtWidgets.QSpinBox()
        self.edit_port.setRange(1,65535)
        self.txt_utp_log = QtWidgets.QTextEdit()
        self.txt_utp_log.setReadOnly(True)
        lay_utp.addWidget(self.cb_utp,0,0,1,2)
        lay_utp.addWidget(QtWidgets.QLabel("IP:"),1,0)
        lay_utp.addWidget(self.edit_ip,1,1)
        lay_utp.addWidget(QtWidgets.QLabel("Port:"),2,0)
        lay_utp.addWidget(self.edit_port,2,1)
        lay_utp.addWidget(QtWidgets.QLabel("Log:"),3,0)
        lay_utp.addWidget(self.txt_utp_log,3,1)
        main.addWidget(gb_utp)

        self.btn_run = QtWidgets.QPushButton("▶ START ALL")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._on_toggle)
        main.addWidget(self.btn_run, alignment=QtCore.Qt.AlignRight)

        self.inf_worker = None
        self.cam_worker = None
        self.utp_worker = None

    def _reload_model_list(self):
        self.cmb_models.clear()
        if os.path.isdir("models"):
            for d in sorted(os.listdir("models")):
                if os.path.isdir(os.path.join("models",d)):
                    self.cmb_models.addItem(d)

    def _scan_cams(self):
        self.cmb_port.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i)
            ok, _ = cap.read()
            cap.release()
            if ok:
                self.cmb_port.addItem(f"Camera {i}", i)
        if not self.cmb_port.count():
            self.cmb_port.addItem("No camera", -1)

    def _configure_inf_mapping(self):
        sel = self.cmb_models.currentText()
        meta = os.path.join("models",sel,"preproc_metadata.npz")
        if not os.path.isfile(meta):
            QtWidgets.QMessageBox.warning(self,"Error","Load a model first")
            return
        data = np.load(meta,allow_pickle=True)
        classes = list(data["classes"].item().keys())
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Map EEG classes to actions")
        lay = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(len(classes),2)
        table.setHorizontalHeaderLabels(["Class","Action key"])
        table.verticalHeader().setVisible(False)
        for r,cls in enumerate(classes):
            itm_c = QtWidgets.QTableWidgetItem(cls)
            itm_c.setFlags(QtCore.Qt.ItemIsEnabled)
            itm_a = QtWidgets.QTableWidgetItem(self.inf_map.get(cls,""))
            table.setItem(r,0,itm_c)
            table.setItem(r,1,itm_a)
        lay.addWidget(table)
        lay.addWidget(QtWidgets.QPushButton("Save",clicked=dlg.accept))
        if dlg.exec_():
            self.inf_map.clear()
            for r in range(table.rowCount()):
                cls = table.item(r,0).text()
                key = table.item(r,1).text().lower().strip()
                if key:
                    self.inf_map[cls] = key
            mapping_file = os.path.join("models",sel,"action_map.json")
            with open(mapping_file,"w") as f:
                json.dump(self.inf_map,f,indent=2)
            self.lb_map_inf.setText("Mapped: " + " | ".join(f"{c}->{k}" for c,k in self.inf_map.items()))

    def _configure_cam_mapping(self):
        dirs = [["up"],["down"],["left"],["right"]]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Map camera directions to actions")
        lay = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(len(dirs),2)
        table.setHorizontalHeaderLabels(["Directions","Action key"])
        table.verticalHeader().setVisible(False)
        for r,d in enumerate(dirs):
            table.setItem(r,0,QtWidgets.QTableWidgetItem("+".join(d)))
            table.setItem(r,1,QtWidgets.QTableWidgetItem(self.cam_map.get(tuple(d),"")))
        lay.addWidget(table)
        lay.addWidget(QtWidgets.QPushButton("Save",clicked=dlg.accept))
        if dlg.exec_():
            self.cam_map.clear()
            for r in range(table.rowCount()):
                d = table.item(r,0).text().split("+")
                key = table.item(r,1).text().lower().strip()
                if key:
                    self.cam_map[tuple(d)] = key
            self.lb_map_cam.setText("Mapped: " + " | ".join(f"{'+'.join(d)}->{k}" for d,k in self.cam_map.items()))

    def _on_toggle(self):
        if self.inf_worker or self.cam_worker or self.utp_worker:
            for w in (self.inf_worker, self.cam_worker, self.utp_worker):
                if w:
                    w.stop()
                    w.wait()
            self.inf_worker = self.cam_worker = self.utp_worker = None
            self.btn_run.setText("▶ START ALL")
            return

        if self.cb_inf.isChecked():
            mf   = self.cmb_models.currentText()
            best = self.rb_best.isChecked()
            thr  = self.spn_thr.value()
            self.inf_worker = InferenceWorker(self.serial, os.path.join("models",mf), best, thr, self.inf_map)
            self.inf_worker.start()

        if self.cb_cam.isChecked():
            idx  = self.cmb_port.currentData()
            center = False
            diag   = False
            self.cam_worker = CameraWorker(idx,100,100,self.cam_map,center,diag)
            self.cam_worker.start()

        if self.cb_utp.isChecked():
            ip   = self.edit_ip.text().strip()
            port = int(self.edit_port.value())
            self.utp_worker = UtpWorker(ip,port,self.utp_cmd_sets.get("default",{}))
            self.utp_worker.log_send.connect(lambda c: self.txt_utp_log.append(f"Sent: {c}"))
            self.utp_worker.log_ack.connect(lambda c: self.txt_utp_log.append(f"Ack: {c}"))
            self.utp_worker.start()

        self.btn_run.setText("■ STOP ALL")

    def _on_return(self):
        self._on_toggle()
        if hasattr(self.launcher, "stack"):
            self.launcher.stack.setCurrentIndex(0)
