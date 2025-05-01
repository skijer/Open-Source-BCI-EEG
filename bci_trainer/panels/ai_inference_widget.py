# bci_trainer/panels/ai_inference_widget.py
import os, json, pyautogui, numpy as np, tensorflow as tf
from PyQt5 import QtCore, QtWidgets

# ──────────────────────────────────────────────────────── THEME CONSTANTS
NEON_BLUE   = "#00d8ff"
BG_CTRL     = "#161b22"    # control background (dark)
TEXT_COLOR  = "#e6edf3"    # light text
PORTAL_FONT = "Consolas, 'Courier New', monospace"


class AIInferenceWidget(QtWidgets.QWidget):
    MAX_HIST = 20

    def __init__(self, serial_thread, parent=None):
        super().__init__(parent)
        self.serial_thread = serial_thread

        # ───────── runtime vars
        self.model_folder = None
        self.model        = None
        self.scaler_mean  = self.scaler_scale = None
        self.class_map, self.idx_to_name = {}, {}
        self.num_chans, self.num_samples = 8, 500
        self.action_map, self.pred_hist   = {}, []
        self.running       = False
        self.timer         = QtCore.QTimer(self, timeout=self._tick)

        # ───────── build UI & styles
        self._build_ui()
        self._apply_styles()

    # ────────────────────────────────────────────────── BUILD UI
    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setSpacing(14)

        # ===== MODEL SELECTION (fixed height) ==========================
        box_model = QtWidgets.QGroupBox()
        box_model.setProperty("notitle", True)
        box_model.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                QtWidgets.QSizePolicy.Fixed)  # FIXED vertical
        mdl = QtWidgets.QHBoxLayout(box_model)
        mdl.setContentsMargins(12, 10, 12, 10)
        mdl.setSpacing(8)

        mdl.addWidget(QtWidgets.QLabel("Model:"))

        self.cb_folders = QtWidgets.QComboBox(activated=self._on_folder_changed)
        self.cb_folders.setMinimumWidth(300)
        self._populate_folders()
        mdl.addWidget(self.cb_folders, 50)

        self.rd_best  = QtWidgets.QRadioButton("Best")
        self.rd_final = QtWidgets.QRadioButton("Final")
        rd_grp = QtWidgets.QButtonGroup(self)
        rd_grp.addButton(self.rd_best)
        rd_grp.addButton(self.rd_final)
        rd_grp.buttonToggled.connect(self._load_selected_model)

        rd_wrap = QtWidgets.QWidget()
        rd_lay  = QtWidgets.QHBoxLayout(rd_wrap)
        rd_lay.setContentsMargins(0, 0, 0, 0)
        rd_lay.setSpacing(4)
        rd_lay.addWidget(self.rd_best)
        rd_lay.addWidget(self.rd_final)
        mdl.addWidget(rd_wrap)

        main.addWidget(box_model)

        # ===== INFERENCE ================================================
        box_inf = QtWidgets.QGroupBox("Inference")
        inf = QtWidgets.QVBoxLayout(box_inf)
        inf.setContentsMargins(12, 18, 12, 12)
        inf.setSpacing(6)

        self.btn_inf   = QtWidgets.QPushButton("▶ Start", clicked=self._toggle)
        self.lb_status = self._lbl("Stopped")
        self.lb_pred   = self._lbl("Prediction: –", big=True)
        self.lb_hist   = self._lbl("History: –")

        inf.addWidget(self.btn_inf)
        inf.addWidget(self.lb_status)
        inf.addWidget(self.lb_pred)
        inf.addWidget(self.lb_hist)

        main.addWidget(box_inf)

        # ===== MAPPING ==================================================
        box_map = QtWidgets.QGroupBox("Class → Action mapping")
        mp = QtWidgets.QVBoxLayout(box_map)
        mp.setContentsMargins(12, 18, 12, 12)
        mp.setSpacing(6)

        self.btn_cfg  = QtWidgets.QPushButton("Configure …", clicked=self._configure_mapping)
        self.lb_map   = self._lbl("Mapped: –")

        mp.addWidget(self.btn_cfg)
        mp.addWidget(self.lb_map)

        main.addWidget(box_map)

        # disable controls until model is loaded
        self._set_enabled(False)

    # helper: plain label
    def _lbl(self, text, *, big=False):
        lab = QtWidgets.QLabel(text)
        lab.setProperty("plain", True)
        if big:
            lab.setProperty("big", True)
        return lab

    # ────────────────────────────────────────────────── STYLES
    def _apply_styles(self):
        self.setStyleSheet(f"""
        QWidget {{ font-family: {PORTAL_FONT}; color: {TEXT_COLOR}; }}

        /* Group boxes */
        QGroupBox {{
            border: 2px solid {NEON_BLUE};
            border-radius: 8px;
            margin-top: 12px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 14px; padding: 0 4px;
        }}
        /* remove top margin of the 'Model' box */
        QGroupBox[notitle="true"] {{ margin-top: 0px; }}

        /* Labels w/o borders */
        QLabel[plain="true"] {{ border: none; }}
        QLabel[big="true"]   {{ font-size: 12pt; font-weight: bold; }}

        /* Dark backgrounds for controls */
        QComboBox, QPushButton {{
            background: {BG_CTRL};
            border: 1px solid {NEON_BLUE};
            border-radius: 4px;
            padding: 2px 6px;
        }}
        QComboBox:hover, QPushButton:hover {{
            background: {NEON_BLUE};
            color: {BG_CTRL};
        }}

        /* Radios: only the indicator circle */
        QRadioButton {{
            background: transparent;
            border: none;
            padding: 0px 4px;
        }}
        QRadioButton::indicator {{
            width: 12px; height: 12px;
        }}
        """)

    # ────────────────────────────────────────────────── HELPERS
    def _populate_folders(self):
        self.cb_folders.clear()
        self.cb_folders.addItem("<select>")
        if os.path.isdir("models"):
            for d in sorted(os.listdir("models")):
                if os.path.isdir(os.path.join("models", d)):
                    self.cb_folders.addItem(d)

    def _set_enabled(self, ok: bool):
        self.btn_inf.setEnabled(ok)
        self.btn_cfg.setEnabled(ok)

    # ────────────────────────────────────────────────── MODEL LOADING
    def _on_folder_changed(self, idx):
        if idx == 0:
            self._set_enabled(False)
            return
        self.model_folder = os.path.join("models", self.cb_folders.currentText())
        self.rd_best.setChecked(True)  # triggers load

    def _load_selected_model(self):
        if not self.model_folder:
            return
        fname = "best_model.keras" if self.rd_best.isChecked() else "final_model.keras"
        mpath = os.path.join(self.model_folder, fname)
        meta  = os.path.join(self.model_folder, "preproc_metadata.npz")

        if not os.path.isfile(mpath):
            QtWidgets.QMessageBox.warning(self, "Missing", f"{fname} not found")
            self._set_enabled(False)
            return

        try:
            self.model = tf.keras.models.load_model(mpath, compile=False)
            data = np.load(meta, allow_pickle=True)
            self.scaler_mean  = data["mean"]
            self.scaler_scale = data["scale"]
            self.class_map    = data["classes"].item()
            self.idx_to_name  = {v: k for k, v in self.class_map.items()}
            self.num_chans    = int(data["chans"])
            self.num_samples  = int(data["samples"])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return

        self.lb_status.setText("Model ready")
        self._set_enabled(True)

    # ────────────────────────────────────────────────── INFERENCE TOGGLE
    def _toggle(self):
        if self.running:
            self.timer.stop()
            self.running = False
            self.btn_inf.setText("▶ Start")
            self.lb_status.setText("Stopped")
            return

        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Load a model first")
            return

        self.pred_hist.clear()
        self.lb_hist.setText("History: –")
        self.timer.start(200)
        self.running = True
        self.btn_inf.setText("■ Stop")
        self.lb_status.setText("Predicting …")

    # ────────────────────────────────────────────────── REAL-TIME STEP
    def _tick(self):
        _, data = self.serial_thread.get_plot_data(length=self.num_samples)
        if data.shape[1] < self.num_samples:
            return

        chunk = data[1:1+self.num_chans, -self.num_samples:].astype(np.float32)
        flat  = (chunk.reshape(1, -1) - self.scaler_mean) / self.scaler_scale
        x     = flat.reshape(1, self.num_chans, self.num_samples, 1)

        idx   = int(np.argmax(self.model.predict(x, verbose=0), axis=1)[0])
        name  = self.idx_to_name.get(idx, f"#{idx}")

        self.lb_pred.setText(f"Prediction: {name}")
        action = self.action_map.get(name)
        if action:
            pyautogui.press(action)
        self.lb_map.setText(f"Mapped: {action or '–'}")

        self.pred_hist.append(name)
        if len(self.pred_hist) > self.MAX_HIST:
            self.pred_hist.pop(0)
        self.lb_hist.setText("History: " + " → ".join(self.pred_hist))

    # ────────────────────────────────────────────────── MAPPING DIALOG
    def _configure_mapping(self):
        if not self.class_map:
            QtWidgets.QMessageBox.information(self, "No classes", "Load a model first")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Map classes to actions")
        lay = QtWidgets.QVBoxLayout(dlg)

        table = QtWidgets.QTableWidget(len(self.class_map), 2)
        table.setHorizontalHeaderLabels(["Class", "Action key"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)

        for r, cls in enumerate(self.class_map):
            itm_c = QtWidgets.QTableWidgetItem(cls)
            itm_c.setFlags(QtCore.Qt.ItemIsEnabled)
            itm_a = QtWidgets.QTableWidgetItem(self.action_map.get(cls, ""))
            table.setItem(r, 0, itm_c)
            table.setItem(r, 1, itm_a)

        lay.addWidget(table)
        lay.addWidget(QtWidgets.QPushButton("Save", clicked=dlg.accept))

        if dlg.exec_():
            self.action_map.clear()
            for r in range(table.rowCount()):
                cls = table.item(r, 0).text().strip()
                act = table.item(r, 1).text().strip().lower()
                if act:
                    self.action_map[cls] = act
            if self.model_folder:
                with open(os.path.join(self.model_folder, "action_map.json"), "w") as f:
                    json.dump(self.action_map, f, indent=2)
            self.lb_map.setText("Mapped: –")
