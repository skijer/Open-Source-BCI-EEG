# bci_trainer/panels/ai_inference_widget.py
import numpy as np, tensorflow as tf
from PyQt5 import QtCore, QtWidgets

class AIInferenceWidget(QtWidgets.QWidget):
    """
    Carga modelo + preproc_metadata.npz y predice sobre el buffer del SerialThread.
    """
    def __init__(self, serial_thread, parent=None):
        super().__init__(parent)
        self.serial_thread = serial_thread
        self.model = None

        # se rellenan al cargar metadata
        self.scaler_mean  = None
        self.scaler_scale = None
        self.class_map    = {}         # {'EyesClosed':0,…}
        self.idx_to_name  = {}
        self.num_chans    = 8
        self.num_samples  = 500

        self.pred_hist    = []
        self.MAXH         = 10
        self._build_ui()
        self.timer = QtCore.QTimer(self, timeout=self._tick)

    # -------------------------------------------------------------- UI
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        btn_load = QtWidgets.QPushButton("Load Model", clicked=self._load_model)
        lay.addWidget(btn_load)
        btn_start = QtWidgets.QPushButton("Start Inference", clicked=self._start)
        lay.addWidget(btn_start)
        btn_stop = QtWidgets.QPushButton("Stop Inference", clicked=lambda: self.timer.stop())
        lay.addWidget(btn_stop)

        self.lb_status   = QtWidgets.QLabel("Idle");          lay.addWidget(self.lb_status)
        self.lb_pred     = QtWidgets.QLabel("Prediction: –"); lay.addWidget(self.lb_pred)
        self.lb_action   = QtWidgets.QLabel("Mapped: –");     lay.addWidget(self.lb_action)
        self.lb_history  = QtWidgets.QLabel("History: –");    lay.addWidget(self.lb_history)
        lay.addStretch()

    # -------------------------------------------------------------- load
    def _load_model(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Keras model", "", "Keras (*.keras);;H5 (*.h5)")
        if not fn: return
        try:
            self.model = tf.keras.models.load_model(fn, compile=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e)); return

        # load metadata
        try:
            meta = np.load("preproc_metadata.npz", allow_pickle=True)
            self.scaler_mean  = meta["mean"]
            self.scaler_scale = meta["scale"]
            self.class_map    = meta["classes"].item()
            self.idx_to_name  = {v:k for k,v in self.class_map.items()}
            self.num_chans    = int(meta["chans"])
            self.num_samples  = int(meta["samples"])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Warning", f"Metadata not found: {e}")

        self.lb_status.setText("Model ready")

    # -------------------------------------------------------------- start / tick
    def _start(self):
        if self.model is None or self.scaler_mean is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Load model/metadata first");  return
        self.pred_hist.clear();  self.lb_history.setText("History:")
        self.timer.start(200);   self.lb_status.setText("Predicting …")

    def _tick(self):
        # 1) acquire last N samples
        _, data = self.serial_thread.get_plot_data(length=self.num_samples)
        if data.shape[1] < self.num_samples:    # not enough yet
            return

        # 2) take the same channel subset used in training (2-9 ==> 1:1+num_chans)
        chunk = data[1:1+self.num_chans, -self.num_samples:].astype(np.float32)

        # 3) scaler
        flat = chunk.reshape(1, -1)
        flat = (flat - self.scaler_mean) / self.scaler_scale

        # 4) reshape & predict
        x = flat.reshape(1, self.num_chans, self.num_samples, 1)
        probs = self.model.predict(x, verbose=0)
        idx   = int(np.argmax(probs, axis=1)[0])

        # 5) UI
        name = self.idx_to_name.get(idx, f"#{idx}")
        self.lb_pred.setText(f"Prediction: {name}")
        self.lb_action.setText(f"Mapped: {name}")   # cambia aquí si quieres teclas/acciones

        self.pred_hist.append(name)
        if len(self.pred_hist) > self.MAXH:
            self.pred_hist.pop(0)
        self.lb_history.setText("History: " + " → ".join(self.pred_hist))
