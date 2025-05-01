# bci_trainer/panels/ai_training_widget.py
import os
import json
import numpy as np
from PyQt5 import QtWidgets, QtCore
from tensorflow.keras.utils import to_categorical          # type: ignore
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint          # type: ignore
import utils.config_manager as cfg
import utils.EEGModels  as EEGModels


class AITrainingWidget(QtWidgets.QWidget):
    """
    Entrena EEGNet y guarda en "models/<run_name>/":
      • best_model.keras        – mejor val_loss
      • final_model.keras       – última época
      • preproc_metadata.npz    – scaler + mapping + dims (para inferencia)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.files          = []
        self.class_mapping  = {}      # {'EyesClosed':0, ...}
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)

        # ------------- file loader
        self.btn_load = QtWidgets.QPushButton("Load Recordings")
        self.btn_load.clicked.connect(self._load_files)
        lay.addWidget(self.btn_load)

        self.lst_files = QtWidgets.QListWidget()
        lay.addWidget(self.lst_files)

        # ------------- preset hyper‑params
        grid = QtWidgets.QGridLayout()
        self.btn_soft_short = QtWidgets.QPushButton("Soft Short")   # 32 × 100
        self.btn_soft_long  = QtWidgets.QPushButton("Soft Long")    # 32 × 300
        self.btn_hard_short = QtWidgets.QPushButton("Hard Short")   # 64 × 100
        self.btn_hard_long  = QtWidgets.QPushButton("Hard Long")    # 64 × 300
        grid.addWidget(self.btn_soft_short, 0, 0)
        grid.addWidget(self.btn_soft_long,  0, 1)
        grid.addWidget(self.btn_hard_short, 0, 2)
        grid.addWidget(self.btn_hard_long,  0, 3)
        lay.addLayout(grid)

        self.btn_soft_short.clicked.connect(lambda: self._set_params(32, 100))
        self.btn_soft_long .clicked.connect(lambda: self._set_params(32, 300))
        self.btn_hard_short.clicked.connect(lambda: self._set_params(64, 100))
        self.btn_hard_long .clicked.connect(lambda: self._set_params(64, 300))

        # ------------- manual hyper‑params
        form = QtWidgets.QFormLayout()
        self.edt_batch  = QtWidgets.QLineEdit("32")
        self.edt_epochs = QtWidgets.QLineEdit("600")
        form.addRow("Batch size", self.edt_batch)
        form.addRow("Epochs",     self.edt_epochs)
        lay.addLayout(form)

        # ------------- train trigger
        self.btn_train = QtWidgets.QPushButton("Start Training")
        self.btn_train.clicked.connect(self._train)
        lay.addWidget(self.btn_train)

        # ------------- log box
        self.txt_log = QtWidgets.QTextEdit(readOnly=True)
        lay.addWidget(self.txt_log)

    # ------------------------------------------------------------------ helpers
    def _set_params(self, batch: int, epochs: int):
        """Update the batch/epochs edits without launching training."""
        self.edt_batch.setText(str(batch))
        self.edt_epochs.setText(str(epochs))

    def _log(self, msg: str):
        self.txt_log.append(msg)
        QtCore.QCoreApplication.processEvents()

    # ------------------------------------------------------------------ file load
    def _load_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select datasets (*.npz)", "", "NPZ Files (*.npz)")
        if not files:
            return
        self.files = files
        self.lst_files.clear()
        self.lst_files.addItems(files)

    # ------------------------------------------------------------------ training
    def _train(self):
        if not self.files:
            QtWidgets.QMessageBox.warning(self, "No files", "Load recordings first")
            return

        # Ask for run name
        run_name, ok = QtWidgets.QInputDialog.getText(
            self, "Folder name", "Enter a name for this training run:")
        if not ok or not run_name.strip():
            return
        run_name = run_name.strip()
        save_dir = os.path.join("models", run_name)
        os.makedirs(save_dir, exist_ok=True)

        X, y, groups = [], [], []   # groups → filename to avoid leakage
        label_ctr = 0

        self._log("Loading datasets …")
        for f in self.files:
            try:
                npz = np.load(f, allow_pickle=True)
                cls   = str(npz["class_name"])
                dur   = float(npz["duration"])
                recs  = npz["recordings"]                # (n, 8, samples)
                if recs.ndim != 3:
                    self._log(f"{f}: bad shape, skipped");  continue
                exp_samp = int(dur * cfg.get("SAMPLE_RATE"))
                if recs.shape[2] != exp_samp:
                    self._log(f"{f}: sample mismatch, skipped");  continue

                if cls not in self.class_mapping:
                    self.class_mapping[cls] = label_ctr;  label_ctr += 1
                lab = self.class_mapping[cls]

                for r in recs:
                    X.append(r);   y.append(lab);   groups.append(os.path.basename(f))
            except Exception as e:
                self._log(f"{f}: {e}")

        if not X:
            self._log("Nothing to train.");  return

        X = np.asarray(X,  dtype=np.float32)             # (N, chans, samples)
        y = np.asarray(y,  dtype=int)
        g = np.asarray(groups)

        # reshape & one-hot
        X = X[..., np.newaxis]                           # (N, chans, samples, 1)
        y = to_categorical(y, num_classes=len(self.class_mapping))

        # ---------------------------------------------------------------- split WITHOUT leakage
        gss = GroupShuffleSplit(test_size=.20, n_splits=1, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=g))
        X_train, X_test     = X[train_idx], X[test_idx]
        y_train, y_test     = y[train_idx], y[test_idx]

        # ---------------------------------------------------------------- scaler
        ns, ch, smp, _ = X_train.shape
        scaler = StandardScaler()
        X_train2d = scaler.fit_transform(X_train.reshape(ns, -1))
        X_test2d  = scaler.transform(X_test.reshape(X_test.shape[0], -1))
        X_train   = X_train2d.reshape(ns, ch, smp, 1)
        X_test    = X_test2d.reshape(X_test.shape[0], ch, smp, 1)

        # save scaler + mapping
        np.savez(os.path.join(save_dir, "preproc_metadata.npz"),
                 mean=scaler.mean_, scale=scaler.scale_,
                 classes=self.class_mapping, chans=ch, samples=smp)

        self._log(f"Train={X_train.shape}  Test={X_test.shape}")
        self._log(f"Classes: {self.class_mapping}")
        self._log(f"Saving models to: {save_dir}")

        # ---------------------------------------------------------------- model
        try:
            bs  = int(self.edt_batch.text())
            ep  = int(self.edt_epochs.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Bad hyper-parameters"); return

        model = EEGModels.EEGNet(nb_classes=len(self.class_mapping),
                                 Chans=ch, Samples=smp, dropoutRate=.25)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss="categorical_crossentropy", metrics=["accuracy"])

        cbs = [
            ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=10, verbose=1),
            EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1),
            ModelCheckpoint(os.path.join(save_dir, "best_model.keras"), monitor="val_loss",
                            save_best_only=True, verbose=1)
        ]
        self._log("Training …")
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=ep, batch_size=bs, callbacks=cbs, verbose=1)

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        self._log(f"Final – loss {loss:.4f}  acc {acc*100:.2f}%)")
        model.save(os.path.join(save_dir, "final_model.keras"))
        self._log("Saved: final_model.keras")
