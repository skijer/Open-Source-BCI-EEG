import os
import re
import numpy as np
from PyQt5 import QtWidgets, QtCore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint # type: ignore
import mne
import utils.config_manager as cfg
import utils.EEGModels  as EEGModels

class AITrainingWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recording_files = []
        self.data = []
        self.labels = []
        self.class_mapping = {}
        self.init_ui()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        self.load_btn = QtWidgets.QPushButton("Load Recordings")
        self.load_btn.clicked.connect(self.load_recordings)
        layout.addWidget(self.load_btn)
        
        self.records_list = QtWidgets.QListWidget()
        layout.addWidget(self.records_list)
        
        form_layout = QtWidgets.QFormLayout()
        self.batch_size_edit = QtWidgets.QLineEdit("32")
        self.epochs_edit = QtWidgets.QLineEdit("600")
        form_layout.addRow("Batch Size:", self.batch_size_edit)
        form_layout.addRow("Epochs:", self.epochs_edit)
        layout.addLayout(form_layout)
        
        self.train_btn = QtWidgets.QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)
        
        self.output_text = QtWidgets.QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
    
    def load_recordings(self):
        options = QtWidgets.QFileDialog.Options()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select Recording Files", "", "NPZ Files (*.npz)", options=options)
        if files:
            self.recording_files = files
            self.records_list.clear()
            for f in files:
                self.records_list.addItem(f)
    
    def log(self, message):
        self.output_text.append(message)
    
    def start_training(self):
        if not self.recording_files:
            QtWidgets.QMessageBox.warning(self, "Error", "No recordings loaded.")
            return
        self.log("Loading recordings...")
        all_data = []
        all_labels = []
        label_counter = 0
        for file in self.recording_files:
            try:
                npz = np.load(file, allow_pickle=True)
                class_name = str(npz['class_name'])
                duration = float(npz['duration'])
                recordings = npz['recordings']
                # Se espera que recordings tenga forma (n_recordings, channels, samples)
                if recordings.ndim != 3:
                    self.log(f"File {file} has invalid recordings shape.")
                    continue
                n_recs, chans, samples = recordings.shape
                self.log(f"Loaded {n_recs} recordings from class '{class_name}' with {samples} samples.")
                expected_samples = int(duration * cfg.get("SAMPLE_RATE"))
                if samples != expected_samples:
                    self.log(f"File {file} sample count mismatch: expected {expected_samples}, got {samples}. Skipping.")
                    continue
                if class_name not in self.class_mapping:
                    self.class_mapping[class_name] = label_counter
                    label_counter += 1
                label = self.class_mapping[class_name]
                for rec in recordings:
                    all_data.append(rec)
                    all_labels.append(label)
            except Exception as e:
                self.log(f"Error loading {file}: {e}")
        if not all_data:
            self.log("No valid recordings loaded.")
            return
        
        X = np.array(all_data)  # (n_samples, channels, samples)
        y = np.array(all_labels, dtype=int)
        self.log(f"Total recordings: {X.shape[0]}")
        
        # Remodelar X a (n_samples, channels, samples, 1)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1)).astype(np.float32)
        num_classes = len(self.class_mapping)
        y = to_categorical(y, num_classes=num_classes)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.log(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Normalización con StandardScaler
        n_train, chans, samples, depth = X_train.shape
        X_train_2d = X_train.reshape(n_train, chans * samples * depth)
        X_test_2d = X_test.reshape(X_test.shape[0], chans * samples * depth)
        
        scaler = StandardScaler()
        scaler.fit(X_train_2d)
        X_train_2d = scaler.transform(X_train_2d)
        X_test_2d = scaler.transform(X_test_2d)
        
        X_train = X_train_2d.reshape(n_train, chans, samples, depth)
        X_test = X_test_2d.reshape(X_test.shape[0], chans, samples, depth)
        
        # Guardar parámetros del scaler para usar en inferencia
        try:
            np.savez("scaler_params.npz", mean=scaler.mean_, scale=scaler.scale_)
            self.log("Scaler parameters saved in scaler_params.npz")
        except Exception as e:
            self.log(f"Error saving scaler parameters: {e}")
        
        try:
            batch_size = int(self.batch_size_edit.text().strip())
            epochs = int(self.epochs_edit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid training parameters.")
            return
        
        self.log("Building model...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
        checkpoint = ModelCheckpoint(
            'best_model_LR.keras',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        chans_input = X_train.shape[1]
        samples_input = X_train.shape[2]
        model = EEGModels.EEGNet(nb_classes=num_classes, Chans=chans_input, Samples=samples_input, dropoutRate=0.25)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary(print_fn=self.log)
        
        self.log("Starting training...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[lr_reducer, early_stop, checkpoint],
            verbose=1
        )
        
        score = model.evaluate(X_test, y_test, verbose=0)
        self.log(f"\nFinal Loss: {score[0]:.4f}, Final Accuracy: {score[1]*100:.2f}%")
        model.save('eegnet_final_LR.keras')
        self.log("\nFinal model saved as eegnet_final_LR.keras")
        self.log("Best model (lowest val_loss) saved as best_model_LR.keras")
