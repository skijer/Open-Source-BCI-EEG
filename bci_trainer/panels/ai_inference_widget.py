import numpy as np
import tensorflow as tf
from PyQt5 import QtCore, QtWidgets
import utils.config_manager as cfg

class AIInferenceWidget(QtWidgets.QWidget):
    """
    Widget que:
     - Carga un modelo de Keras (botón "Load Model")
     - Empieza a inferir en tiempo real (botón "Start Inference"):
       * Toma las últimas N muestras del SerialThread (por ejemplo, 500 muestras = 1s)
       * Aplica el mismo preprocesado que se usó en entrenamiento:
         StandardScaler, reshaping a (1, chans, samples, 1), etc.
       * Ejecuta model.predict() y muestra la clase predicha.
     - Añadimos un label que muestre "Predicting..." mientras se actualiza.
     - Añadimos un pequeño historial (cola) para ver las últimas N predicciones.
    """

    def __init__(self, serial_thread, parent=None):
        super().__init__(parent)
        self.serial_thread = serial_thread
        self.model = None
        self.num_channels = 8  # Asumimos que en el entrenamiento se usaron 8 canales
        self.num_samples = 500  # Asumimos 1s de ventana a 500Hz

        # Parámetros del StandardScaler (se cargarán al cargar el modelo)
        self.scaler_mean = None   
        self.scaler_scale = None

        # Diccionario para mapear índice de clase -> "tecla"/acción
        self.class_to_key = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }

        # Para llevar un historial de las últimas N predicciones
        self.prediction_history = []
        self.MAX_HISTORY_SIZE = 10  # Cambia el tamaño de la cola a tu gusto

        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_inference)

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.load_model_btn = QtWidgets.QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_btn)

        self.start_inference_btn = QtWidgets.QPushButton("Start Inference")
        self.start_inference_btn.clicked.connect(self.start_inference)
        layout.addWidget(self.start_inference_btn)

        self.stop_inference_btn = QtWidgets.QPushButton("Stop Inference")
        self.stop_inference_btn.clicked.connect(self.stop_inference)
        layout.addWidget(self.stop_inference_btn)

        # Label que indicará cuando se esté haciendo la inferencia
        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setStyleSheet("font-size: 14px; color: gray;")
        layout.addWidget(self.status_label)

        # Etiqueta para mostrar la clase predicha
        self.prediction_label = QtWidgets.QLabel("Prediction: [None]")
        self.prediction_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.prediction_label)

        # Etiqueta para mostrar la acción/tecla asociada a la clase
        self.mapped_action_label = QtWidgets.QLabel("Mapped Action: [None]")
        self.mapped_action_label.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.mapped_action_label)

        # Cola / historial de predicciones
        self.history_label = QtWidgets.QLabel("History: (empty)")
        self.history_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.history_label)

        layout.addStretch()
        self.setLayout(layout)

    def load_model(self):
        """
        Permite seleccionar un archivo de modelo Keras (.keras, .h5, etc.) y lo carga.
        Además, intenta cargar los parámetros del StandardScaler usados en el entrenamiento.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Keras Model",
            "",
            "Keras Model (*.keras *.h5);;All Files (*)",
            options=options
        )
        if filename:
            try:
                self.model = tf.keras.models.load_model(filename, compile=False)
                self.prediction_label.setText("Prediction: [Model loaded]")
                # Intentar cargar parámetros del scaler
                try:
                    scaler_data = np.load("scaler_params.npz")
                    self.scaler_mean = scaler_data["mean"]
                    self.scaler_scale = scaler_data["scale"]
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Warning", f"Scaler parameters not loaded: {e}")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Cannot load model:\n{e}")

    def start_inference(self):
        """
        Inicia un QTimer que cada 200 ms toma datos del serial_thread, aplica el preprocesado y realiza la predicción.
        """
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No model loaded.")
            return
        if self.scaler_mean is None or self.scaler_scale is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Scaler parameters are not loaded.")
            return

        # Vacía la cola de historial previo
        self.prediction_history.clear()
        self.history_label.setText("History: (empty)")

        # Arranca el timer
        self.status_label.setText("Predicting...")
        self.timer.start(200)

    def stop_inference(self):
        """Detiene el QTimer para dejar de inferir."""
        self.timer.stop()
        self.status_label.setText("Idle")

    def update_inference(self):
        """
        Toma un bloque de self.num_samples desde el final del buffer del serial_thread,
        aplica el mismo preprocesado que en entrenamiento, y realiza la predicción.
        """
        # Actualiza el label a "Predicting..." (por si quieres forzarlo cada vez)
        self.status_label.setText("Predicting...")

        # 1) Obtener x_axis y data del serial_thread (9 canales mínimo)
        x_axis, data = self.serial_thread.get_plot_data(length=self.num_samples)
        if data.shape[1] < self.num_samples:
            return  # No hay suficientes muestras aún

        # 2) Seleccionar canales 2..9, forma (8, num_samples)
        chunk = data[1:9, -self.num_samples:].astype(np.float32)

        # 3) Aplanar para reproducir preprocesado en 2D
        chunk_2d = chunk.reshape(1, -1)  # (1, 8*num_samples)

        # 4) Aplicar la transformación del escalador (StandardScaler manual)
        chunk_2d = (chunk_2d - self.scaler_mean) / self.scaler_scale

        # 5) Remodelar a 4D
        chunk_4d = chunk_2d.reshape(1, self.num_channels, self.num_samples, 1)

        # 6) Hacer la predicción
        preds = self.model.predict(chunk_4d)
        class_index = int(np.argmax(preds, axis=1)[0])

        # 7) Mostrar la predicción y la acción/tecla
        self.prediction_label.setText(f"Prediction: Class {class_index}")
        mapped_key = self.class_to_key.get(class_index, "N/A")
        self.mapped_action_label.setText(f"Mapped Action: {mapped_key}")

        # 8) Agregar la predicción a una “cola” para visualización
        self.prediction_history.append(class_index)
        if len(self.prediction_history) > self.MAX_HISTORY_SIZE:
            self.prediction_history.pop(0)

        # Mostrar la historia de predicciones en la etiqueta
        # (ej. "History: 0 -> 0 -> 1 -> 3")
        history_str = " -> ".join(str(c) for c in self.prediction_history)
        self.history_label.setText(f"History: {history_str}")
