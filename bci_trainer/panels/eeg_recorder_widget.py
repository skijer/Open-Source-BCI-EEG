# eeg_recorder.py

import numpy as np
import json
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy.signal import resample  # Para re-muestrear
import utils.config_manager as cfg

###############################################################################
#                           FullScreenRecorder
###############################################################################
class FullScreenRecorder(QtWidgets.QWidget):
    """
    Ventana a pantalla completa que muestra la siguiente secuencia:
      0. Negro por 5 s (reposo).
      1. Rojo por 1 s.
      2. Verde por 1 s (preparación).
      3. Grabación en verde durante n segundos.
      4. Verde por 1 s (post grabación).
    Luego se cierra la ventana.

    Durante la fase 3, se llama serial_thread.start_recording().
    Al terminar esa fase 3, se llama serial_thread.stop_recording() y
    se emite la señal recording_finished con los datos capturados.
    """
    recording_finished = QtCore.pyqtSignal(np.ndarray)  # Emite el registro obtenido

    def __init__(self, serial_thread, record_duration, parent=None):
        super().__init__(parent)
        self.serial_thread = serial_thread
        self.record_duration = record_duration  # en segundos
        self.target_samples = int(self.record_duration * 500)  # forzamos 500 Hz

        # Hacemos la ventana sin bordes y en topmost, luego FullScreen
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.showFullScreen()

        # Fases:
        #  0: Negro (5s)
        #  1: Rojo (1s)
        #  2: Verde (1s)
        #  3: Grabación en verde (n segundos)
        #  4: Verde (1s)
        self.current_phase = 0
        self.phase_times = [5000, 1000, 1000, int(self.record_duration * 1000), 1000]
        self.phase_colors = ["black", "red", "green", "green", "green"]

        # Timer que maneja las fases
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_phase)

        # Etiqueta para centrar (por si quieres mostrar texto como "Recording...")
        self.label = QtWidgets.QLabel("", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 48px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def showEvent(self, event):
        """
        Al mostrarse la ventana, la ponemos en negro,
        y programamos iniciar la secuencia con un leve retardo (100 ms).
        """
        super().showEvent(event)
        self.setStyleSheet("background-color: black;")

        # Espera 100 ms antes de arrancar la secuencia de fases,
        # así evitamos que la ventana se quede "congelada" en negro.
        QtCore.QTimer.singleShot(100, self.start_sequence)

    def start_sequence(self):
        """Inicia la secuencia de fases."""
        self.current_phase = 0
        self.setStyleSheet(f"background-color: {self.phase_colors[0]};")
        self.label.setText("")
        self.timer.start(self.phase_times[0])

    def next_phase(self):
        """Pasa a la siguiente fase cada vez que el QTimer termina."""
        self.timer.stop()
        self.current_phase += 1

        if self.current_phase < len(self.phase_times):
            # Cambiamos el color de fondo a la siguiente fase
            self.setStyleSheet(f"background-color: {self.phase_colors[self.current_phase]};")

            # Si estamos en la fase 3 -> iniciamos grabación
            if self.current_phase == 3:
                self.label.setText("")  # o "Recording..."
                self.serial_thread.start_recording()

                # Al terminar la fase 3, se llama stop_and_continue()
                QtCore.QTimer.singleShot(self.phase_times[3], self.stop_and_continue)
            else:
                # Las demás fases simplemente esperan su tiempo
                self.timer.start(self.phase_times[self.current_phase])
        else:
            # Al terminar la última fase (fase 4), cerramos la ventana
            self.close()

    def stop_and_continue(self):
        """
        Se llama cuando acaba la fase 3 de grabación.
        Detiene la grabación, emite la señal con los datos,
        y arranca la fase 4 (post-grabación).
        """
        recorded = self.serial_thread.stop_recording()  # (channels, samples)

        # Seleccionamos canales 2-9 (índices 1..8) si se grabaron >= 9 canales
        if recorded.shape[0] >= 9:
            recorded = recorded[1:9, :]

        # Re-muestrear para que haya exactamente target_samples = n * 500
        actual_samples = recorded.shape[1]
        if actual_samples != self.target_samples:
            recorded = resample(recorded, self.target_samples, axis=1)

        # Emitir la señal para que EEGRecorderWidget lo reciba
        self.recording_finished.emit(recorded)

        # Pasamos a la fase 4 (verde post grabación)
        self.timer.start(self.phase_times[4])


###############################################################################
#                            EEGRecorderWidget
###############################################################################
class EEGRecorderWidget(QtWidgets.QWidget):
    """
    Widget para grabar EEG con:
     - ComboBox de clases (cargadas de classes.json con {'name','duration'})
     - Número de repeticiones deseadas
     - Botón que inicia la secuencia
    Al terminar las repeticiones, guarda en un .npz.
    """
    def __init__(self, serial_thread, parent=None):
        super().__init__(parent)
        self.serial_thread = serial_thread
        self.recordings = []  # guarda cada grabación (arreglo numpy)
        self.class_info = None  # diccionario {'name','duration'}
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        form_layout = QtWidgets.QFormLayout()

        # ComboBox de clases (opcionalmente se cargan de un JSON: classes.json)
        self.class_combo = QtWidgets.QComboBox()
        self.load_classes_from_file()
        form_layout.addRow("Select Class:", self.class_combo)

        # Número de repeticiones
        self.repetitions_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Number of Recordings:", self.repetitions_edit)

        layout.addLayout(form_layout)

        self.start_btn = QtWidgets.QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.start_recording_sequence)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

    def load_classes_from_file(self):
        """
        Intenta cargar un archivo 'classes.json' que contenga
        una lista de objetos con {"name":..., "duration":...},
        por ejemplo:
        [
          {"name": "EyesClosed", "duration": 3},
          {"name": "EyesOpen", "duration": 3},
          ...
        ]
        y los añade al comboBox.
        """
        try:
            with open("bci_trainer/classes.json", "r") as f:
                classes = json.load(f)
            self.class_combo.clear()
            for cls in classes:
                self.class_combo.addItem(f"{cls['name']} ({cls['duration']} s)", cls)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Cannot load classes.json: {e}")

    def start_recording_sequence(self):
        """
        Lee la clase seleccionada y el número de repeticiones,
        y lanza la grabación secuencial con FullScreenRecorder.
        """
        current_index = self.class_combo.currentIndex()
        if current_index < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No class selected.")
            return

        self.class_info = self.class_combo.itemData(current_index)
        try:
            repetitions = int(self.repetitions_edit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid number of recordings.")
            return

        if repetitions <= 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Number of recordings must be positive.")
            return

        # Reiniciamos contadores y arrancamos la secuencia
        self.recordings = []
        self.current_rep = 0
        self.total_rep = repetitions
        self.start_next_recording()

    def start_next_recording(self):
        """
        Crea la ventana FullScreenRecorder para la siguiente repetición,
        o si ya terminamos, llama a finish_sequence().
        """
        if self.current_rep < self.total_rep:
            self.current_rep += 1
            rec_dur = self.class_info['duration']
            self.recorder = FullScreenRecorder(self.serial_thread, rec_dur)
            self.recorder.recording_finished.connect(self.handle_recording_finished)
            self.recorder.show()
        else:
            self.finish_sequence()

    def handle_recording_finished(self, recorded_data):
        """
        Recibe los datos grabados de la señal al terminar la fase 3
        y programa la siguiente grabación con 500 ms de retraso.
        """
        self.recordings.append(recorded_data)
        QtCore.QTimer.singleShot(500, self.start_next_recording)

    def finish_sequence(self):
        """
        Al concluir todas las repeticiones, se guarda en un archivo .npz
        con datos relevantes (clase, duración, grabaciones, etc.)
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Recordings",
            "",
            "NPZ Files (*.npz)",
            options=options
        )
        if filename:
            np.savez(
                filename,
                class_name=self.class_info['name'],
                duration=self.class_info['duration'],
                recordings=np.array(self.recordings),  # (repeticiones, 8 canales, muestras)
                notch_freq=cfg.get("NOTCH_FREQ"),
                filter_cutoff=cfg.get("FILTER_CUTOFF"),
                butter_cutoff=cfg.get("BUTTER_CUTOFF")
            )
            QtWidgets.QMessageBox.information(self, "Saved", f"Recordings saved to {filename}")
