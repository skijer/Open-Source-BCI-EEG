# bci_trainer/panels/eeg_recorder_widget.py
import numpy as np
import json
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy.signal import resample
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

    Durante la fase 3, se capturan datos del serial en un buffer.
    Al terminar esa fase 3, se emite la señal recording_finished con los datos capturados.
    """
    recording_finished = QtCore.pyqtSignal(np.ndarray)  # Emite el registro obtenido

    def __init__(self, serial, record_duration, parent=None):
        super().__init__(parent)
        self.serial = serial
        self.record_duration = record_duration  # en segundos
        self.target_samples = int(self.record_duration * 500)  # forzamos 500 Hz
        
        # Buffer para guardar los datos de grabación
        self.recording_data = []
        self.is_recording = False

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
        
        # Timer para muestrear datos durante la grabación (100ms = 10Hz de polling)
        self.sample_timer = QtCore.QTimer(self)
        self.sample_timer.setInterval(100)  # 10 Hz de polling
        self.sample_timer.timeout.connect(self.collect_sample)

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
                self.label.setText("Recording...")
                self.start_recording()

                # Al terminar la fase 3, se llama stop_and_continue()
                QtCore.QTimer.singleShot(self.phase_times[3], self.stop_and_continue)
            else:
                # Las demás fases simplemente esperan su tiempo
                self.timer.start(self.phase_times[self.current_phase])
        else:
            # Al terminar la última fase (fase 4), cerramos la ventana
            self.close()

    def start_recording(self):
        """Inicia la captura de datos del serial"""
        self.recording_data = []
        self.is_recording = True
        self.sample_timer.start()

    def collect_sample(self):
        """Captura una muestra del serial y la guarda en el buffer"""
        if self.is_recording:
            try:
                # Obtener datos actuales del serial
                _, data = self.serial.get_plot_data()
                if data.size > 0:
                    self.recording_data.append(np.copy(data))
            except Exception as e:
                print(f"Error collecting sample: {e}")

    def stop_recording(self):
        """Detiene la grabación y procesa los datos capturados"""
        self.is_recording = False
        self.sample_timer.stop()
        
        if not self.recording_data:
            # Si no hay datos, devolver un array vacío
            return np.zeros((8, self.target_samples))
            
        # Concatenar todos los datos capturados
        all_data = np.hstack(self.recording_data)
        
        # Asegurar que tengamos al menos 8 canales
        if all_data.shape[0] < 8:
            # Rellenar con ceros si faltan canales
            padded = np.zeros((8, all_data.shape[1]))
            padded[:all_data.shape[0], :] = all_data
            all_data = padded
        elif all_data.shape[0] > 8:
            # Usar solo los primeros 8 canales
            all_data = all_data[:8, :]
            
        return all_data

    def stop_and_continue(self):
        """
        Se llama cuando acaba la fase 3 de grabación.
        Detiene la grabación, emite la señal con los datos,
        y arranca la fase 4 (post-grabación).
        """
        recorded = self.stop_recording()  # (channels, samples)

        # Re-muestrear para que haya exactamente target_samples = n * 500
        actual_samples = recorded.shape[1]
        if actual_samples != self.target_samples and actual_samples > 0:
            try:
                recorded = resample(recorded, self.target_samples, axis=1)
            except Exception as e:
                print(f"Error resampling data: {e}")
                # En caso de error, crear un array de ceros
                recorded = np.zeros((8, self.target_samples))

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
    def __init__(self, serial=None, parent=None):
        super().__init__(parent)
        self.serial = serial
        self.recordings = []  # guarda cada grabación (arreglo numpy)
        self.class_info = None  # diccionario {'name','duration'}
        self.init_ui()

    def set_serial(self, serial):
        """Permite cambiar la conexión serial en tiempo de ejecución"""
        self.serial = serial

    def update_panel(self):
        """Método requerido por TrainerWindow para actualizaciones de configuración"""
        self.load_classes_from_file()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        form_layout = QtWidgets.QFormLayout()

        # ComboBox de clases (opcionalmente se cargan de un JSON: classes.json)
        self.class_combo = QtWidgets.QComboBox()
        form_layout.addRow("Select Class:", self.class_combo)

        # Número de repeticiones
        self.repetitions_edit = QtWidgets.QLineEdit()
        self.repetitions_edit.setText("1")  # valor predeterminado
        form_layout.addRow("Number of Recordings:", self.repetitions_edit)

        layout.addLayout(form_layout)

        self.start_btn = QtWidgets.QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.start_recording_sequence)
        layout.addWidget(self.start_btn)

        # Área de status
        self.status_label = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        
        # Cargar clases después de crear todos los widgets
        self.load_classes_from_file()

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
            current_text = self.class_combo.currentText()
            self.class_combo.clear()
            for cls in classes:
                self.class_combo.addItem(f"{cls['name']} ({cls['duration']} s)", cls)
            
            # Intentar restaurar la selección previa
            index = self.class_combo.findText(current_text)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
            
            # Solo actualizar label de estado si ya existe
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Loaded {len(classes)} classes")
        except Exception as e:
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Cannot load classes.json: {str(e)}")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", f"Cannot load classes.json: {str(e)}")

    def start_recording_sequence(self):
        """
        Lee la clase seleccionada y el número de repeticiones,
        y lanza la grabación secuencial con FullScreenRecorder.
        """
        if not self.serial:
            QtWidgets.QMessageBox.warning(self, "Error", "No serial connection available.")
            return
            
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
        self.status_label.setText(f"Starting recording sequence ({repetitions} repetitions)")
        self.start_next_recording()

    def start_next_recording(self):
        """
        Crea la ventana FullScreenRecorder para la siguiente repetición,
        o si ya terminamos, llama a finish_sequence().
        """
        if self.current_rep < self.total_rep:
            self.current_rep += 1
            self.status_label.setText(f"Recording {self.current_rep}/{self.total_rep}")
            rec_dur = self.class_info['duration']
            self.recorder = FullScreenRecorder(self.serial, rec_dur)
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
        self.status_label.setText(f"Completed recording {self.current_rep}/{self.total_rep}")
        QtCore.QTimer.singleShot(500, self.start_next_recording)

    def finish_sequence(self):
        """
        Al concluir todas las repeticiones, se guarda en un archivo .npz
        con datos relevantes (clase, duración, grabaciones, etc.)
        """
        self.status_label.setText("Sequence complete. Saving data...")
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
            self.status_label.setText(f"Recordings saved to {filename}")
            QtWidgets.QMessageBox.information(self, "Saved", f"Recordings saved to {filename}")
        else:
            self.status_label.setText("Save cancelled")