# class_manager.py
import json
from PyQt5 import QtWidgets, QtCore

class ClassManager(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.classes = []  # Lista de diccionarios: {'name': str, 'duration': float}
        self.init_ui()
        self.load_classes()
    
    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Tabla para mostrar las clases
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class Name", "Duration (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        
        # Formulario para ingresar nueva clase
        form_layout = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.duration_edit = QtWidgets.QLineEdit()
        form_layout.addRow("Class Name:", self.name_edit)
        form_layout.addRow("Duration (s):", self.duration_edit)
        layout.addLayout(form_layout)
        
        # Botones para operaciones CRUD
        btn_layout = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add")
        self.update_btn = QtWidgets.QPushButton("Update")
        self.delete_btn = QtWidgets.QPushButton("Delete")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.delete_btn)
        layout.addLayout(btn_layout)
        
        self.add_btn.clicked.connect(self.add_class)
        self.update_btn.clicked.connect(self.update_class)
        self.delete_btn.clicked.connect(self.delete_class)
        
        self.table.itemSelectionChanged.connect(self.load_selected_class)
    
    def load_classes(self):
        try:
            with open("bci_trainer/classes.json", "r") as f:
                self.classes = json.load(f)
            self.refresh_table()
        except Exception as e:
            # Si no existe o hay error, se inicia con lista vacía.
            self.classes = []
    
    def save_classes(self):
        try:
            with open("bci_trainer/classes.json", "w") as f:
                json.dump(self.classes, f, indent=4)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Error saving classes: {e}")
    
    def add_class(self):
        name = self.name_edit.text().strip()
        try:
            duration = float(self.duration_edit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Duration must be a number.")
            return
        if name == "":
            QtWidgets.QMessageBox.warning(self, "Error", "Class name cannot be empty.")
            return
        # Verifica si ya existe la clase
        for cls in self.classes:
            if cls['name'] == name:
                QtWidgets.QMessageBox.warning(self, "Error", "Class already exists.")
                return
        self.classes.append({'name': name, 'duration': duration})
        self.refresh_table()
        self.save_classes()
        self.name_edit.clear()
        self.duration_edit.clear()
    
    def update_class(self):
        selected = self.table.currentRow()
        if selected < 0 or selected >= len(self.classes):
            return
        name = self.name_edit.text().strip()
        try:
            duration = float(self.duration_edit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Duration must be a number.")
            return
        if name == "":
            QtWidgets.QMessageBox.warning(self, "Error", "Class name cannot be empty.")
            return
        self.classes[selected] = {'name': name, 'duration': duration}
        self.refresh_table()
        self.save_classes()
    
    def delete_class(self):
        selected = self.table.currentRow()
        if selected < 0 or selected >= len(self.classes):
            return
        del self.classes[selected]
        self.refresh_table()
        self.save_classes()
    
    def load_selected_class(self):
        selected = self.table.currentRow()
        if selected < 0 or selected >= len(self.classes):
            return
        cls = self.classes[selected]
        self.name_edit.setText(cls['name'])
        self.duration_edit.setText(str(cls['duration']))
    
    def refresh_table(self):
        self.table.setRowCount(len(self.classes))
        for i, cls in enumerate(self.classes):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(cls['name']))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(cls['duration'])))
    
    def get_classes(self):
        return self.classes
