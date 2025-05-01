import cv2, json, time, numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import styles
from utils.theme_manager import ThemeManager
from utils.ui_helpers   import nav_bar
_tm = ThemeManager.instance()

# ────────────────────────────────────────────────────────── image helpers
def crop_to_aspect_ratio(image, width=640, height=480):
    h, w = image.shape[:2]
    ratio = width / height
    if w / h > ratio:
        new_w = int(h * ratio); off = (w - new_w) // 2
        image = image[:, off:off + new_w]
    else:
        new_h = int(w / ratio); off = (h - new_h) // 2
        image = image[off:off + new_h, :]
    return cv2.resize(image, (width, height))

def apply_binary_threshold(gray, darkest_val, add):
    _, out = cv2.threshold(gray, darkest_val + add, 255, cv2.THRESH_BINARY_INV)
    return out

def get_darkest_area(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ib, skip, area, isk = 20, 20, 20, 10
    best, pt = float("inf"), None
    for y in range(ib, g.shape[0] - ib, skip):
        for x in range(ib, g.shape[1] - ib, skip):
            s = np.sum(g[y:y+area:isk, x:x+area:isk])
            if s < best: best, pt = s, (x + area // 2, y + area // 2)
    return pt

def mask_outside_square(img, center, size):
    x, y = center; hs = size // 2
    mask = np.zeros_like(img)
    mask[max(0, y - hs):y + hs, max(0, x - hs):x + hs] = 255
    return cv2.bitwise_and(img, mask)

def filter_contours_by_area_and_return_largest(cnts, pix_thr, ratio_thr):
    best, chosen = 0, None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < pix_thr: continue
        x, y, w, h = cv2.boundingRect(c)
        if max(w / h, h / w) > ratio_thr: continue
        if a > best: best, chosen = a, c
    return [chosen] if chosen is not None else []

# ────────────────────────────────────────────────────────── settings dialog
class EyeConfigDialog(QtWidgets.QDialog):
    """
    Editable form – values stored in eye_config.json.
    On accept, the modified dict is available as self.result.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eye-tracker settings")

        try:
            cfg = json.load(open("eye_config.json"))
        except Exception:
            cfg = {}
        cx, cy = (cfg.get("calibrated_center") or [None, None])
        sx = cfg.get("sensitivity_x", 100)
        sy = cfg.get("sensitivity_y", 100)

        form = QtWidgets.QFormLayout(self)
        self.e_cx = QtWidgets.QLineEdit("" if cx is None else str(cx))
        self.e_cy = QtWidgets.QLineEdit("" if cy is None else str(cy))
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.e_cx); row.addWidget(self.e_cy)
        form.addRow("calibrated_center (x y):", row)

        self.e_sx = QtWidgets.QLineEdit(str(sx))
        self.e_sy = QtWidgets.QLineEdit(str(sy))
        form.addRow("sensitivity_x:", self.e_sx)
        form.addRow("sensitivity_y:", self.e_sy)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addWidget(bb)

    def accept(self):
        try:
            cx = int(self.e_cx.text());  cy = int(self.e_cy.text())
            sx = int(self.e_sx.text());  sy = int(self.e_sy.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error", "Todos los valores deben ser enteros.")
            return
        self.result = {"calibrated_center": [cx, cy],
                       "sensitivity_x": sx,
                       "sensitivity_y": sy}
        json.dump(self.result, open("eye_config.json", "w"), indent=2)
        super().accept()

# ────────────────────────────────────────────────────────── main widget
class CameraWidget(QtWidgets.QWidget):
    ICON_SIZE = QtCore.QSize(32, 32)

    def __init__(self, launcher=None, parent=None):
        super().__init__(parent)
        self.launcher = launcher

        # camera state
        self.cap   = None
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._grab_frame)

        self.calibrated_center = None
        self.sensitivity_x = 100
        self.sensitivity_y = 100
        self.last_time     = None

        # ── UI ────────────────────────────────────────────────────
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(nav_bar(self, self._on_return, self._open_settings), stretch=0)

        form = QtWidgets.QFormLayout()
        self.cmb = QtWidgets.QComboBox(); self._scan_cameras()
        form.addRow("Camera port:", self.cmb)

        self.sld_x, self.lbl_x = self._add_slider(
            form, "Sensitivity X:", self.sensitivity_x,
            lambda v: setattr(self, "sensitivity_x", v))
        self.sld_y, self.lbl_y = self._add_slider(
            form, "Sensitivity Y:", self.sensitivity_y,
            lambda v: setattr(self, "sensitivity_y", v))

        h = QtWidgets.QHBoxLayout()
        self.btn_test = QtWidgets.QPushButton("▶ Test")
        self.btn_cal  = QtWidgets.QPushButton("⏱ Calibrate")
        h.addWidget(self.btn_test); h.addWidget(self.btn_cal)
        form.addRow("", h)
        layout.addLayout(form, stretch=0)

        self.lbl = QtWidgets.QLabel(); self.lbl.setFixedSize(640, 480)
        self.lbl.setStyleSheet("background:black;")
        layout.addWidget(self.lbl, stretch=1, alignment=QtCore.Qt.AlignCenter)

        # signals / palette
        self.btn_test.clicked.connect(self._toggle_test)
        self.btn_cal.clicked.connect(self._do_calibration)
        _tm.themeChanged.connect(self._apply_theme); self._apply_theme(_tm.is_dark)

    # ── settings dialog ------------------------------------------------
    def _open_settings(self):
        dlg = EyeConfigDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            cfg = dlg.result
            self.calibrated_center = tuple(cfg["calibrated_center"])
            self.sensitivity_x     = cfg["sensitivity_x"]
            self.sensitivity_y     = cfg["sensitivity_y"]
            # update sliders / labels
            self.sld_x.setValue(self.sensitivity_x); self.lbl_x.setText(str(self.sensitivity_x))
            self.sld_y.setValue(self.sensitivity_y); self.lbl_y.setText(str(self.sensitivity_y))

    # ── small helpers --------------------------------------------------
    def _add_slider(self, layout, label, value, setter):
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setRange(0, 200); s.setValue(value)
        l = QtWidgets.QLabel(str(value))
        s.valueChanged.connect(lambda v: (setter(v), l.setText(str(v))))
        row = QtWidgets.QHBoxLayout(); row.addWidget(s); row.addWidget(l)
        layout.addRow(label, row); return s, l

    def _scan_cameras(self):
        self.cmb.clear()
        for i in range(5):
            cap = cv2.VideoCapture(i); ok, _ = cap.read(); cap.release()
            if ok: self.cmb.addItem(f"Camera {i}", i)
        if not self.cmb.count(): self.cmb.addItem("No camera", -1)

    # ── return to launcher --------------------------------------------
    def _on_return(self):
        resp = QtWidgets.QMessageBox.question(
            self, "Save Eye Config",
            "Guardar configuración de sensibilidad y calibración?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if resp == QtWidgets.QMessageBox.Yes:
            json.dump({"calibrated_center": list(self.calibrated_center) if self.calibrated_center else [None, None],
                       "sensitivity_x": self.sensitivity_x,
                       "sensitivity_y": self.sensitivity_y},
                      open("eye_config.json", "w"), indent=2)

        if self.timer.isActive(): self.timer.stop()
        if self.cap: self.cap.release()
        if self.launcher and hasattr(self.launcher, "stack"):
            self.launcher.stack.setCurrentIndex(0)

    # ── test / calibration --------------------------------------------
    def _toggle_test(self):
        if not self.timer.isActive():
            port = self.cmb.currentData()
            if port < 0:
                QtWidgets.QMessageBox.warning(self, "Camera", "No camera selected"); return
            self.cap = cv2.VideoCapture(port)
            self.timer.start(30); self.btn_test.setText("■ Stop")
        else:
            self.timer.stop(); self.btn_test.setText("▶ Test")
            if self.cap: self.cap.release()

    def _do_calibration(self):
        if not self.cap or not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Calibrate", "Start Test first"); return
        QtWidgets.QMessageBox.information(self, "Calibrate", "Hold still for 5 s…")
        end = time.time() + 5; last = None
        while time.time() < end:
            ok, frm = self.cap.read();  QtWidgets.QApplication.processEvents()
            if ok:
                pt = get_darkest_area(crop_to_aspect_ratio(frm))
                if pt: last = pt
        if last:
            self.calibrated_center = last
            QtWidgets.QMessageBox.information(self, "Calibrated", f"Center = {last}")
        else:
            QtWidgets.QMessageBox.warning(self, "Calibrate", "No contour found")

    # ── frame processing ----------------------------------------------
    def _grab_frame(self):
        if not self.cap: return
        ok, frm = self.cap.read();  now = time.time()
        if not ok: return

        im = crop_to_aspect_ratio(frm)
        pt = get_darkest_area(im)
        g  = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        dv = g[pt[1], pt[0]] if pt else g.min()
        thr = apply_binary_threshold(g, dv, 15)
        msk = mask_outside_square(thr, pt, 250) if pt else thr
        dil = cv2.dilate(msk, np.ones((5, 5), np.uint8), 2)
        cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flt = filter_contours_by_area_and_return_largest(cnts, 1000, 3)

        center = None
        if flt and len(flt[0]) > 5:
            ell = cv2.fitEllipse(flt[0]); center = tuple(map(int, ell[0]))
            cv2.ellipse(im, ell, (0, 255, 0), 2); cv2.circle(im, center, 3, (255, 255, 0), -1)

        if hasattr(self, "last_time") and self.last_time:
            fps = int(1 / (now - self.last_time))
            cv2.putText(im, f"FPS:{fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.last_time = now

        if self.calibrated_center and center:
            dx = center[0] - self.calibrated_center[0]
            dy = center[1] - self.calibrated_center[1]
            dirs = []
            if dy < -self.sensitivity_y: dirs.append("Arriba")
            if dy >  self.sensitivity_y: dirs.append("Abajo")
            if dx < -self.sensitivity_x: dirs.append("Izquierda")
            if dx >  self.sensitivity_x: dirs.append("Derecha")
            txt = " ".join(dirs) or "Centro"
            cv2.putText(im, txt, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.lbl.width(), self.lbl.height()))
        qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.lbl.setPixmap(QtGui.QPixmap.fromImage(qimg))

    # ── theme tweaks ---------------------------------------------------
    def _apply_theme(self, dark: bool):
        if dark:
            self.cmb.setStyleSheet(""); self.sld_x.setStyleSheet(""); self.sld_y.setStyleSheet("")
        else:
            self.cmb.setStyleSheet(
                "QComboBox{background:white;color:#0d1117;border:1px solid %s}" % styles.NEON_BLUE)
            groove = ("QSlider::groove:horizontal{background:#ccc;height:8px;border-radius:4px}"
                      "QSlider::handle:horizontal{background:%s;width:16px;margin:-4px}" %
                      styles.NEON_BLUE)
            self.sld_x.setStyleSheet(groove); self.sld_y.setStyleSheet(groove)

    # ── cleanup --------------------------------------------------------
    def closeEvent(self, e):
        if self.timer.isActive(): self.timer.stop()
        if self.cap: self.cap.release()
        super().closeEvent(e)
