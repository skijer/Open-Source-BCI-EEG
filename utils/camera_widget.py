# ───────────────────────────────────────── utils/camera_widget.py ───
# Stand‑alone pupil tracker  ⟶  keyboard (W A S D + SPACE)            #
# --------------------------------------------------------------------
import os, json, time, cv2, pyautogui
import numpy as np
from PyQt5 import QtCore, QtWidgets
from utils.ui_helpers    import nav_bar
from utils.theme_manager import ThemeManager
from utils.Orlosky3DEyeTracker import process_frame   # pupil+gaze detector

pyautogui.FAILSAFE = False

# ───────────── direction → key map ──────────────────────────────────
DIR_TO_KEY = {
    "arriba"   : "w",
    "abajo"    : "s",
    "izquierda": "a",
    "derecha"  : "d",
    "atras"    : "space",  # lean backwards → space bar
}

# ───────────── persistent dead‑zone config ──────────────────────────
_CFG = os.path.join(os.path.dirname(__file__), "eye_config.json")
def _load_cfg():
    try:
        return json.load(open(_CFG))
    except Exception:
        return dict(x_min=-30, x_max=20,
                    y_min=-30, y_max=20,
                    z_min=-30, z_max=20)
def _save_cfg(d): json.dump(d, open(_CFG,"w"), indent=2)

# ───────────── helper: print & title bar update ─────────────────────
def _log(msg: str):
    print(msg, flush=True)
    cv2.setWindowTitle("Frame with Ellipse and Rays", msg)

# ────────────────────────── UI widget (tester) ──────────────────────
class CameraWidget(QtWidgets.QWidget):
    def __init__(self, launcher=None, parent=None):
        super().__init__(parent)
        self.launcher = launcher
        self.cap  = None
        self.timer = QtCore.QTimer(self, timeout=self._grab)

        cfg = _load_cfg()

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(nav_bar(self, self._return, lambda: None), 0)

        # dead‑zone spinners -------------------------------------------------
        form = QtWidgets.QFormLayout()
        for ax in "XYZ":
            row = QtWidgets.QHBoxLayout()
            sb_min = QtWidgets.QSpinBox(); sb_min.setRange(-100,100); sb_min.setValue(cfg[f"{ax.lower()}_min"])
            sb_max = QtWidgets.QSpinBox(); sb_max.setRange(-100,100); sb_max.setValue(cfg[f"{ax.lower()}_max"])
            setattr(self, f"sb_{ax.lower()}_min", sb_min)
            setattr(self, f"sb_{ax.lower()}_max", sb_max)
            sb_min.valueChanged.connect(self._save)
            sb_max.valueChanged.connect(self._save)
            row.addWidget(sb_min); row.addWidget(sb_max)
            form.addRow(f"{ax} dead‑zone [min — max]:", row)
        lay.addLayout(form)

        # control buttons / labels ------------------------------------------
        self.btn = QtWidgets.QPushButton("▶ Test", clicked=self._toggle)
        self.lbl = QtWidgets.QLabel("Obs: —")
        lay.addWidget(self.btn, alignment=QtCore.Qt.AlignCenter)
        lay.addWidget(self.lbl, alignment=QtCore.Qt.AlignCenter)

        ThemeManager.instance().themeChanged.connect(lambda _: None)

    # ---------------------------------------------------------------------
    def _save(self):
        _save_cfg(dict(
            x_min=self.sb_x_min.value(), x_max=self.sb_x_max.value(),
            y_min=self.sb_y_min.value(), y_max=self.sb_y_max.value(),
            z_min=self.sb_z_min.value(), z_max=self.sb_z_max.value()
        ))

    def _toggle(self):
        if self.timer.isActive():
            self._stop(); return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self,"Camera","Could not open camera"); return
        cv2.startWindowThread()
        cv2.namedWindow("Frame with Ellipse and Rays", cv2.WINDOW_NORMAL)
        self.timer.start(0)
        self.btn.setText("■ Stop")

    # ========================== MAIN LOOP =================================
    def _grab(self):
        ok, frm = self.cap.read() if self.cap else (False,None)
        if not ok:
            self._stop(); return

        res = process_frame(frm)
        if not res:
            return
        (_,_,_), gaze = res
        if gaze is None:
            return

        x, y, z = gaze * 100
        xm,xM = self.sb_x_min.value(), self.sb_x_max.value()
        ym,yM = self.sb_y_min.value(), self.sb_y_max.value()
        zm,zM = self.sb_z_min.value(), self.sb_z_max.value()

        # logical observations ---------------------------------------------
        obs_x = "centro"
        if x > xM: obs_x = "derecha"
        elif x < xm: obs_x = "izquierda"

        obs_y = "centro"
        if y > yM: obs_y = "abajo"
        elif y < ym: obs_y = "arriba"

        obs_z = "centro"
        if z < zm: obs_z = "atras"     # backwards only
        # adelante ignored for now

        # fire corresponding key presses -----------------------------------
        fired = []
        for obs in (obs_x, obs_y, obs_z):
            key = DIR_TO_KEY.get(obs)
            if key:
                pyautogui.press(key)
                fired.append(key.upper())

        # UI / logging
        self.lbl.setText(f"Obs: {obs_x.capitalize()}, {obs_y.capitalize()}, {obs_z.capitalize()}")
        if fired:
            _log("Pressed " + " ".join(fired) +
                 f"   (axes {int(x):+4d},{int(y):+4d},{int(z):+4d})")

        # quit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            self._stop()

    # ---------------------------------------------------------------------
    def _stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.btn.setText("▶ Test")

    def _return(self):
        if self.timer.isActive():
            self._stop()
        if self.launcher and hasattr(self.launcher,"stack"):
            self.launcher.stack.setCurrentIndex(0)
