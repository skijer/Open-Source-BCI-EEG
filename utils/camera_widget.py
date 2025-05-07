# ───────────────────────────────────────── utils/camera_widget.py ───
# Stand‑alone pupil tracker → vJoy   or   keyboard fallback (WASD)    #
# --------------------------------------------------------------------
import os, json, time, math, random, cv2, numpy as np
from PyQt5 import QtCore, QtWidgets
from utils.ui_helpers    import nav_bar
from utils.theme_manager import ThemeManager

# ───────────── unified input binding helper ─────────────────────────
try:
    import pyvjoy
    _VJOY_OK = True
except ImportError:
    _VJOY_OK = False

import pyautogui
pyautogui.FAILSAFE = False

# ────────────── default camera‑direction → key map (W A S D) ────────
DEFAULT_CAM_MAP = {
    ("up",)       : "w",
    ("down",)     : "s",
    ("left",)     : "a",
    ("right",)    : "d",
    ("adelante",) : "space",
    ("atras",)    : "space",
}

# ────────────── human‑readable logger ───────────────────────────────
def _log(msg: str):
    print(msg)
    cv2.setWindowTitle("Frame with Ellipse and Rays", f"{msg} – ESC to quit")

def _send_input(action: str, vj: "pyvjoy.VJoyDevice|None" = None):
    """Send a vJoy button (if action is digit) or a keyboard key."""
    if vj and action.isdigit():
        idx = int(action)
        vj.set_button(idx, 1)
        time.sleep(0.04)
        vj.set_button(idx, 0)
    else:
        pyautogui.press(action.lower())

# ────────────── persistent dead‑zone config ─────────────────────────
_CFG = os.path.join(os.path.dirname(__file__), "eye_config.json")
def _load_cfg():
    try:  return json.load(open(_CFG))
    except Exception:
        return dict(x_min=-30, x_max=20, y_min=-30, y_max=20,
                    z_min=-30, z_max=20)
def _save_cfg(d): json.dump(d, open(_CFG,"w"), indent=2)

# heavy vision code lives elsewhere
from utils.Orlosky3DEyeTracker import process_frame

# ────────────────────────── Camera → bindings worker ────────────────
class CameraWorker(QtCore.QThread):
    def __init__(self, cam_idx: int, mapping: dict|None, dz_conf: dict):
        super().__init__()
        self.cam_idx = cam_idx
        self.map     = mapping or DEFAULT_CAM_MAP.copy()
        self.dz      = dz_conf
        self._stop   = False

        self.vj = None
        if _VJOY_OK:
            try:    self.vj = pyvjoy.VJoyDevice(1)
            except: self.vj = None
        if not self.vj:
            print("⚠  vJoy not available – using keyboard")

    # -----------------------------------------------------------------
    # ────────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("⚠  Camera cannot be opened"); return

        xm,xM = self.dz["x_min"], self.dz["x_max"]
        ym,yM = self.dz["y_min"], self.dz["y_max"]
        zm,zM = self.dz["z_min"], self.dz["z_max"]

        while not self._stop:
            ok, frm = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            res = process_frame(frm)
            if not res:
                continue
            (_, _, _), gaze = res
            if gaze is None:
                continue

            x, y, z = gaze * 100

            # build logical directions
            dirs = []
            if   x > xM: dirs.append(("right",))
            elif x < xm: dirs.append(("left",))
            if   y > yM: dirs.append(("down",))
            elif y < ym: dirs.append(("up",))
            if   z > zM: dirs.append(("adelante",))
            elif z < zm: dirs.append(("atras",))

            messages = [f"axes=({int(x):+4d},{int(y):+4d},{int(z):+4d})"]

            # ========== vJoy =================================================
            if self.vj:
                def _norm(v): return int((max(-1, min(1, v/100.)) + 1) / 2 * 0x8000)
                self.vj.set_axis(pyvjoy.HID_USAGE_X, _norm(x))
                self.vj.set_axis(pyvjoy.HID_USAGE_Y, _norm(y))
                self.vj.set_axis(pyvjoy.HID_USAGE_Z, _norm(z))

                b1 = int(z > zM)
                b2 = int(z < zm)
                self.vj.set_button(1, b1)
                self.vj.set_button(2, b2)
                if b1: messages.append("btn1")
                if b2: messages.append("btn2")

                for d in dirs:
                    messages.append(d[0])

                print("vJoy  → " + " | ".join(messages), flush=True)
                cv2.setWindowTitle("Frame with Ellipse and Rays",
                                   "vJoy  → " + " | ".join(messages))
            # ========== keyboard fallback ===================================
            else:
                for d in dirs:
                    key = self.map.get(d)
                    if key:
                        _send_input(key)
                        messages.append(key.upper())
                print("KB    → " + " | ".join(messages), flush=True)
                cv2.setWindowTitle("Frame with Ellipse and Rays",
                                   "KB    → " + " | ".join(messages))

            time.sleep(0.05)

        cap.release()

    def stop(self): self._stop = True

# ────────────────────────── On‑screen tester widget ─────────────────
class CameraWidget(QtWidgets.QWidget):
    def __init__(self, launcher=None, parent=None):
        super().__init__(parent)
        self.launcher = launcher
        self.cap = None
        self.timer = QtCore.QTimer(self, timeout=self._grab)

        cfg = _load_cfg()

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(nav_bar(self, self._return, lambda:None), 0)

        form = QtWidgets.QFormLayout()
        for ax in "XYZ":
            row = QtWidgets.QHBoxLayout()
            sb_min = QtWidgets.QSpinBox(); sb_min.setRange(-100,100); sb_min.setValue(cfg[f"{ax.lower()}_min"])
            sb_max = QtWidgets.QSpinBox(); sb_max.setRange(-100,100); sb_max.setValue(cfg[f"{ax.lower()}_max"])
            setattr(self,f"sb_{ax.lower()}_min",sb_min)
            setattr(self,f"sb_{ax.lower()}_max",sb_max)
            sb_min.valueChanged.connect(self._save); sb_max.valueChanged.connect(self._save)
            row.addWidget(sb_min); row.addWidget(sb_max)
            form.addRow(f"{ax} dead‑zone [min — max]:", row)
        lay.addLayout(form)

        self.btn = QtWidgets.QPushButton("▶ Test", clicked=self._toggle)
        self.lbl = QtWidgets.QLabel("Obs: —")
        lay.addWidget(self.btn, QtCore.Qt.AlignCenter)
        lay.addWidget(self.lbl, QtCore.Qt.AlignCenter)

        ThemeManager.instance().themeChanged.connect(lambda _: None)

    # ------------------------ helpers -------------------------------
    def _save(self):
        _save_cfg(dict(
            x_min=self.sb_x_min.value(), x_max=self.sb_x_max.value(),
            y_min=self.sb_y_min.value(), y_max=self.sb_y_max.value(),
            z_min=self.sb_z_min.value(), z_max=self.sb_z_max.value()
        ))

    def _toggle(self):
        if self.timer.isActive(): return self._stop()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cv2.startWindowThread()
        cv2.namedWindow("Frame with Ellipse and Rays", cv2.WINDOW_NORMAL)
        self.timer.start(0); self.btn.setText("■ Stop")

    def _grab(self):
        ok, frm = self.cap.read() if self.cap else (False,None)
        if not ok: return self._stop()
        res = process_frame(frm)
        if not res: return
        (_,_,_), gaze = res
        if gaze is None: return
        x,y,z = gaze*100
        xm,xM = self.sb_x_min.value(), self.sb_x_max.value()
        ym,yM = self.sb_y_min.value(), self.sb_y_max.value()
        zm,zM = self.sb_z_min.value(), self.sb_z_max.value()
        obs_x = "Centro" if xm<=x<=xM else ("Derecha" if x>xM else "Izquierda")
        obs_y = "Centro" if ym<=y<=yM else ("Abajo"   if y>yM else "Arriba")
        obs_z = "Centro" if zm<=z<=zM else ("Adelante" if z>zM else "Atrás")
        self.lbl.setText(f"Obs: {obs_x}, {obs_y}, {obs_z}")
        if cv2.waitKey(1)&0xFF==27:   # ESC quits window
            self._stop()

    def _stop(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        self.btn.setText("▶ Test")

    def _return(self):
        if self.timer.isActive(): self._stop()
        if self.launcher and hasattr(self.launcher,"stack"):
            self.launcher.stack.setCurrentIndex(0)
