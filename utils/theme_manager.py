# utils/theme_manager.py
from PyQt5 import QtWidgets, QtCore, QtGui
import styles                                   # tu paleta

class ThemeManager(QtCore.QObject):
    themeChanged = QtCore.pyqtSignal(bool)      # True = dark, False = light
    _instance    = None

    # ── singleton ────────────────────────────────────────────────────
    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    # ── init ─────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self._dark = True
        self._apply_stylesheet()                # seguro (ver función abajo)

    # ── API ──────────────────────────────────────────────────────────
    @property
    def is_dark(self) -> bool:
        return self._dark

    def toggle(self):
        self._dark = not self._dark
        self._apply_stylesheet()
        self.themeChanged.emit(self._dark)

    def tinted_icon(self, svg: str, size: QtCore.QSize) -> QtGui.QIcon:
        from PyQt5.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(svg)
        pix = QtGui.QPixmap(size)
        pix.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pix)
        renderer.render(painter)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        fg = "#e6edf3" if self._dark else "#0d1117"
        painter.fillRect(pix.rect(), QtGui.QColor(fg))
        painter.end()
        return QtGui.QIcon(pix)

    # ── internal ─────────────────────────────────────────────────────
    def _apply_stylesheet(self):
        """
        Aplica la QSS a la aplicación **solo** si QApplication ya existe.
        Si aún no existe, reintenta al próximo ciclo del event-loop.
        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            # Ejecuta una sola vez cuando el event-loop ya esté vivo
            QtCore.QTimer.singleShot(0, self._apply_stylesheet)
            return

        css = styles.DARK if self._dark else styles.LIGHT
        app.setStyleSheet(css)
