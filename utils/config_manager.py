import json
import pathlib
import sys
import os
from typing import Any, Dict
_base_dir = getattr(sys, '_MEIPASS', None) or os.path.dirname(__file__)

# Ahora apunta a utils/config.json dentro de _base_dir
CONFIG_FILE = pathlib.Path(_base_dir) / 'utils' / 'config.json'

# Defaults en caso de que no exista el JSON
DEFAULTS: Dict[str, Any] = {
    "DATA_LENGTH":    30000,
    "PLOT_LENGTH":    2000,
    "FFT_LENGTH":     120,
    "FFT_FREQ_MIN":   3.0,
    "FFT_FREQ_MAX":   50.0,
    "UPDATE_INTERVAL":40,
    "SAMPLE_RATE":    500,
    "NOTCH_FREQ":     60.0,
    "QUALITY_FACTOR": 30,
    "BANDPASS_LO":    4.0,
    "BANDPASS_HI":    60.0,
    "BUTTER_ORDER":   4,
    "RECORD_LENGTH":  5
}

def _read() -> Dict[str, Any]:
    try:
        if CONFIG_FILE.exists():
            on_disk = json.loads(CONFIG_FILE.read_text())
            # Merge defaults con lo de disco (disk overrides defaults)
            return {**DEFAULTS, **on_disk}
    except Exception:
        # si hay cualquier fallo, ignoramos y devolvemos defaults
        pass
    return dict(DEFAULTS)

_cfg = _read()

def get(key: str) -> Any:
    return _cfg.get(key, DEFAULTS[key])

def all() -> Dict[str, Any]:
    return dict(_cfg)

def save(**kw) -> None:
    _cfg.update(kw)
    # Asegura que la carpeta exista
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(_cfg, indent=2))

def reload(cfg_dict: Dict[str, Any]) -> None:
    global _cfg
    _cfg = {**DEFAULTS, **cfg_dict}
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(_cfg, indent=2))
