import json, pathlib
from typing import Any, Dict

CONFIG_FILE = pathlib.Path(__file__).with_name("config.json")

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
    if CONFIG_FILE.exists():
        try:
            on_disk = json.loads(CONFIG_FILE.read_text())
            return {**DEFAULTS, **on_disk}
        except:
            pass
    return dict(DEFAULTS)

_cfg = _read()

def get(key: str) -> Any:
    return _cfg.get(key, DEFAULTS[key])

def all() -> Dict[str, Any]:
    return dict(_cfg)

def save(**kw) -> None:
    _cfg.update(kw)
    CONFIG_FILE.write_text(json.dumps(_cfg, indent=2))

def reload(cfg_dict: Dict[str, Any]) -> None:
    global _cfg
    _cfg = {**DEFAULTS, **cfg_dict}
    CONFIG_FILE.write_text(json.dumps(_cfg, indent=2))
