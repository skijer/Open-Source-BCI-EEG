![Open-Source-BCI-EEG](icons/logo.svg)

A fully open‑source Python toolkit for **brain–computer‑interface (BCI)** and **electroencephalography (EEG)** experimentation.  
It bundles a real‑time monitor, a model‑training front‑end, and a single launcher script so you can stream data from affordable hardware, explore spectral features, and prototype machine‑learning pipelines in minutes.

---

## Quick Start

```bash
git clone https://github.com/skijer/Open-Source-BCI-EEG.git
cd Open-Source-BCI-EEG
python -m venv .venv && source .venv/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python bci_launcher.py
```

---

## Requirements

| Category           | Core packages                               | Notes |
|--------------------|---------------------------------------------|-------|
| GUI & plotting     | `PyQt5`, `pyqtgraph`                        | Fast, OpenGL‑backed real‑time graphs |
| Signal processing  | `numpy`, `scipy`, `mne`, `pandas`           | End‑to‑end EEG pipeline |
| Machine learning   | `tensorflow` **or** `scikit-learn`          | GPU optional (TF ≥ 2.16 supports CUDA 12.x) |
| Vision / eye‑track | `opencv-python`, `Pillow`                   | Optional hybrid input |
| Packaging          | `pyinstaller`                               | Build standalone apps when ready |

> **Tip for NVIDIA GPUs:** install the matching CUDA toolkit 12.x before running    `pip install tensorflow==2.16.*`; TensorFlow auto‑detects the GPU at runtime.

---

## Project Structure

```text
bci_monitor/      # Qt‑based live viewer (time & frequency domains)
bci_trainer/      # Model training / evaluation UI
models/           # Placeholder for future pretrained networks (EEGNet, etc.)
utils/            # Serial I/O, config handling, DSP helpers
installer/        # (unused) packaging scripts kept for reference
config.json       # Global settings (baud rate, SR, channel names…)
requirements.txt  # Exact dependency versions
```

---

## Compatible Hardware (tested)

| Device                       | Connection | Status |
|------------------------------|------------|--------|
| **YuEEG ADS1299 (8‑ch)**     | USB / UART | ✅ stable stream |
| **OpenBCI Cyton (8‑ch)**     | USB Dongle | ❔ *not tested yet* |
| Generic Serial/OSC boards    | UART / TCP | ⚠️ experimental |

Planned integrations: **BrainFlow** SDK for 30+ boards and **LabStreamingLayer (LSL)** for network synchronisation.

---

## Mobile VR Companion App

We built a lightweight Unity‑based VR companion (working name **Phone VR Playground**) for cardboard‑style headsets.  
It mirrors 3‑D scenes rendered on the PC, receives gyro data, and forwards EEG‑based commands from `Open‑Source‑BCI‑EEG` back to the game engine.

* **Download (Android APK)**: [`PhoneVRPlayground.apk`](https://github.com/skijer/Open-Source-BCI-EEG/releases/latest/download/PhoneVRPlayground.apk)  
* **iOS TestFlight**: currently in closed beta – request an invite on the [issues tab](https://github.com/skijer/Open-Source-BCI-EEG/issues).  
* **Open‑sourcing plan**: the Unity project will become public once we finish cleaning proprietary art assets and documenting the networking layer. Stay tuned! 🚀

---

## Roadmap (next features only)

- Interactive Jupyter tutorials  
- Pre‑trained CNNs (EEGNet & variants) as downloadable assets  
- Full open‑source release of the **Phone VR Playground** Unity project
- 
Offline replay of a saved BDF/EDF/CSV file:

```bash
python bci_monitor/main_monitor.py --file example.bdf
```


---

## Contributing

1. **Fork** → `git checkout -b feature-myIdea`  
2. Follow **PEP‑8**; add docstrings where relevant.  
3. Run `pytest` (coming soon) and ensure lint passes.  
4. Open a pull request describing *what* and *why*.

Issues, ideas and hardware reports are very welcome!

---

## License

`Open-Source-BCI-EEG` is released under the **MIT License**.  
See [LICENSE](LICENSE) for the full text.

---

## Acknowledgements & Related Projects

* **PyQtGraph** – ultra‑fast scientific plotting  
* **PyQt5 / Qt 5** – cross‑platform GUI toolkit  
* **MNE‑Python** – modern EEG/MEG analysis suite  
* **OpenBCI** – affordable open hardware for biosensing  
* **ADS1299** reference designs by Texas Instruments & community  
* **EEGNet** – compact CNN architecture for BCI  
* **YuEEG** – 8‑channel ADS1299 BCI device & software ([GitHub](https://github.com/YuTaoV5/YuEEG))  
* **JEOresearch** – eye‑tracking & XR‑BCI research group led by Dr. Jason Orlosky ([Website](https://www.jeoresearch.com/))  
