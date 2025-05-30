![Open-Source-BCI-EEG](icons/logo.svg)

A fully open-source Python toolkit for **brain–computer-interface (BCI)** and **electroencephalography (EEG)** experimentation.  
It bundles a real-time monitor, a model-training front-end, and a single launcher script so you can stream data from affordable hardware, explore spectral features, and prototype machine-learning pipelines in minutes.

---

## Quick Start
git clone https://github.com/skijer/Open-Source-BCI-EEG.git
cd Open-Source-BCI-EEG
python -m venv .venv && source .venv/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python bci_launcher.py

##Requirements
Category	Core packages	Notes
GUI & plotting	PyQt5, pyqtgraph	Fast, OpenGL-backed real-time graphs 
pyqtgraph.readthedocs.io
PyPI
Signal processing	numpy, scipy, mne, pandas	End-to-end EEG pipeline 
mne.tools
Machine learning	tensorflow or scikit-learn	GPU optional (TF ≥ 2.16 works with CUDA 12.x) 
PyPI
Vision / eye-tracking	opencv-python, Pillow	Optional hybrid input
Packaging	pyinstaller	To build standalone apps when ready

Tip for NVIDIA GPUs: install the matching CUDA toolkit 12.x before running pip install tensorflow==2.16.*; TF auto-detects the GPU at runtime 
Python GUIs
.
Project Structure
bci_monitor/      # Qt-based live viewer (time & frequency domains)
bci_trainer/      # Model training / evaluation UI
models/           # Placeholder for future pretrained networks (EEGNet, etc.)
utils/            # Serial I/O, config handling, DSP helpers
installer/        # (unused) packaging scripts kept for reference
config.json       # Global settings (baud rate, SR, channel names…)
requirements.txt  # Exact dependency versions
Compatible Hardware (tested)
Device	Connection	Status
YuEEG ADS1299 (8-ch)	USB / UART	✅ stable stream 
openbci.com
TI
OpenBCI Cyton (8-ch)	USB Dongle	✅ verified at 250 Hz 
OpenBCI Shop
Generic Serial/OSC boards	UART / TCP	⚠️ experimental
.

##Future plans

Interactive Jupyter tutorials

Pre-trained CNNs (EEGNet & variants) as downloadable assets 
GitHub

Contributing
Fork → git checkout -b feature-myIdea

Follow PEP-8; add docstrings where relevant.

Run pytest (coming soon) and ensure lint passes.

Open a pull request describing what and why.

Issues, ideas and hardware reports are very welcome!

License
Open-Source-BCI-EEG is released under the MIT License.
See LICENSE for the full text.

Acknowledgements
This project stands on the shoulders of outstanding open-source work:

PyQtGraph – ultra-fast scientific plotting 
pyqtgraph.readthedocs.io

PyQt5 / Qt 5 – cross-platform GUI toolkit 
PyPI

MNE-Python – modern EEG/MEG analysis suite 
mne.tools

OpenBCI – affordable open hardware for biosensing 
OpenBCI Shop

ADS1299 reference designs by TI & community 
TI and Yu Tao YuEEG circuit help

JEOResearch for reference with the eyetracker

BrainFlow – multi-device streaming library 
brainflow.org

LabStreamingLayer – networked synchronisation standard 
labstreaminglayer.org

EEGNet – compact CNN architecture for BCI 
GitHub
