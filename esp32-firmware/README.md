# ESP32 + AD8232 Hardware Setup Guide for সহায়ক (Sahayak)

This guide accompanies **Phase 6 (Hardware Integration)** of the Sleep Apnea Detection project.

---

## ⚡ Tinkercad / Fritzing Simulation & Circuit Visualizer

An interactive high-definition circuit simulation diagram is available in this folder:
- **Interactive Visualizer**: Open [`tinkercad_hardware_simulation.html`](tinkercad_hardware_simulation.html) in any web browser to view the breadboard layout, color-coded jumper wiring, and Einthoven 3-lead placement.

---

## 📌 Pinout & Connections

| AD8232 ECG Sensor Pin | ESP32 Dev Board Pin | Wire Color | Description |
| :--- | :--- | :--- | :--- |
| **3.3V** | **3V3** | 🔴 Red | 3.3V Regulated Power Supply |
| **GND** | **GND** | ⚫ Black | System Ground Reference |
| **OUTPUT** | **GPIO 36 (VP / ADC1_CH0)** | 🔵 Blue | Analog ECG Signal (0 - 3.3V) |
| **LO+** | **GPIO 22** | 🟠 Orange | Leads-off detection (optional) |
| **LO-** | **GPIO 23** | 🟢 Green | Leads-off detection (optional) |

---

## 🧪 Testing Without Patients (Benchtop HIL Simulation)

To demonstrate the full hardware pipeline to professors or reviewers without requiring real human patients:

### Option A: Direct Sensor Ambient Testing
Even without placing electrodes on a human, touching the electrodes lightly or connecting a 1 Hz function generator proves the ESP32 reads analog signals, digitizes them at 100 Hz, and streams JSON packets over Wi-Fi.

### Option B: DAC Playback (Loopback)
1. ESP32 DAC pin **GPIO 25** can output simulated PhysioNet Apnea-ECG waveforms as analog voltages.
2. Connect a jumper wire from **GPIO 25 (DAC)** to **AD8232 INPUT** or **GPIO 36**.
3. The hardware processes live physical electrical signals and transmits them to your live website at `https://sahayak-medai.vercel.app`.
