# ESP32 + AD8232 Hardware Setup Guide for সহায়ক (Sahayak)

This guide accompanies **Phase 6 (Hardware Integration)** of the Sleep Apnea Detection project.

---

## 📌 Pinout & Connections

| AD8232 ECG Sensor Pin | ESP32 Dev Board Pin | Description |
| :--- | :--- | :--- |
| **3.3V** | **3V3** | 3.3V Power Supply |
| **GND** | **GND** | Ground |
| **OUTPUT** | **GPIO 36 (VP / ADC1_CH0)** | Analog ECG signal |
| **LO+** | **GPIO 22** | Leads-off detection (optional) |
| **LO-** | **GPIO 23** | Leads-off detection (optional) |

---

## 🧪 Testing Without Patients (Benchtop HIL Simulation)

To demonstrate the full hardware pipeline to professors or reviewers without requiring real human patients:

### Option A: Direct Sensor Ambient Testing
Even without placing electrodes on a human, touching the electrodes lightly or connecting a 1 Hz function generator proves the ESP32 reads analog signals, digitizes them at 100 Hz, and streams JSON packets over Wi-Fi.

### Option B: DAC Playback (Loopback)
1. ESP32 DAC pin **GPIO 25** can output simulated PhysioNet Apnea-ECG waveforms as analog voltages.
2. Connect a jumper wire from **GPIO 25 (DAC)** to **AD8232 INPUT** or **GPIO 36**.
3. The hardware processes live physical electrical signals and transmits them to your live website at `https://sahayak-medai.vercel.app`.
