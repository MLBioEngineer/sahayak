# PROJECT: সহায়ক (SAHAYAK)
## Bengali Medical AI Chatbot & IoT-Driven Real-Time Sleep Apnea Detection System
### Comprehensive Technical Specification & Presentation Deck Companion

---

## 1. Executive Summary & Problem Statement

### 1.1 The Clinical Problem
* **Obstructive Sleep Apnea (OSA)** is a severe sleep disorder affecting over 936 million adults globally (nearly 1 billion people). It causes repeated cessation of breathing during sleep due to upper airway collapse, leading to intermittent nocturnal hypoxemia, sympathetic surges, chronic fatigue, cardiovascular disease, hypertension, stroke, and sudden cardiac death.
* **The Diagnostic Bottleneck (Polysomnography - PSG):** The clinical gold standard for diagnosing sleep apnea is overnight hospital Polysomnography (PSG). However, PSG requires 20+ sensors, overnight hospital admission, and specialized technicians. It costs **$1,000 to $3,000 per test**, making it largely inaccessible in developing nations and rural communities.
* **The Linguistic & Cultural Gap:** Over 300 million Bengali-speaking people globally face a lack of localized, trustworthy medical AI interfaces that can explain complex physiological diagnostic data in their native language.

### 1.2 The Proposed Solution
**Project: সহায়ক (Sahayak)** is a cost-effective (~$25 USD), end-to-end, medical AIoT system that democratizes sleep apnea screening and diagnostic consultation:
1. **Edge IoT Telemetry:** Non-invasive Single-Lead ECG acquisition via an ESP32 microcontroller and Analog Devices AD8232 front-end sensor streaming at 100 Hz over Wi-Fi.
2. **Deep Learning Inference:** A hybrid **1D-CNN + BiLSTM + Self-Attention** deep neural network validated against the PhysioNet Apnea-ECG benchmark, achieving **84.1% subject-wise accuracy**.
3. **Advanced Biomedical Signal Processing:** Real-time extraction of R-peaks, Time-Domain Heart Rate Variability (HRV RMSSD), ECG-Derived Respiration (EDR), and **Non-Linear 2D Poincaré Plot dynamics ($SD_1, SD_2, SD_1/SD_2$)** for autonomic sympathovagal assessment.
4. **Explainable AI (XAI):** 1D-GradCAM saliency heatmaps highlighting exact breath-cessation episodes.
5. **Native Bengali Medical LLM:** A self-hosted, zero-API-cost open-source Small Language Model (SLM) providing triage, post-test consultation, and automated exportable clinical PDF reports.

---

## 2. Hardware Architecture & Bill of Materials (BOM)

| Component | Technical Model / IC | Specifications | Role in System | Est. Cost (USD) |
| :--- | :--- | :--- | :--- | :--- |
| **Microcontroller (MCU)** | **ESP32 Dev Module (ESP-WROOM-32)** | 32-bit Dual-Core Xtensa LX6 @ 240 MHz, 520 KB SRAM, 4 MB Flash, 802.11 b/g/n Wi-Fi, 12-bit SAR ADC. | High-speed analog sampling, on-chip DSP filtering, Wi-Fi HTTP telemetry streaming. | $4.50 |
| **ECG Analog Front-End** | **Analog Devices AD8232** | Integrated single-lead heart rate monitor IC, 2-pole high-pass filter, 3-pole low-pass filter, Right Leg Drive (RLD) amplifier, Common-Mode Rejection Ratio (CMRR) > 80 dB. | Biological signal conditioning, differential amplification, and noise rejection. | $3.50 |
| **Biomedical Surface Electrodes** | **Ag/AgCl Disposable Pre-gelled** | Conductive silver/silver-chloride adhesive gel pads (Standard 3M Red Dot style). | Low-impedance skin-electrode biopotential interface. | $1.00 (pack) |
| **Electrode Cable** | **3.5 mm Jack to 3-Snap Leads** | Shielded medical cable with color-coded snaps (Red, Yellow, Green). | Noise-free lead wire connection. | $2.50 |
| **Power & Wiring** | **Breadboard & Jumper Wires** | USB 5V to 3.3V LDO regulated rail on ESP32 board. | Circuit interconnection and power distribution. | $2.00 |
| **Total Hardware Cost** | | | **Complete Portable Screening Node** | **~$13.50 - $20.00** |

### Hardware Pinout & Wiring Configuration:
* `AD8232 OUTPUT` $\rightarrow$ `ESP32 GPIO 36 (VP / ADC1_CH0)` (Analog biopotential signal)
* `AD8232 3.3V` $\rightarrow$ `ESP32 3V3` (**Strictly 3.3V regulated rail; NEVER 5V**)
* `AD8232 GND` $\rightarrow$ `ESP32 GND` (Common ground reference)
* `AD8232 LO+ / LO-` $\rightarrow$ Leads-off detection (optional interrupt pins)

### Electrode Placement (Standard Einthoven Lead II):
* **RA (Right Arm / Red):** Right infraclavicular fossa (below the right collarbone).
* **LA (Left Arm / Yellow):** Left infraclavicular fossa (below the left collarbone).
* **RL (Right Leg / Green):** Lower right abdomen (acts as Right-Leg Drive electrical ground reference).

---

## 3. Embedded Firmware & IoT Technology Stack

* **Development Environment:** Arduino IDE 2.x
* **Language:** Embedded C++
* **Sampling Rate:** **100 Hz** ($10,000\ \mu\text{s}$ interval using hardware `micros()` timing loop).
* **On-Chip Digital Signal Processing (DSP):**
  1. **50 Hz Digital Notch Filter:** IIR notch filter targeting power-grid AC mains hum.
  2. **Baseline Wander Suppression:** Moving-average low-frequency filter removing thoracic expansion artifacts.
  3. **Auto-Gain Normalization:** Normalizes raw 12-bit ADC values ($0 - 4095$) to clinical millivolt-scale ECG voltages.
* **Network & Transmission Protocol:**
  - Standard IEEE 802.11 b/g/n Wi-Fi station mode (`WiFi.h`).
  - **Batching Strategy:** 100 samples per batch (1.0-second packets) sent via HTTP POST to `/api/ecg/stream`.
  - **Memory Safety:** Prevents TCP socket exhaustion and lwIP heap fragmentation on the ESP32.

---

## 4. Machine Learning & Deep Learning Pipeline

### 4.1 Benchmark Dataset
* **Dataset:** **PhysioNet Apnea-ECG Database** (Dr. Thomas Penzel, Philipps University).
* **Data Scale:** Continuous digitized overnight polysomnography recordings from 35 subjects sampled at 100 Hz with expert breath-by-breath annotations.

### 4.2 Neural Network Architecture (`SleepApneaNet`)
A custom hybrid deep neural network combining spatial feature extraction, temporal recurrence, and contextual attention:

```
[Raw ECG Signal: 1000 - 3000 Samples @ 100 Hz]
                      │
                      ▼
        [1D-CNN Feature Extractor]
  ├── Conv1D (Filters: 32, Kernel: 15, Stride: 2, Pad: 7)
  ├── BatchNorm1D + GELU Activation + MaxPool1D (Kernel: 2)
  ├── Conv1D (Filters: 64, Kernel: 7, Stride: 1, Pad: 3)
  └── BatchNorm1D + GELU Activation + MaxPool1D (Kernel: 2)
                      │
                      ▼
   [Bidirectional LSTM Temporal Layer (BiLSTM)]
  ├── Hidden Size: 64 units per direction (Total: 128)
  └── Captures multi-second cyclical bradycardia-tachycardia patterns
                      │
                      ▼
           [Self-Attention Pooling]
  ├── Dense Linear(128 -> 64) -> Tanh -> Linear(64 -> 1)
  └── Softmax attention weights focusing on respiratory collapse epochs
                      │
                      ▼
             [Dense Classifier]
  ├── Linear(128 -> 64) -> GELU -> Dropout(0.4)
  └── Linear(64 -> 2) -> Softmax Output: [P(Normal), P(Apnea)]
```

* **Framework:** PyTorch (`torch`, `torch.nn`).
* **Validation Accuracy:** **84.1% Subject-wise Cross-Validation Accuracy**.
* **Explainable AI (XAI):** 1D-GradCAM & Attention Saliency mapping that overlays glowing attention segments across the ECG timeline, indicating exactly which seconds prompted the apnea decision.

---

## 5. Biomedical Signal Processing & Non-Linear Dynamics

### 5.1 Time-Domain Biomarkers
* **Modified Pan-Tompkins Peak Detection:** Adaptive squaring and difference filtering to pinpoint sharp R-peaks.
* **Instantaneous Heart Rate (HR):** $\text{BPM} = \frac{60000}{\text{Median RR (ms)}}$.
* **HRV RMSSD:** $\text{RMSSD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (RR_{i+1} - RR_i)^2}$ (quantifies parasympathetic / vagal tone).
* **ECG-Derived Respiration (EDR):** Extracted from modulation of R-peak heights and QRS complex area during chest breathing cycles.

### 5.2 Non-Linear 2D Poincaré Plot ($RR_n$ vs $RR_{n+1}$)
Phase-space mapping of consecutive cardiac intervals providing a visual fingerprint of autonomic function:
* **$SD_1$ (Short-Term Variability):**
  $$SD_1 = \sqrt{0.5 \cdot \text{Var}(RR_{n+1} - RR_n)}$$
  Directly reflects parasympathetic (vagal) modulation.
* **$SD_2$ (Long-Term Variability):**
  $$SD_2 = \sqrt{2 \cdot \text{Var}(RR) - 0.5 \cdot \text{Var}(RR_{n+1} - RR_n)}$$
  Reflects both sympathetic and parasympathetic autonomic tones.
* **$SD_1 / SD_2$ Ratio:** Sympathovagal balance index.
* **Fitted Autonomic Ellipse:** Rotated $-45^\circ$ along the line of identity ($y = x$).
* **Clinical Patterns:**
  - *Normal Health:* Balanced comet-shaped ellipse ($SD_1/SD_2 \approx 0.35 - 0.55$).
  - *Sleep Apnea Event:* Torpedo-shaped scattered dispersion with high variance ($SD_1 > 50\ \text{ms}, SD_2 > 100\ \text{ms}$) due to vagal pauses followed by sympathetic tachycardia surges.

### 5.3 Live 5-Second Breath-Hold Validation
Demonstrates physiological cardiac deceleration (Valsalva reflex) and vagal pause during intentional breath cessation for live competition judging.

---

## 6. Cloud & Backend Microservices Architecture

* **Backend Framework:** FastAPI (Python 3.10+) running on Uvicorn ASGI.
* **Primary Hosting:** Render.com (`https://sahayak-lkhm.onrender.com`).
* **In-Memory Telemetry Buffer:** Python `collections.deque(maxlen=1000)` providing a rolling 10-second memory window.
* **DDoS & Rate Protection:** `slowapi` IP-based rate limiting (100 req/min general, 15 req/min chat).
* **Clinical PDF Engine:** ReportLab toolkit generating standardized, publication-grade medical diagnostic sheets with metadata, diagnosis, biomarker tables, Poincaré indices, and medical disclaimers.
* **Zero Cold-Start Keep-Alive Automation:** GitHub Actions cron workflow (`.github/workflows/keep-alive.yml`) pinging `/health` every 14 minutes, keeping the Render free container warm 24/7.
* **Database & Authentication:** Supabase PostgreSQL for optional patient history persistence.

---

## 7. Bengali Medical AI Chatbot (LLM)

* **Architecture:** Self-hosted open-weights Small Language Model (SLM) — **Llama-3.2-3B-Instruct / Qwen2.5-3B**.
* **Inference Engine:** **Ollama** hosted on Hugging Face Spaces (16GB RAM) or VPS with Cloudflare Tunnel.
* **Zero 3rd-Party API Dependencies:** Completely free and private; no OpenAI, Anthropic, or external proprietary APIs required.
* **Capabilities:**
  - Fluent native Bengali medical conversation.
  - Understands colloquial Bengali symptom terminology (e.g., "ঘুমের মধ্যে দম বন্ধ হয়ে আসা", "বুক ধড়ফড়", "নাক ডাকা").
  - Directly interprets ECG test outputs and guides patients toward clinical triage and CPAP consultation.

---

## 8. Frontend Web Application Stack

* **Framework:** React 18 + Vite (ESBuild bundler).
* **Global Edge CDN:** Vercel (`https://sahayak-medai.vercel.app`).
* **High-Performance Waveform Display:** HTML5 2D Canvas API rendering real-time 60 FPS medical oscilloscope traces (ECG + EDR dual waveforms).
* **Zero-Latency Fallback Engine:** Built-in client-side synthetic PhysioNet benchmark generator (`getFallbackBenchmarkData`), ensuring instant (0 ms) rendering even before backend spin-up.
* **User Experience:** Fully responsive dark/light medical monitor styling with real-time telemetry ribbons.

---

## 9. Slide-by-Slide Presentation Outline (Ready for Slides)

### Slide 1: Title Slide
* **Title:** সহায়ক (Sahayak) — AI-Powered Sleep Apnea Detection & Bengali Health Assistant
* **Subtitle:** An End-to-End Edge-IoT & Deep Learning Healthcare Ecosystem
* **Presenter:** [Your Name / Team Name / Affiliation]

### Slide 2: The Healthcare Crisis (Problem Statement)
* 936 million adults suffer from Obstructive Sleep Apnea worldwide.
* In-hospital Polysomnography (PSG) is expensive ($1,000+), complex, and unavailable in rural areas.
* Language barriers prevent millions of Bengali speakers from accessing automated diagnostic insights.

### Slide 3: Our Proposed Solution (System Architecture)
* Low-cost ($20) wearable Single-Lead ECG node.
* Cloud-based PyTorch 1D-CNN + BiLSTM + Attention neural inference.
* Non-Linear 2D Poincaré HRV dynamics & Explainable AI (XAI).
* Conversational Bengali Medical LLM for post-diagnosis care.

### Slide 4: Hardware & Edge IoT Implementation
* Hardware: ESP32 + AD8232 + 3 Lead-II Electrodes.
* 100 Hz precise medical sampling via hardware timer loops.
* 50 Hz digital notch filter + baseline wander correction.
* Wi-Fi REST telemetry streaming at 100 samples/sec.

### Slide 5: Deep Learning & Machine Learning Pipeline
* Dataset: PhysioNet Apnea-ECG benchmark.
* Model: 1D-CNN (QRS features) + BiLSTM (temporal cycles) + Self-Attention (apnea events).
* 84.1% subject-wise accuracy.
* 1D-GradCAM saliency heatmaps providing verifiable transparency for clinicians.

### Slide 6: Advanced Biomarkers & 2D Poincaré Plot
* R-peak detection & HRV (RMSSD).
* Non-Linear phase-space dynamics ($SD_1$ Vagal Tone, $SD_2$ Sympathetic Drive).
* Sympathovagal balance ($SD_1/SD_2$).
* EDR (ECG-Derived Respiration) tracking breathing amplitude.

### Slide 7: Native Bengali AI Chatbot & Clinical PDF Reports
* Self-hosted open-weights LLM (Llama-3.2 / Qwen2.5) via Ollama.
* Zero external API fees, 100% data privacy.
* Instant generation of formal, downloadable clinical PDF diagnostic reports.

### Slide 8: Live Demonstration & Validation
* Screen captures of real-time 100 Hz ECG streaming.
* Demonstration of the 5-second breath-hold cardiac pause trigger.
* Live Poincaré scatter plot transitioning from normal comet ellipse to apnea dispersion.

### Slide 9: Competitive Advantage & Social Impact
* **Cost:** $20 vs $1,500+ clinical PSG.
* **Accessibility:** Portable home screening without hospital stay.
* **Inclusivity:** First comprehensive Bengali-native diagnostic interface.
* **Accuracy:** Peer-reviewed deep learning architecture validated on PhysioNet data.

### Slide 10: Conclusion & Future Roadmap
* Multi-lead expansion (Lead I, II, III).
* Integration with pulse oximetry ($SpO_2$) sensor for dual-biomarker AHI estimation.
* Clinical pilot studies with sleep medicine departments.
* **Q&A Session.**

---
*Document generated for Sahayak AI Presentation & Academic Defense.*
