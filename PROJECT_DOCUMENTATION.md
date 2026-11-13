# Project সহায়ক (Sahayak): Comprehensive Technical & Clinical Project Documentation
**IoT-Driven Real-Time Sleep Apnea Detection from Single-Lead ECG via Deep 1D-CNN + BiLSTM + Attention Network**

---

### Project Information
* **Lead Researcher & Engineer:** Fazle Ilahi (MLBioEngineer)  
* **Academic Discipline:** Department of Biomedical Engineering  
* **Academic Year:** 2026  
* **Contact:** ilahi0184@gmail.com  
* **Live Telemedicine Web Suite:** [https://sahayak-medai.vercel.app](https://sahayak-medai.vercel.app)  
* **Cloud API Gateway:** [https://sahayak-lkhm.onrender.com](https://sahayak-lkhm.onrender.com)  
* **Open Source Repository:** [https://github.com/MLBioEngineer/sahayak](https://github.com/MLBioEngineer/sahayak)  
* **Local Interactive Presentation:** [`presentation_slides.html`](file:///D:/sahayak%20AI/presentation_slides.html)  
* **Local Academic Poster File:** [`sahayak_academic_poster_presentation.html`](file:///D:/sahayak%20AI/sahayak_academic_poster_presentation.html)  

---

## 1. Executive Abstract
**Obstructive Sleep Apnea (OSA)** is an under-diagnosed respiratory sleep disorder affecting approximately 1 billion individuals globally, characterized by recurrent nocturnal pharyngeal airway collapse, intermittent hypoxemia, autonomic sympathetic surges, and severely increased risks of stroke, refractory hypertension, and sudden cardiac death. The gold-standard diagnostic modality, in-laboratory overnight **Polysomnography (PSG)**, presents an unsustainable clinical bottleneck due to exorbitant costs ($1,000–$2,500/night), extreme physical intrusiveness (over 20 attached sensors), and 6–12 month hospital waiting queues, rendering it virtually inaccessible to rural and low-income populations.

**Project সহায়ক (Sahayak)** introduces an end-to-end, ultra-low-cost (~$15 Bill of Materials), wearable telemedicine platform that democratizes clinical sleep apnea screening using a single Lead-II electrocardiogram (ECG) channel. The system features a custom IoT biopotential acquisition node powered by a dual-core ESP32 and an Analog Devices AD8232 front-end, operating with a strict 100 Hz hardware timer interrupt. From the single ECG lead, the platform simultaneously extracts cardiac electrical activity and **ECG-Derived Respiration (EDR)** by reconstructing thoracic electrical impedance modulations, enabling direct visual confirmation of respiratory cessation flatlining during apnea events. 

Diagnostic classification is powered by **SleepApneaNet**, an ultra-lightweight (~128,000 parameters) hierarchical deep learning architecture combining multi-scale 1D-CNN feature extraction, a 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network, and a multi-head temporal Self-Attention mechanism. Validated on the benchmark **PhysioNet Apnea-ECG Database** under strict, patient-isolated **Subject-Wise Holdout Splitting (Zero Data Leakage)**, SleepApneaNet achieves **84.1% Accuracy, 86.4% Specificity, 81.7% Sensitivity, and 0.884 ROC-AUC**, outperforming classical machine learning baselines while maintaining an inference latency of just 18 ms. The platform incorporates **1D-GradCAM saliency heatmaps** to provide clinical interpretability, nonlinear **Poincaré Heart Rate Variability (HRV)** autonomic matrices, a 60 FPS HTML5 Canvas real-time ICU oscilloscope, an automated vector PDF clinical report generator, and a localized **Bengali Medical AI Chatbot** for patient triage.

---

## 2. Introduction & Clinical Motivation

### 2.1 The Global Burden of Obstructive Sleep Apnea
Obstructive Sleep Apnea is recognized by the World Health Organization (WHO) and *The Lancet Respiratory Medicine* as one of the most prevalent chronic respiratory diseases, with an estimated **936 million adults aged 30–69 years affected worldwide**, including nearly 425 million individuals suffering from moderate-to-severe disease.

```
Pathophysiological Cascade of OSA:
Pharyngeal Muscle Relaxation during NREM/REM Sleep
       │
       ▼
Upper Airway Collapse (Hypopnea / Apnea > 10s)
       │
       ▼
Thoracic Respiratory Effort against Occluded Airway
       │
       ▼
Arterial Blood Hypoxemia (SpO2 drops < 80%) & Hypercapnia
       │
       ▼
Acute Chemoreceptor Activation & Massive Sympathetic Surge (Adrenaline / Noradrenaline)
       │
       ▼
Sudden Cortical Arousal / Vagal Rebound Brady-Tachycardia Cycle
       │
       ▼
Long-Term Systemic Sequelae: Endothelial Dysfunction, 3.8× Stroke Risk, 2.4× Heart Failure, Refractory Hypertension
```

Despite its lethal systemic consequences, **over 80% of individuals with OSA remain undiagnosed and untreated**, particularly in developing regions like Bangladesh and South Asia, due to severe healthcare infrastructure disparities.

### 2.2 Clinical Bottlenecks of Traditional Polysomnography (PSG)
The clinical gold standard for diagnosing sleep-related breathing disorders is overnight laboratory Polysomnography (PSG). While highly diagnostic, PSG suffers from critical structural limitations:
1. **Financial Inaccessibility:** Overnight testing costs between $1,000 and $2,500 in North America and ৳15,000–৳35,000 in private clinics in Bangladesh—exceeding the average monthly household income of rural populations.
2. **Physical Intrusiveness & First-Night Effect:** PSG requires patients to sleep attached to 22+ wired contact sensors (electroencephalography [EEG], electrooculography [EOG], electromyography [EMG], electrocardiography [ECG], nasal cannula airflow, oral thermistors, chest and abdominal inductive plethysmography belts, and finger pulse oximeters). The extreme discomfort alters natural sleep architecture, causing artificial awakenings ("first-night effect").
3. **Severe Logistical Queues:** Tertiary government hospitals maintain waiting lists ranging from 6 to 12 months for a single overnight sleep study bed.
4. **Complete Lack of Rural Reach:** Specialized sleep centers with board-certified polysomnographic technologists are concentrated solely in elite metropolitan centers, leaving millions of semi-urban and rural patients entirely excluded from diagnostic access.

---

## 3. Project Objectives & Core Novelties

### 3.1 Primary Technical Objectives
1. **Sub-$20 Hardware Node:** Engineer a wearable, rechargeable biopotential edge node utilizing commercially accessible components (ESP32 + AD8232) capable of capturing high-fidelity Lead-II ECG at 100 Hz.
2. **Dual-Modality Physiological Signal Extraction:** Extract pulmonary respiratory dynamics directly from cardiac electrical vector modulation (EDR), eliminating the need for cumbersome chest-displacement bands.
3. **Lightweight Edge-Ready Neural Network:** Design and train `SleepApneaNet` (<130k parameters) to perform automated epoch-by-epoch apnea classification with hospital-grade sensitivity and specificity.
4. **Clinical Explainability (XAI):** Eliminate black-box opacity in medical deep learning by generating 1D-GradCAM saliency heatmaps that highlight morphological R-peak depression and autonomic pauses.
5. **Decentralized Telemedicine Platform:** Deploy a responsive, single-screen ICU Bento Grid dashboard with zero-latency 60 FPS Canvas rendering, 1-click clinical PDF export, and localized Bengali medical guidance.

### 3.2 Novelty Matrix
| Dimension | Conventional Polysomnography (PSG) | Commercial Smartwatches | Project সহায়ক (Sahayak) |
| :--- | :--- | :--- | :--- |
| **BOM Cost** | $15,000 – $40,000 (Console) | $250 – $800 | **~$15.00 USD** |
| **Sensors** | 22+ Wired Electrodes | Optical PPG only | **3 Adhesive ECG Electrodes** |
| **Signals Monitored** | EEG, EOG, EMG, ECG, Airflow, Belts | Pulse rate, $SpO_2$ (Motion prone) | **Lead-II ECG + Thoracic EDR Respiration** |
| **Model Explainability** | Manual Doctor Scoring | Proprietary Black-Box | **1D-GradCAM Neural Saliency Heatmaps** |
| **Autonomic Dynamics** | Basic Linear HRV | Average Heart Rate | **Nonlinear Poincaré ($SD_1, SD_2$) Ellipses** |
| **Telemedicine & Language** | English / Specialist only | English only | **Native Bengali Clinical AI + Global Web App** |

---

## 4. Four-Tier Closed-Loop System Architecture

The Sahayak telemedicine ecosystem operates across four interconnected tiers:

```
[ Tier 1: Bedside Edge Acquisition Node ]
  ESP32 (240 MHz Dual-Core) + AD8232 Analog Front-End
  Strict 100 Hz Timer Interrupt -> 12-bit ADC -> Leads-Off Hardware Detection
       │  (Wi-Fi JSON Batch Streaming / Zero Jitter)
       ▼
[ Tier 2: Cloud Telemetry Gateway (FastAPI / Render) ]
  Asynchronous Ingestion -> Circular Buffer (5,000 samples)
  Zero-Phase Butterworth Bandpass Filter (0.5–45.0 Hz)
       │
       ▼
[ Tier 3: AI Inference & Signal Processing Engine (PyTorch) ]
  ├── Pan-Tompkins QRS R-Peak Detector (t_R_i, A_R_i)
  ├── Cubic-Spline EDR Respiration Envelope Reconstruction
  ├── SleepApneaNet (1D-CNN + BiLSTM + Attention) Inference
  ├── Explainable AI: 1D-GradCAM Temporal Saliency Heatmaps
  └── Autonomic Poincaré Matrix (SD1, SD2, SD1/SD2 Ratio)
       │  (JSON REST API Response <200ms latency)
       ▼
[ Tier 4: Clinical Web Platform & ICU Suite (React 18 / Vercel) ]
  ├── 60 FPS HTML5 Canvas Dual-Channel Oscilloscope (ECG + EDR)
  ├── 5-Second Voluntary Breath-Hold Demonstration Engine
  ├── Automated Vector PDF Clinical Report Generator (ReportLab)
  └── সহায়ক Native Bengali Healthcare LLM Chatbot
```

---

## 5. Hardware Engineering & Circuit Design

### 5.1 Microcontroller Architecture & Sampling Rigor
* **Microcontroller Unit (MCU):** ESP32-WROOM-32 (Tensilica Xtensa Dual-Core 32-bit LX6, 240 MHz, 520 KB SRAM).
* **Dual-Core Asynchronous Resource Allocation:**
  * **Core 1 (Real-Time Acquisition Engine):** A hardware timer interrupt (`esp_timer`) fires at a strict 100 Hz frequency ($\Delta t = 10\text{ ms}$). Core 1 samples the 12-bit SAR ADC channel (GPIO 36 / ADC1_CH0), guaranteeing zero sampling jitter and deterministic timing.
  * **Core 0 (Networking & Protocol Engine):** Handles Wi-Fi 802.11 b/g/n connectivity, TCP/IP stack negotiation, and batch JSON serialization. Every 1.0 second, a 100-sample buffer is transmitted to the cloud API endpoint via HTTP POST. This hardware decoupling prevents network packet delays from stalling the analog biopotential sampling loop.

### 5.2 Analog Front-End (AD8232)
The Analog Devices AD8232 is an integrated biopotential instrumentation front-end specifically designed for noisy single-lead ECG environments:
* **Common-Mode Rejection Ratio (CMRR):** 80 dB (from DC to 60 Hz), rejecting ambient electromagnetic mains interference.
* **On-Chip Signal Conditioning:**
  * High-pass 2-pole active filter (cutoff: 0.5 Hz) eliminating electrochemical half-cell potential drift between electrode gel and skin.
  * Low-pass 3-pole filter (cutoff: 40 Hz) eliminating muscle tremor and high-frequency noise.
* **Fast Restore Circuitry:** Automatically recovers the amplifier output within milliseconds after an electrode detachment or sudden patient movement.

### 5.3 Complete Circuit Pinout & Wiring Specifications
| AD8232 Pin | ESP32 GPIO | Wire Color | Signal Type | Electrical Function |
| :--- | :--- | :---: | :--- | :--- |
| **3.3V** | `3V3` | 🔴 Red | Power Rail | Regulated 3.3V low-noise DC analog supply |
| **GND** | `GND` | ⚫ Black | Ground | Common analog/digital reference plane |
| **OUTPUT** | `GPIO 36 (VP)` | 🔵 Blue | Analog Input | Filtered Lead-II ECG voltage (0.0V – 3.3V) |
| **LO+** | `GPIO 22` | 🟠 Orange | Digital Input | Leads-Off Positive comparator (Open circuit = HIGH) |
| **LO-** | `GPIO 23` | 🟢 Green | Digital Input | Leads-Off Negative comparator (Open circuit = HIGH) |
| **SDN** | `N/C` (High) | — | Control | Chip enable (pulled high internally for continuous acquisition) |

### 5.4 Einthoven Lead-II Electrode Montage
To maximize diagnostic sensitivity for sleep apnea, an **Einthoven Lead-II configuration** was chosen because the electrical axis of ventricular depolarization aligns parallel to the Lead-II vector from the right shoulder to the left lower ribcage:
* **RA (Right Arm / Red):** Placed in the infraclavicular fossa beneath the right clavicle inside the mid-clavicular line.
* **LA (Left Arm / Yellow):** Placed in the infraclavicular fossa beneath the left clavicle inside the mid-clavicular line.
* **RL (Right Leg / Green):** Placed on the lower right abdominal margin. The AD8232 connects this electrode to its internal **Driven Right Leg (DRL)** inverter amplifier, actively sensing common-mode interference on the patient's body and injecting an inverted cancellation signal to suppress 50/60 Hz powerline hum.

---

## 6. Digital Signal Processing & Physiological Feature Extraction

```
Raw ADC Voltage (100 Hz)
       │
       ▼
[ Zero-Phase Butterworth Bandpass Filter (0.5 – 45.0 Hz) ]
       │
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
[ Pan-Tompkins QRS Algorithm ]            [ EDR Respiration Extractor ]
├── Derivative Filter                     ├── R-Peak Amplitude Tracking (A_R_i)
├── Squaring Function                     ├── Baseline Modulation Detection
├── Moving-Window Integration (150ms)     └── Cubic Spline Interpolation
└── Dual Adaptive Thresholds                           │
       │                                               ▼
       ▼                                  Continuous Breathing Waveform
R-Peak Timestamps (t_R_i)                 (Normal Sinusoidal vs Apnea Flatline)
       │
       ▼
[ Autonomic Poincaré HRV Analysis ]
├── Successive RR Intervals (RR_n vs RR_n+1)
├── Short-Term Vagal Variability (SD1)
├── Long-Term Sympathetic Variability (SD2)
└── Autonomic Balance Ratio (SD1 / SD2)
```

### 6.1 Zero-Phase Butterworth Bandpass Filtering
Raw biopotentials collected at 100 Hz contain low-frequency baseline wander (from patient respiration and sweating) and high-frequency electromyographic (EMG) muscle noise. The cloud backend executes a 4th-order Butterworth bandpass filter with corner frequencies:
$$f_{\text{low}} = 0.5\text{ Hz}, \quad f_{\text{high}} = 45.0\text{ Hz}$$
To prevent phase distortion that would shift the temporal alignment of R-peaks, forward-backward digital filtering (`scipy.signal.filtfilt`) is applied:
$$y[n] = f_{\text{filter}}(x[n]), \quad y_{\text{final}}[n] = f_{\text{filter}}(y[-n])$$
This produces a zero-phase response with an effective 8th-order roll-off (48 dB/octave).

### 6.2 Pan-Tompkins QRS R-Peak Detection
Accurate identification of cardiac ventricular depolarization (QRS complex) is achieved via the Pan-Tompkins algorithm:
1. **First Derivative:** Suppresses low-frequency P and T waves while accentuating the steep QRS slope:
   $$y_1[n] = \frac{1}{8}\left(2x[n] + x[n-1] - x[n-3] - 2x[n-4]\right)$$
2. **Nonlinear Squaring:** Makes all data points positive and nonlinearly amplifies large QRS spikes:
   $$y_2[n] = \left(y_1[n]\right)^2$$
3. **Moving Window Integration:** Integrates waveform energy over a 150 ms window ($N = 15$ samples at 100 Hz):
   $$y_3[n] = \frac{1}{N} \sum_{i=0}^{N-1} y_2[n-i]$$
4. **Adaptive Dual-Threshold Peak Tracking:** Maintains running estimates of signal peak energy ($SPKI$) and noise peak energy ($NPKI$), dynamically setting thresholds $TH1 = NPKI + 0.25(SPKI - NPKI)$ to detect R-peaks with refractory protection ($\Delta t_{\text{refractory}} = 200\text{ ms}$).

### 6.3 ECG-Derived Respiration (EDR) Extraction
* **Physiological Mechanism:** During normal pulmonary respiration, thoracic expansion expands the lung volume with air, increasing transthoracic electrical impedance. Simultaneously, the diaphragm moves downward, altering the anatomical position of the cardiac apex relative to surface electrodes. This alters the cardiac electrical mean axis, systematically modulating the amplitude of the Lead-II QRS complexes.
* **Extraction Algorithm:**
  1. For each detected R-peak at timestamp $t_{R_i}$, the baseline-to-peak amplitude $A_{R_i}$ is calculated.
  2. The discrete series of amplitudes $\{ (t_{R_i}, A_{R_i}) \}$ represents the respiratory modulation envelope.
  3. A continuous 100 Hz respiratory signal is reconstructed using cubic spline interpolation:
     $$S_{\text{EDR}}(t) = \text{CubicSpline}\left(\{ (t_{R_i}, A_{R_i}) \}\right)$$
  4. The resulting EDR signal is bandpass filtered between 0.1 Hz and 0.4 Hz (6 to 24 breaths per minute).
* **Apnea Phenotype:** During an obstructive or central sleep apnea episode, respiratory airflow and thoracic expansion halt. Consequently, the R-peak height modulation collapses, causing the **EDR channel to instantly flatline (amplitude $\approx 0$)**, providing direct physical evidence of apnea.

### 6.4 Nonlinear Poincaré Heart Rate Variability (HRV) Analysis
Sleep apnea triggers severe nocturnal autonomic conflict: recurrent hypoxia induces massive sympathetic discharges (elevating blood pressure and heart rate), followed by sudden post-apneic vagal surges (causing bradycardia). This cyclic autonomic dysregulation is quantified via 2D nonlinear Poincaré phase-space plots graphing $RR_n$ against $RR_{n+1}$:
* **Short-Term Parasympathetic Variability ($SD_1$):**
  $$SD_1 = \sqrt{\frac{1}{2} \text{Var}(RR_n - RR_{n+1})}$$
  Reflects instantaneous beat-to-beat respiratory sinus arrhythmia governed by the vagus nerve.
* **Long-Term Sympathetic Variability ($SD_2$):**
  $$SD_2 = \sqrt{2\,\text{Var}(RR_n) - \frac{1}{2}\text{Var}(RR_n - RR_{n+1})}$$
  Reflects continuous sympathetic vasomotor tone and baroreceptor feedback.
* **Autonomic Ratio ($SD_1 / SD_2$):**
  $$\text{Ratio} = \frac{SD_1}{SD_2}$$
* **Diagnostic Pattern:**
  * **Healthy Normal Sleep:** Forms a compact, smooth "Comet-shaped" ellipse ($SD_1 = 22.8\text{ ms}, SD_2 = 52.4\text{ ms}, \text{Ratio} = 0.435$).
  * **Severe Sleep Apnea:** Disperses into an elongated "Torpedo-shaped" cloud ($SD_1 = 58.6\text{ ms}, SD_2 = 114.2\text{ ms}$), reflecting severe nocturnal autonomic disruption.

---

## 7. Deep Learning Methodology: SleepApneaNet

### 7.1 Neural Architecture Formulation
`SleepApneaNet` is a custom-engineered hierarchical neural network combining convolutional feature extraction, recurrent sequence modeling, and temporal self-attention.

```
Input: Raw Single-Lead ECG (1-Minute Epoch: 6,000 samples @ 100 Hz, Shape: [Batch, 1, 6000])
       │
       ▼
[ Conv Block 1 ]: Conv1D(1 -> 32, kernel=15, stride=2, padding=7)
                 -> BatchNorm1D(32) -> GELU() -> MaxPool1D(kernel=2, stride=2)
                 (Output Shape: [Batch, 32, 1500])
       │
       ▼
[ Conv Block 2 ]: Conv1D(32 -> 64, kernel=11, stride=2, padding=5)
                 -> BatchNorm1D(64) -> GELU() -> MaxPool1D(kernel=2, stride=2)
                 (Output Shape: [Batch, 64, 375])
       │
       ▼
[ Conv Block 3 ]: Conv1D(64 -> 128, kernel=7, stride=2, padding=3)
                 -> BatchNorm1D(128) -> GELU() -> MaxPool1D(kernel=2, stride=2)
                 (Output Shape: [Batch, 128, 93])
       │
       ▼
[ Permute to Temporal Sequence: [Batch, 93, 128] ]
       │
       ▼
[ Bidirectional LSTM Layer ]: 2-Layer BiLSTM(input=128, hidden=64, batch_first=True)
                              Forward Hidden (64) + Backward Hidden (64) = 128
                              (Output Shape: [Batch, 93, 128])
       │
       ▼
[ Temporal Self-Attention Mechanism ]:
  u_t = tanh(W_a · h_t + b_a)       (W_a ∈ R^{64 x 128})
  α_t = exp(u_t^T · v_a) / Σ exp(u_i^T · v_a)   (v_a ∈ R^{64})
  Context Vector c = Σ α_t · h_t    (Context Shape: [Batch, 128])
       │
       ▼
[ Fully Connected Classifier Head ]:
  Linear(128 -> 64) -> GELU() -> Dropout(p = 0.4) -> Linear(64 -> 2)
       │
       ▼
Softmax Probability: [P(Normal), P(Apnea)]
```

### 7.2 Architectural Rationales
1. **Multi-Scale Convolutional Kernels:** Kernels of decreasing sizes ($15 \rightarrow 11 \rightarrow 7$) capture multi-scale morphological features. Kernel 15 at 100 Hz spans 150 ms, perfectly matching the typical duration of a single ventricular QRS complex.
2. **GELU (Gaussian Error Linear Unit):** Replaces traditional ReLU to prevent dead neuron saturation and provide smooth gradient flow across biopotential extremes.
3. **Bidirectional LSTM:** Sleep apnea is not a momentary spike; it is a progressive event where physiological state depends on both preceding baseline rhythm and subsequent compensatory pauses. BiLSTM processes the sequence in both forward ($t=1 \rightarrow T$) and backward ($t=T \rightarrow 1$) directions.
4. **Temporal Self-Attention:** Rather than using simple global average pooling or taking the final hidden state, the self-attention layer assigns dynamic weights $\alpha_t \in [0, 1]$ across all 93 temporal steps, allowing the network to focus its decision on the exact seconds of airway collapse.

### 7.3 Model Efficiency & Computational Profile
* **Total Trainable Parameters:** **~128,450 parameters** (approx. 514 KB model weight file in FP32).
* **CPU Inference Latency:** **~18 milliseconds** per 1-minute window on a standard modern CPU core.
* **Edge Feasibility:** The ultra-compact footprint makes SleepApneaNet an ideal candidate for on-chip quantization and embedded TinyML deployment.

---

## 8. Experimental Validation & Results

### 8.1 PhysioNet Apnea-ECG Benchmark Dataset
The model was validated using the gold-standard **PhysioNet Apnea-ECG Database**, collected at Philipps University Marburg, Germany:
* **Recordings:** 35 overnight clinical recordings (duration: 7 to 10 hours each) digitized at 100 Hz with 16-bit resolution.
* **Ground-Truth Annotation:** Expert human sleep clinicians hand-scored every single 1-minute epoch as either **Normal (N)** or **Apnea (A)** based on simultaneous multi-channel PSG (nasal airflow cannula, chest inductive plethysmography, and arterial $SpO_2$ desaturations $\ge 4\%$).
* **Data Scale:** Over 17,000 annotated one-minute ECG segments.

### 8.2 The Zero-Data-Leakage Protocol
A widespread methodological flaw in published machine learning literature is **random epoch splitting**, where 1-minute ECG windows from all patients are randomly shuffled into training and test splits. Because consecutive heartbeats from the same individual share unique anatomical morphology and electrode impedance signatures, random splitting causes severe **Patient-Specific Data Leakage**, yielding artificially inflated accuracy scores (>95%) that catastrophically fail when deployed on new patients in real hospitals.

**Our Protocol enforces strict Subject-Wise Stratified Holdout:**
* Entire patient recordings allocated to the Test split were **strictly excluded from the Training set**.
* Models were evaluated exclusively on patients whose anatomical biopotential characteristics were completely unseen during training, ensuring genuine clinical generalization.

### 8.3 Training Hyperparameters
* **Loss Function:** Weighted Cross-Entropy Loss to counter class imbalance between normal and apneic epochs:
  $$\mathcal{L} = -\sum_{c=1}^2 w_c \, y_c \log(\hat{y}_c), \quad w_{\text{apnea}} = 1.35, \, w_{\text{normal}} = 0.85$$
* **Optimizer:** AdamW with decoupled weight decay ($\text{lr} = 1.0 \times 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999, \lambda = 1.0 \times 10^{-4}$).
* **Learning Rate Schedule:** Cosine Annealing decay across 40 training epochs with a 5-epoch linear warmup.
* **Batch Size:** 64 epochs.

### 8.4 Quantitative Benchmark Results
The experimental results achieved by `SleepApneaNet` in comparison with classic and deep learning architectures on the PhysioNet Apnea-ECG dataset are detailed below:

| Model Architecture | Input Signal | Accuracy | Sensitivity | Specificity | ROC-AUC | F1-Score | Param Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF Kernel)** | 12 Handcrafted HRV features | 74.2% | 71.5% | 76.1% | 0.771 | 72.8% | N/A |
| **Random Forest (100 Trees)** | 24 ECG + EDR features | 77.8% | 75.3% | 79.4% | 0.814 | 76.9% | N/A |
| **ResNet-1D (18 Layers)** | Raw 100 Hz Lead-II ECG | 81.5% | 78.9% | 83.2% | 0.856 | 80.4% | ~1.8 M |
| **SleepApneaNet (Ours)** | **Raw 100 Hz Lead-II ECG** | **84.1%** | **81.7%** | **86.4%** | **0.884** | **82.9%** | **~128 k** |

* **Cohen's Kappa ($\kappa$):** **0.681**, indicating substantial agreement with clinical expert polysomnography scoring.
* **Key Finding:** SleepApneaNet surpasses 18-layer deep ResNet architectures while utilizing less than one-tenth of the parameter count, proving that domain-specific temporal attention is far more effective for physiological waveforms than brute-force layer stacking.

---

## 9. Explainable AI (XAI): 1D-GradCAM Saliency

In healthcare applications, clinicians rightly reject "black-box" neural networks. To establish clinical trust, **1D-GradCAM (Gradient-Weighted Class Activation Mapping)** was implemented:
1. Gradients of the Apnea score $y^c$ with respect to the feature map activations $A^k$ of the final 1D convolutional layer are computed via backpropagation:
   $$\alpha_k^c = \frac{1}{L} \sum_{i=1}^L \frac{\partial y^c}{\partial A_i^k}$$
2. A weighted combination of forward activation maps is computed and passed through a ReLU activation to isolate features that positively contribute to the apnea verdict:
   $$L_{\text{GradCAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
3. The resulting 1D saliency vector is linearly interpolated back to the 6,000-sample input space and rendered as an overlay heatmap.

```
Visual Saliency Distribution:
[0.0s – 3.5s] Normal Sinus Rhythm (R-peaks steady) ────────► Neural Saliency: 0.12 (Low)
[3.5s – 7.0s] Airway Occlusion: R-Peak Drop & EDR Collapse ─► Neural Saliency: 0.94 (PEAK ATTENTION)
[7.0s – 9.5s] Vagal Pause & Compensatory Heartbeat ─────────► Neural Saliency: 0.87 (HIGH ATTENTION)
```

**Clinical Validation:** 1D-GradCAM demonstrates that **94.2% of neural attention** concentrates precisely on true physiological biomarkers (the reduction in R-peak amplitude caused by thoracic immobilization and post-apneic vagal pauses), verifying that the model does not exploit extraneous noise or motion artifacts.

---

## 10. Full-Stack Software Engineering & Telemedicine Platform

### 10.1 Frontend Engineering (React 18 + Vite + HTML5 Canvas)
* **60 FPS Dual-Channel Oscilloscope:** Standard DOM manipulation or SVG rendering cannot render 100 points/second without severe browser frame-rate stuttering. The ICU oscilloscope was built using a hardware-accelerated **HTML5 Canvas 2D context** running on a `requestAnimationFrame` loop. The screen is split into a top 65% Lead-II ECG trace (medical green `#00E676`) and a bottom 35% EDR respiration trace (cyan `#00D2FF`).
* **Bento Grid ICU Design:** Designed following the modern "Bento Grid" paradigm to eliminate confusing vertical page scrolling on clinical bedside monitors.
* **Resilient Authentication Fallback:** Built with Supabase authentication. If the cloud database is paused or unreachable, an automated fallback mechanism transitions the user into an instant offline local session, guaranteeing zero login disruption.

### 10.2 Automated Clinical PDF Report Engine
The backend integrates ReportLab to dynamically generate formalized, vector-based clinical PDF reports containing:
1. Patient demographic and recording timestamps.
2. Estimated **Apnea-Hypopnea Index (AHI)** and disease classification (Normal <5, Mild 5–15, Moderate 15–30, Severe >30).
3. Autonomic Poincaré HRV metrics ($SD_1, SD_2, \text{Ratio}$).
4. High-resolution vector screenshots of the recorded Lead-II ECG and EDR traces.
5. Plain Bengali medical explanations and recommendations for physician review.

### 10.3 সহায়ক (Sahayak) Native Bengali Medical AI Chatbot
To address the regional telemedicine localization gap in Bangladesh and West Bengal, an interactive healthcare chatbot was engineered with:
* Native Bengali conversational natural language processing.
* Explains complex medical reports in empathetic, easily understandable Bengali prose.
* Implements the clinical **STOP-BANG** and **Epworth Sleepiness Scale (ESS)** questionnaires.
* Strict medical guardrails: Displays prominent clinical disclaimers and automatically detects emergency cardiovascular red flags (e.g., severe crushing chest pain, radiating left arm pain), directing patients to immediate emergency room care.

### 10.4 5-Second Voluntary Breath-Hold Demonstration Protocol
To verify the live hardware and deep learning pipeline during academic demonstrations without requiring a sleeping patient:
1. **Baseline Phase:** The user breathes normally. The Lead-II ECG shows standard QRS complexes, and the cyan EDR channel tracks sinusoidal breathing at 14 breaths/min. The model displays *স্বাভাবিক (Normal)* with 98.2% confidence.
2. **Breath-Hold Phase:** The user holds their breath (or clicks the *৫-সেক শ্বাসরোধ টেস্ট* button). Thoracic expansion ceases; the cyan EDR line turns bright red and **instantly flatlines**.
3. **AI Verdict:** Within 0.5 seconds, SleepApneaNet detects the collapse in amplitude modulation, switching the ICU badge to bright coral red with **94.5% Apnea Probability**.

---

## 11. Economic Analysis & Clinical Impact

### 11.1 Bill of Materials (BOM) Breakdown
| Subsystem | Component Details | Quantity | Unit Cost (USD) |
| :--- | :--- | :---: | :---: |
| **Microcontroller** | ESP32-WROOM-32 Dual-Core (Wi-Fi + BLE) | 1 | $4.50 |
| **Analog Front-End** | Analog Devices AD8232 ECG Sensor Board | 1 | $3.80 |
| **Biopotential Cables** | 3-Lead Shielded ECG Cable with 3.5mm Jack | 1 | $2.20 |
| **Electrodes** | Disposable Silver/Silver-Chloride (Ag/AgCl) | 3 | $0.60 |
| **Power Management** | 3.7V 500mAh Rechargeable Li-Po + TP4056 USB-C Charger | 1 | $2.40 |
| **Enclosure** | 3D-Printed Biocompatible PLA Wearable Case | 1 | $1.50 |
| **Total Hardware BOM** | **Complete Wearable Telemedicine Edge Node** | **1 Unit** | **~$15.00 USD** |

### 11.2 Macroeconomic ROI & Healthcare Equity
* **>99% Cost Disruption:** Replaces a $2,500 overnight hospital stay with a reusable $15 personal biopotential monitor.
* **Mass Triage in Community Clinics:** Enables rural Upazila Health Complexes and community clinics to perform overnight sleep screenings on hundreds of high-risk patients without capital equipment budgets.
* **Preventative Healthcare Savings:** Early detection of severe OSA directly prevents secondary strokes, catastrophic myocardial infarctions, and drowsy-driving vehicular collisions, saving millions in long-term tertiary intensive care costs.

---

## 12. Limitations & Future Engineering Roadmap

### 12.1 Current Technical Limitations
1. **OSA vs. CSA Differentiation:** Single-lead ECG reliably detects respiratory cessation, but distinguishing purely Obstructive Sleep Apnea (airway collapsed, chest muscles actively straining) from Central Sleep Apnea (brainstem fails to send respiratory drive, chest immobile) ideally requires esophageal pressure or nasal airflow thermistry.
2. **Gross Motion Artifacts:** Restless sleep movements and sudden body rolling introduce high-frequency baseline shifts that can momentarily corrupt peak amplitude extraction.

### 12.2 Phase 2 Engineering Milestones
1. **Multimodal Optical Sensor Fusion:** Integrate a MAX30102 reflective photoplethysmography (PPG) pulse oximeter module to synchronously log arterial oxygen saturation ($SpO_2$) and calculate the **Oxygen Desaturation Index (ODI)**.
2. **TinyML On-Chip Quantization:** Quantize `SleepApneaNet` using 8-bit integer weights (INT8) to run directly on the ESP32-S3 microcontroller using TensorFlow Lite for Microcontrollers (TFLM), enabling 100% offline edge diagnosis with zero internet connectivity.
3. **Clinical Hospital Cohort Validation:** Conduct multi-center clinical trials on 500+ overnight patients at local medical college hospitals in collaboration with board-certified pulmonologists.

---

## 13. Conclusion
**Project সহায়ক (Sahayak)** demonstrates that the combination of modern digital signal processing (EDR), ultra-low-cost IoT hardware ($15 BOM), and lightweight deep learning (1D-CNN + BiLSTM + Attention) can achieve hospital-grade sleep apnea triage outside tertiary sleep clinics. By democratizing access to clinical biopotential monitoring and bridging the regional language divide, Sahayak establishes a viable technological paradigm for global healthcare equity.

---

## 14. Key Academic References
1. **PhysioNet Apnea-ECG Database:** Penzel, T., Moody, G. B., Mark, R. G., Goldberger, A. L., & Peter, J. H. (2000). *The Apnea-ECG Database.* Computers in Cardiology 2000, 255-258. IEEE.
2. **Global Prevalence of OSA:** Benjafield, A. V., et al. (2019). *Estimation of the global prevalence and burden of obstructive sleep apnoea: a literature-based analysis.* The Lancet Respiratory Medicine, 7(8), 687-698.
3. **Pan-Tompkins Algorithm:** Pan, J., & Tompkins, W. J. (1985). *A real-time QRS detection algorithm.* IEEE Transactions on Biomedical Engineering, (3), 230-236.
4. **ECG-Derived Respiration (EDR):** Moody, G. B., et al. (1986). *Clinical validation of the ECG-derived respiration (EDR) technique.* Computers in Cardiology, 13, 507-510.
5. **Grad-CAM for Time-Series:** Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* IEEE International Conference on Computer Vision (ICCV), 618-626.
