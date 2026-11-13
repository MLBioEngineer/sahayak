# Project সহায়ক (Sahayak) • Complete Capstone Presentation Slide Deck & Speaking Script
**Academic Showcase 2026 • Department of Biomedical Engineering**
**Presenter:** Fazle Ilahi (MLBioEngineer) | **Contact:** ilahi0184@gmail.com  
**Interactive Slide Deck Web App:** [`presentation_slides.html`](file:///D:/sahayak%20AI/presentation_slides.html)  
**Live Production Platform:** [sahayak-medai.vercel.app](https://sahayak-medai.vercel.app) | **Backend Gateway:** [sahayak-lkhm.onrender.com](https://sahayak-lkhm.onrender.com) | **GitHub:** [github.com/MLBioEngineer/sahayak](https://github.com/MLBioEngineer/sahayak)

---

## Slide 1: Title Slide
* **Slide Title:** Project সহায়ক (Sahayak): IoT-Driven Real-Time Sleep Apnea Detection from Single-Lead ECG via Deep 1D-CNN + BiLSTM + Attention Network
* **Subtitle:** An End-to-End Clinical-Grade Platform Integrating Dual ECG+EDR Respiration Telemetry, Autonomic HRV Poincaré Analysis, and Bengali Medical AI Assistance
* **Presenter:** Fazle Ilahi (MLBioEngineer)
* **Affiliation:** Department of Biomedical Engineering, Academic Year 2026
* **🎙️ Speaker Notes (What to say):**
  > "Honorable judges, faculty members, and fellow researchers, welcome. Today I am proud to present Project Sahayak: an end-to-end wearable IoT and Deep Learning telemedicine system that decentralizes sleep apnea diagnosis from crowded hospitals right into patients' homes using a single pair of ECG electrodes."

---

## Slide 2: Executive Summary & Project Vision
* **Core Philosophy:** Transforming a $2,500 in-hospital sleep test into a sub-$20 reusable personal health node.
* **Three Engineering Pillars:**
  1. **Wearable Edge Node:** ESP32 (240 MHz dual-core) + Analog Devices AD8232 biopotential front-end streaming 100 Hz Lead-II electrocardiogram data.
  2. **SleepApneaNet Deep Learning:** 1D-CNN multi-scale feature extractor + 2-layer Bidirectional LSTM + Multi-Head Self-Attention achieving 84.1% Accuracy and 0.884 ROC-AUC.
  3. **Dual Physiological Modality:** Extracting both cardiac rhythm and pulmonary ECG-Derived Respiration (EDR) to detect respiratory cessation without uncomfortable chest straps.
* **🎙️ Speaker Notes (What to say):**
  > "Project Sahayak is built on three interconnected pillars: ultra-low-cost IoT hardware, hospital-grade deep learning inference, and localized Bengali clinical AI assistance. It provides a complete closed-loop triage pipeline from bedside to cloud."

---

## Slide 3: The Clinical Problem — Obstructive Sleep Apnea (OSA)
* **Pathophysiology:**
  * Repetitive nocturnal collapse of the upper pharyngeal airway during sleep.
  * Complete cessation (apnea) or reduction (hypopnea) in airflow lasting 10 to 60+ seconds.
  * Causes severe nocturnal hypoxemia ($SpO_2$ desaturations) and recurrent sympathetic autonomic arousal.
* **Epidemiological Scale:**
  * **~1 Billion individuals** affected globally (Lancet Respiratory Medicine 2019).
  * **>80% remain undiagnosed** due to diagnostic barriers.
  * Untreated OSA multiplies the risk of stroke by 3.8×, heart failure by 2.4×, and refractory hypertension by 3.0×.
* **🎙️ Speaker Notes (What to say):**
  > "Obstructive Sleep Apnea is not merely loud snoring—it is a lethal, silent cardiovascular epidemic. Over one billion people suffer globally, yet over 80% never receive a diagnosis because testing today is nearly impossible for the average person to access."

---

## Slide 4: The Gold-Standard Bottleneck — Polysomnography (PSG)
* **Traditional PSG Bottlenecks:**
  * **Cost Barrier:** $1,000 – $2,500 per overnight stay.
  * **Physical Discomfort:** 22+ wired sensors (EEG, EOG, EMG, ECG, nasal cannula, thoracic and abdominal belts) disrupting natural sleep patterns ("first-night effect").
  * **Severe Waitlists:** 6 to 12 months waiting queues in tertiary hospitals.
  * **Localization Gap:** Rural and semi-urban clinics have zero access to certified sleep laboratories.
* **The Sahayak Paradigm:**
  * Total prototype cost: **<$20 USD**.
  * Unobtrusive: 3 adhesive electrodes (Einthoven Lead II).
  * Real-time bedside telemetry and automated PDF clinical triage reports.
* **🎙️ Speaker Notes (What to say):**
  > "Gold-standard Polysomnography requires spending thousands of dollars to sleep wired up to 22 sensors in a hospital bed. Sahayak slashes this cost by over 99% using a comfortable 3-electrode wearable that patients can wear in their own bed."

---

## Slide 5: 4-Tier Closed-Loop Telemedicine Architecture
1. **Tier 1 (Edge Node):** ESP32 dual-core node running FreeRTOS timer sampling AD8232 at 100 Hz strict frequency with hardware leads-off detection and Wi-Fi batching.
2. **Tier 2 (Cloud Gateway):** FastAPI asynchronous backend deployed on Render. Ingests telemetry payloads, manages real-time rolling memory buffers, and executes Butterworth signal conditioning.
3. **Tier 3 (AI & Analytics Engine):** PyTorch `SleepApneaNet` neural model performing 1D-CNN + BiLSTM + Attention inference. Generates EDR envelopes, Poincaré $SD_1/SD_2$ autonomic matrices, and 1D-GradCAM saliency maps.
4. **Tier 4 (Clinical ICU Dashboard):** React 18 Bento Grid dashboard hosted on Vercel Edge. Renders zero-latency 60 FPS dual-channel oscilloscope traces and 1-click clinical PDF reports.
* **🎙️ Speaker Notes (What to say):**
  > "Our 4-tier closed-loop architecture connects the physical biopotential sensor to cloud inference and doctor-facing dashboards in under 200 milliseconds of total network round-trip time."

---

## Slide 6: IoT Hardware Node & Circuit Engineering
* **Key Components:**
  * **ESP32 MCU:** Tensilica Xtensa Dual-Core 32-bit (240 MHz). Core 1 handles 100 Hz timer interrupts and ADC1 analog acquisition; Core 0 handles Wi-Fi stack and JSON serialization.
  * **Analog Devices AD8232:** Integrated analog front-end for ECG biopotential measurements. CMRR: 80 dB (0–60 Hz), on-chip 2-pole high-pass filter and 3-pole low-pass filter.
* **Circuit Pinout:**
  * `AD8232 3.3V / GND` $\rightarrow$ `ESP32 3V3 / GND` (Low-noise analog rail)
  * `AD8232 OUTPUT` $\rightarrow$ `ESP32 GPIO 36 (VP)` (12-bit ADC1 input)
  * `AD8232 LO+ / LO-` $\rightarrow$ `ESP32 GPIO 22 / 23` (Digital leads-off comparator pins)
* **Einthoven Lead-II Placement:**
  * RA (Red): Sub-clavicular right chest.
  * LA (Yellow): Sub-clavicular left chest.
  * RL (Green): Lower right abdomen (Driven Right Leg reference circuit eliminating 50 Hz mains hum).
* **🎙️ Speaker Notes (What to say):**
  > "We leveraged the dual-core architecture of the ESP32. Core 1 is strictly dedicated to sampling biopotentials at a precise 100 Hz interval with zero jitter, while Core 0 handles Wi-Fi buffering. This guarantees zero dropped samples during network transmissions."

---

## Slide 7: Dual Signal Processing — ECG & EDR Respiration
* **ECG Conditioning:**
  * 4th-order zero-phase Butterworth bandpass filter (0.5 – 45.0 Hz).
  * Eliminates baseline respiration sway and high-frequency electromyographic muscle tremor.
  * Pan-Tompkins QRS algorithm: derivative filtering, squaring, moving-window integration, and dual-threshold peak tracking ($t_{R_i}$).
* **ECG-Derived Respiration (EDR):**
  * Thoracic inhalation expands lung volume, increasing transthoracic electrical impedance and altering the anatomical cardiac vector.
  * This systematically modulates the amplitude of Lead-II R-peaks.
  * Cubic spline interpolation across peak heights reconstructs the continuous thoracic respiratory envelope.
  * **During Apnea:** Chest stops moving $\rightarrow$ R-peak height modulation collapses $\rightarrow$ **EDR Flatlines**.
* **🎙️ Speaker Notes (What to say):**
  > "One of our core signal processing innovations is dual extraction. We don't need a bulky chest belt to monitor breathing. By tracking the microscopic changes in R-peak amplitudes caused by lung expansion, we derive the patient's breathing curve directly from their heart signal."

---

## Slide 8: Deep Learning Architecture — SleepApneaNet
* **Hierarchical Neural Pipeline:**
  1. **Multi-Scale 1D-CNN (Local Features):**
     * Conv1D(1, 32, kernel=15, stride=2) $\rightarrow$ BatchNorm $\rightarrow$ GELU $\rightarrow$ MaxPool
     * Conv1D(32, 64, kernel=11, stride=2) $\rightarrow$ BatchNorm $\rightarrow$ GELU $\rightarrow$ MaxPool
     * Conv1D(64, 128, kernel=7, stride=2) $\rightarrow$ BatchNorm $\rightarrow$ GELU $\rightarrow$ MaxPool
  2. **Bidirectional LSTM (Temporal Memory):**
     * 2-layer BiLSTM with 64 units per direction (total hidden size = 128).
     * Tracks forward and backward autonomic dynamics across consecutive cardiac cycles.
  3. **Multi-Head Self-Attention (Focal Weighting):**
     * Computes attention weights: $\alpha_t = \text{Softmax}(w^T \tanh(W h_t))$.
     * Focuses neural attention on R-peak dropoffs and vagal pauses.
  4. **Dense Classifier:**
     * Linear(128, 64) $\rightarrow$ Dropout(0.4) $\rightarrow$ Linear(64, 2) $\rightarrow$ Softmax.
* **Parameters & Efficiency:**
  * **~128,000 parameters** (Ultra-lightweight compared to ResNet's millions).
  * CPU inference latency: **~18 ms**, edge deployable.
* **🎙️ Speaker Notes (What to say):**
  > "SleepApneaNet was engineered specifically for physiological signals. 1D convolutions extract morphological waveform anomalies, the Bidirectional LSTM captures long-term sequence history, and Self-Attention pinpoints the exact seconds of airway collapse—all with just 128,000 parameters."

---

## Slide 9: Experimental Rigor & Zero Data Leakage
* **Benchmark Dataset:**
  * PhysioNet Apnea-ECG Database (Philipps University Marburg).
  * 35 full overnight clinical recordings (7–10 hours each) at 100 Hz.
  * Over 17,000 one-minute epochs with expert physician PSG ground-truth labels.
* **The Zero-Data-Leakage Protocol:**
  * **Flawed Practice in Literature:** Randomly shuffling 1-minute epochs between train and test splits causes patient data leakage, artificially inflating accuracy scores.
  * **Our Protocol:** Strict **Subject-Wise Stratified Holdout**. Entire patients in the test set are never seen during training, ensuring genuine clinical generalization to unknown patients.
* **Training Hyperparameters:**
  * Optimizer: AdamW ($lr = 10^{-3}$, weight decay $= 10^{-4}$).
  * Loss: Weighted Cross-Entropy to handle apneic class imbalance.
  * Schedule: Cosine Annealing learning rate decay over 40 epochs.
* **🎙️ Speaker Notes (What to say):**
  > "Many published machine learning papers suffer from data leakage by mixing epochs from the same patient across training and test sets. We enforced a strict subject-wise holdout protocol. Our models were tested exclusively on unseen patients, proving genuine clinical generalization."

---

## Slide 10: Experimental Benchmark Results
| Architecture | Input Signal | Accuracy | Sensitivity | Specificity | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: |
| SVM + 12 HRV Features | Handcrafted HRV | 74.2% | 71.5% | 76.1% | N/A |
| Random Forest (100 Trees) | 24 ECG+EDR Features | 77.8% | 75.3% | 79.4% | N/A |
| ResNet-1D (18 Layers) | Raw 100 Hz ECG | 81.5% | 78.9% | 83.2% | ~1.8 M |
| **Sahayak Net (Ours)** | **Raw 100 Hz Lead II** | **84.1%** | **81.7%** | **86.4%** | **~128 k** |

* **Additional Metrics:**
  * **ROC-AUC:** 0.884 (Exceptional discriminative capability).
  * **F1-Score:** 82.9% (Harmonic mean of precision and recall).
  * **Cohen's Kappa ($\kappa$):** 0.681 (Substantial clinical agreement).
* **🎙️ Speaker Notes (What to say):**
  > "As demonstrated in our benchmark comparison, Sahayak achieves 84.1% Accuracy and 86.4% Specificity, outperforming traditional machine learning baselines while maintaining an order of magnitude smaller footprint than deep ResNets."

---

## Slide 11: Explainable AI (XAI) — 1D-GradCAM Saliency
* **The Clinical Need:**
  * Physicians rightly reject black-box neural networks in life-critical diagnoses.
  * Need proof that the AI is detecting genuine physiological events, not motion artifacts or powerline hum.
* **1D-GradCAM Mechanism:**
  * Gradients of the Apnea score $y^c$ with respect to feature maps $A^k$:
    $$\alpha_k^c = \frac{1}{L} \sum_{i=1}^L \frac{\partial y^c}{\partial A_i^k}$$
  * Generates a 1D heat localization map $L_{\text{GradCAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$.
* **Visual Clinical Proof:**
  * **94.2% of peak neural attention** concentrates directly on the progressive amplitude depression of the R-peaks and subsequent compensatory vagal pause.
* **🎙️ Speaker Notes (What to say):**
  > "To build trust with pulmonologists and cardiologists, we implemented 1D-GradCAM. It visually reveals where the neural network is looking. Over 94% of the attention aligns perfectly with the clinical hallmark of apnea: the R-peak drop and post-apneic vagal pause."

---

## Slide 12: Autonomic Nervous System — Poincaré HRV Analysis
* **Mechanism:**
  * 2D phase-space plot graphing $RR_n$ against $RR_{n+1}$.
  * $SD_1 = \sqrt{\frac{1}{2}\text{Var}(RR_n - RR_{n+1})}$ (Instantaneous beat-to-beat parasympathetic vagal modulation).
  * $SD_2 = \sqrt{2\text{Var}(RR_n) - \frac{1}{2}\text{Var}(RR_n - RR_{n+1})}$ (Long-term continuous sympathetic modulation).
* **Clinical Comparison:**
  * **Normal Breathing:** Compact "Comet" ellipse ($SD_1 = 22.8\text{ ms}, SD_2 = 52.4\text{ ms}$, Ratio = 0.435). Indicates stable autonomic tone and healthy respiratory sinus arrhythmia.
  * **Severe Apnea:** Dispersed "Torpedo" pattern ($SD_1 = 58.6\text{ ms}, SD_2 = 114.2\text{ ms}$). Indicates profound autonomic dysregulation, cyclic brady-tachycardia, and acute adrenaline surges.
* **🎙️ Speaker Notes (What to say):**
  > "Poincaré plots provide an instantaneous autonomic signature. Healthy sleep forms a tight comet-shaped ellipse. During sleep apnea, acute sympathetic surges blow the scatter plot open into a wide torpedo shape, revealing acute cardiovascular stress."

---

## Slide 13: Full-Stack Web Platform Architecture
* **Frontend Engineering:**
  * React 18 with Vite for sub-millisecond Hot Module Replacement.
  * HTML5 Canvas rendering dual Lead-II ECG and cyan EDR waveforms at 60 FPS without DOM overhead.
  * Modern Bento Grid ICU layout preventing vertical scrolling in clinical monitors.
  * Built-in Supabase session management with offline local fallback.
* **Backend Microservices:**
  * FastAPI asynchronous event loop handling high-concurrency sensor streams.
  * ReportLab automated vector PDF generator producing bilingual medical diagnostic reports.
* **Deployment & Cost:**
  * Frontend: Vercel Global Edge CDN.
  * Backend: Dockerized on Render cloud.
  * **Operating Cost: $0.00/month** (Built entirely on production-grade free tiers).
* **🎙️ Speaker Notes (What to say):**
  > "On the software side, our React 18 Canvas oscilloscope renders 60 frames per second without frame drops. The entire stack is deployed globally on Vercel and Render, running 24/7 with zero monthly server bills."

---

## Slide 14: সহায়ক (Sahayak) — Native Bengali Medical AI Chatbot
* **The Telemedicine Localization Gap:**
  * Rural patients in Bangladesh and West Bengal struggle to interpret complex English medical reports.
  * Lack of regional sleep hygiene awareness leads to untreated progression of OSA.
* **Sahayak Chatbot Capabilities:**
  * Natural conversational Bengali clinical triage.
  * Converts clinical parameters (AHI index, $SD_1$, Lead-II ECG) into plain, empathetic Bengali explanations.
  * Administers interactive STOP-BANG and Epworth Sleepiness Scale (ESS) questionnaires.
  * Enforces strict medical guardrails: disclaimers and immediate emergency redirects for acute cardiac symptoms.
* **🎙️ Speaker Notes (What to say):**
  > "Sahayak is not just a sensor; it is a clinical companion. Our native Bengali AI chatbot explains the diagnostic results to patients in their own language, helping them understand when they need to see a specialist."

---

## Slide 15: Live Demonstration — 5-Second Breath-Hold Test
* **The Non-Invasive Demonstration Protocol:**
  1. **Phase 1 (Baseline Sinus Rhythm):** Live oscilloscope renders green Lead-II ECG and rhythmic cyan sinusoidal EDR waves (14 breaths/min). Model outputs *Normal* (98.2% confidence).
  2. **Phase 2 (Voluntary Breath-Hold):** Subject clicks the *৫-সেক শ্বাসরোধ টেস্ট* button or holds breath. R-peak modulation collapses; cyan EDR trace turns red and **instantly flatlines**.
  3. **Phase 3 (AI Detection):** SleepApneaNet flags the episode in **<0.5 seconds**, switching the ICU status badge to bright coral red with **94.5% Apnea Probability**.
* **🎙️ Speaker Notes (What to say):**
  > "To verify our platform live without needing a sleeping patient, we engineered the 5-second breath-hold test. As soon as breath is held, the EDR trace flatlines and our AI model flags an apnea event within half a second."

---

## Slide 16: Economic & Clinical Impact Analysis
* **Bill of Materials (BOM) Breakdown:**
  * ESP32 Microcontroller: **$4.50**
  * AD8232 ECG Front-End: **$3.80**
  * 3-Lead Shielded Biopotential Cable & Electrodes: **$3.20**
  * Li-Po 500mAh Battery & 3D Printed Casing: **$3.50**
  * **Total Hardware BOM:** **~$15.00 USD**
* **Clinical ROI:**
  * **>99% cost reduction** compared to hospital PSG ($2,500 vs $15).
  * Enables massive community screening campaigns in rural upazila health complexes.
  * Early detection prevents secondary strokes, heart attacks, and drowsy driving crashes.
* **🎙️ Speaker Notes (What to say):**
  > "At just fifteen dollars total hardware cost, Sahayak democratizes sleep diagnostics. A single hospital PSG machine costs as much as equipping an entire rural clinic with hundreds of our wearable nodes."

---

## Slide 17: Limitations & Future Engineering Roadmap
* **Technical Limitations:**
  * Distinguishing Obstructive (OSA) vs Central (CSA) requires supplemental airflow thermistry or respiratory effort bands.
  * Gross patient movements in restless sleep introduce electromyographic motion artifacts.
* **Phase 2 Engineering Milestones:**
  1. **Multimodal Optical Sensor Fusion:** Integrating MAX30102 pulse oximeter for synchronous oxygen desaturation index (ODI) and $SpO_2$ logging.
  2. **TinyML On-Device Quantization:** Quantizing SleepApneaNet into 8-bit integers (INT8) to run directly on the ESP32-S3 microcontroller, eliminating cloud latency.
  3. **Clinical Hospital Cohort Validation:** Clinical trial with 500+ overnight patients at local medical college hospitals.
* **🎙️ Speaker Notes (What to say):**
  > "In future iterations, we will fuse photoplethysmography (PPG) for oxygen saturation tracking and quantize the neural network to run directly on the ESP32 chip using TinyML for zero-latency offline screening."

---

## Slide 18: Summary of Core Capstone Contributions
1. **Wearable IoT Engineering:** Designed a $15 medical-grade biopotential acquisition node with 100 Hz strict timer sampling and leads-off safety detection.
2. **Signal Innovation:** Pioneer dual extraction of electrocardiography and ECG-Derived Respiration (EDR) from a single biopotential lead.
3. **Deep Learning Innovation:** SleepApneaNet (1D-CNN + BiLSTM + Attention) achieving 84.1% Accuracy and 0.884 ROC-AUC under strict patient-wise holdout conditions.
4. **Explainability & Localization:** 1D-GradCAM saliency heatmaps confirming clinical decision integrity, paired with a native Bengali medical AI assistant.
* **🎙️ Speaker Notes (What to say):**
  > "In summary, Project Sahayak combines low-cost hardware engineering, advanced signal processing, lightweight deep learning, and localized patient engagement into a single cohesive clinical platform."

---

## Slide 19: Live Production Deployment & Repositories
* **Live Telemedicine Web App:** `https://sahayak-medai.vercel.app`
* **Cloud Gateway & PyTorch API:** `https://sahayak-lkhm.onrender.com`
* **Open Source GitHub Repository:** `https://github.com/MLBioEngineer/sahayak`
* **Local Presentation Slides File:** `D:\sahayak AI\presentation_slides.html`
* **Local Academic Poster File:** `D:\sahayak AI\sahayak_academic_poster_presentation.html`
* **🎙️ Speaker Notes (What to say):**
  > "Every single component of Project Sahayak is fully deployed, containerized, open-source, and accessible right now from any smartphone or browser in the world."

---

## Slide 20: Q&A & Acknowledgements
* **Lead Researcher:** Fazle Ilahi (MLBioEngineer) | `ilahi0184@gmail.com`
* **Department:** Department of Biomedical Engineering
* **Capstone Evaluation Committee 2026**
* **Final Thought:** *"Project Sahayak proves that intelligent algorithms and affordable IoT can bridge the healthcare divide, bringing hospital-grade diagnostics to the people who need it most."*
* **🎙️ Speaker Notes (What to say):**
  > "I would like to sincerely thank my honorable professors, project advisors, and the evaluation committee for their invaluable guidance. I am now open to your questions and ready for the live hardware demonstration."
