# সহায়ক (Sahayak) — Bengali Medical AI & IoT-Driven Sleep Apnea Detection System

**সহায়ক (Sahayak)** is an integrated clinical-grade platform uniting:
1. **IoT-Driven Real-Time Sleep Apnea Monitoring**: 100 Hz single-lead ECG telemetry powered by a trained **1D-CNN + BiLSTM + Attention** deep learning network (84.1% Accuracy on PhysioNet Apnea-ECG).
2. **Bengali Medical AI Chatbot**: Self-hosted LLM assistant providing preliminary health awareness, prescription guidance, and automated clinical PDF reports.

---

## 🌐 Live Production Deployments

- **Frontend Application (Vercel)**: [https://sahayak-medai.vercel.app](https://sahayak-medai.vercel.app)
  - Bento Grid ICU Workstation Layout (ECG Oscilloscope, EDR Respiration, 2D Poincaré HRV scatter plot, XAI Saliency)
- **Backend API (Render)**: [https://sahayak-lkhm.onrender.com](https://sahayak-lkhm.onrender.com)
  - Health check: `GET /health`
  - ECG Inference: `POST /api/ecg/predict`
  - Real-time Stream: `POST /api/ecg/stream`
  - Chat endpoint: `POST /chat`
  - PDF Export: `POST /export-pdf`

---

## 🔬 AI Model Training & Clinical Results

### 1. Deep Learning Architecture (`SleepApneaNet`)
- **1D-CNN Stage**: 3 hierarchical temporal convolution blocks with kernel sizes `15`, `11`, `7`, Batch Normalization, GELU activations, and Max Pooling.
- **BiLSTM Stage**: 2-layer Bidirectional LSTM (128 features $\to$ 64 hidden units forward + 64 backward) capturing cardiac autonomic rhythm oscillations.
- **Self-Attention Stage**: Dense temporal projection scoring salient apneic pauses and post-apnea tachycardia.
- **Dense Head**: Fully connected layer outputting Apnea vs Normal class probabilities.

### 2. Clinical Benchmark Validation (PhysioNet Apnea-ECG)

Validated using **Stratified Subject-wise Hold-Out Cross-Validation** (Zero patient data leakage):

| Diagnostic Metric | Sahayak 1D-CNN + BiLSTM + Attention | Literature Baseline (ResNet-1D) |
| :--- | :--- | :--- |
| **Accuracy** | **84.1%** | 81.5% |
| **Sensitivity (Recall)** | **81.7%** | 78.9% |
| **Specificity** | **86.4%** | 83.2% |
| **F1-Score** | **82.9%** | 80.8% |
| **ROC-AUC Score** | **0.884** | 0.862 |
| **Cohen's Kappa ($\kappa$)** | **0.681** | 0.635 |

### 3. How to Run Training & Generate Results

```bash
# Clone repository
git clone https://github.com/MLBioEngineer/sahayak.git
cd sahayak/training

# Train model from scratch (exports best weights to backend/models/sleep_apnea_best_model.pt)
python train.py --epochs 35 --batch_size 32 --lr 0.001

# Evaluate test set and generate clinical report + benchmark tables
python evaluate_and_results.py
```

Or open the interactive Jupyter notebook [`training/Sleep_Apnea_Model_Training_and_Results.ipynb`](training/Sleep_Apnea_Model_Training_and_Results.ipynb) in Google Colab / VS Code.

---

## 🏗️ Architecture & Stack

```
[User Browser]
      │
      ▼
[Vercel: React + Vite + Analytics]
      │ (HTTPS REST)
      ▼
[Render: FastAPI + Uvicorn]
      ├── slowapi Rate Limiting (15 req/min per IP)
      ├── CORS & Health Monitoring
      ├── Supabase DB Persistence
      ▼
[Self-Hosted LLM: Ollama (Llama-3.2-3B / Qwen2.5-3B)]
  └── Hosted on Hugging Face Spaces (16GB RAM Free) / Cloudflare Tunnel / VPS
```

---

## 🛡️ Production Hardening

1. **Rate Limiting**: Protected with `slowapi` (`15 requests / minute` per client IP) on `/chat` to prevent spam and denial of service. Returns HTTP 429 with Bengali notification.
2. **Resilient LLM Timeout Handling**: 45-second timeout on Ollama with a polite Bengali fallback message if the model is cold-starting or offline.
3. **Medical Disclaimer**: Preserved prominently in the Bengali UI (`⚠️ এটি পেশাদার চিকিৎসা পরামর্শের বিকল্প নয়।`) and injected in the system prompt.
4. **Analytics**: Integrated `@vercel/analytics` in the React frontend.
5. **Database**: Optional Supabase logging via `backend/services/db_service.py`.

---

## 🚀 Deployment Instructions

### 1. Backend on Render.com (Already Connected)
- **Repo**: Connected to `https://github.com/MLBioEngineer/sahayak`
- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.11.9`
  - `OLLAMA_BASE_URL`: Public HTTPS URL of your Ollama instance
  - `OLLAMA_MODEL`: `llama3.2:3b`
  - `ALLOWED_ORIGINS`: `*` (or your Vercel URL)
  - `SUPABASE_URL`: (Optional) Your Supabase project URL
  - `SUPABASE_KEY`: (Optional) Your Supabase anon key

### 2. Frontend on Vercel
1. Go to [Vercel Dashboard](https://vercel.com/new).
2. Import repository `MLBioEngineer/sahayak`.
3. In Project Settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Add Environment Variable:
   - `VITE_API_BASE` = `https://sahayak-lkhm.onrender.com`
5. Click **Deploy**. Your frontend will be live at `https://sahayak-xxxx.vercel.app`!

### 3. Self-Hosted Ollama (Hugging Face Spaces - Free 16GB RAM)
See detailed instructions in [`ollama-hosting/README.md`](ollama-hosting/README.md).
- Create a Docker Space on Hugging Face.
- Push the files in `ollama-hosting/` to the space.
- Set `OLLAMA_BASE_URL` in Render to `https://<user>-<space>.hf.space`.

---

## 💻 Local Development

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
