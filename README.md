# সহায়ক (Sahayak) — Bengali Medical AI Chatbot

**সহায়ক (Sahayak)** is a Bengali-first medical AI chatbot designed to provide preliminary health guidance and medical awareness using self-hosted LLMs without any third-party AI APIs.

---

## 🌐 Live Production Deployments

- **Backend API (Render)**: [https://sahayak-lkhm.onrender.com](https://sahayak-lkhm.onrender.com)
  - Health check: `GET /health`
  - Chat endpoint: `POST /chat`
  - PDF Export: `POST /export-pdf`
- **Frontend (Vercel)**: Connects to Render backend and deploys via Vercel Git integration.

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
