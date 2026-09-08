# সহায়ক (Sahayak) — Project Scaffold

Website-first scaffold. The AI model is currently mocked so you can build
and test the full site before your trained model is ready.

## Structure
```
sahayak/
  backend/           FastAPI app
    main.py
    routers/          chat.py, pdf.py
    services/
      ai_service.py    <-- swap this to plug in your trained model later
      pdf_service.py   PDF export logic (reportlab)
    models/schemas.py
    requirements.txt
    .env.example
  frontend/          React + Vite app
    src/
      components/ChatWindow.jsx
      api/client.js
```

## Run the backend
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit http://localhost:8000/health to confirm it's running.

## Run the frontend
```bash
cd frontend
npm install
npm run dev
```
Opens at http://localhost:5173 by default. It talks to the backend at
`http://localhost:8000` (change via `VITE_API_BASE` env var if needed).

## Where your trained model plugs in later
Open `backend/services/ai_service.py` — there is one function,
`get_ai_response()`. Right now it returns a mock reply. When your
fine-tuned model is hosted (Colab / HF Spaces / your own server), replace
the body of that function to call it. Nothing else in the app needs to
change.

## Free-tier deployment (when ready)
- Frontend → Vercel or Netlify (free)
- Backend → Render or Railway (free tier)
- Auth/DB → Supabase (free tier) — not yet wired up in this scaffold
- Model hosting → Hugging Face Spaces (free tier, dev/demo scale)

## Not included yet (by design, per current scope)
- Image/file upload — deferred to a later phase (needs GPU hosting, higher cost)
- Supabase auth wiring — routes are stubbed, auth logic not yet added
- Real model connection — see ai_service.py above
