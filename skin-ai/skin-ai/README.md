# Skin AI — Clinical Decision Support

FastAPI backend (DINOv2 lesion classification + disease-aware RAG chatbot) with a
Next.js/React frontend.

```
skin-ai/
├── backend/
│   ├── main.py            # FastAPI app: /predict, /chat, /session, /health
│   ├── web_rag_agent.py   # RAG chain, disease->URL map, suggested questions
│   ├── config.py          # env-based config (GROQ_API_KEY, CORS origins)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── app/                # Next.js app router pages
    ├── components/         # ImageUpload, ChatPanel
    ├── lib/api.ts          # typed client for the backend
    └── .env.local.example
```

## 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY (and ALLOWED_ORIGINS if not localhost:3000)

uvicorn main:app --reload --port 8000
```

The first `/predict` call downloads the DINOv2 model from Hugging Face; the first
`/chat` call for a given disease scrapes and embeds that disease's reference URL(s)
(cached in-process afterward). `/health` returns `{"status": "ok"}` once the API is up.

## 2. Frontend setup

```bash
cd frontend
npm install

cp .env.local.example .env.local
# edit if your backend isn't on localhost:8000

npm run dev
```

Open http://localhost:3000. Upload a dermoscopic image → it's classified via
`/predict` → the chat panel activates with suggested questions for that specific
diagnosis, grounded only in reference material for that disease.

## Notes for production deployment

- **Auth**: neither endpoint is authenticated yet. Put this behind your auth layer
  (e.g. a reverse proxy checking a session cookie/JWT, or FastAPI dependency-injected
  auth) before exposing it beyond localhost — this handles clinical data.
- **Session storage**: `web_rag_agent.py` keeps chat history and the RAG chain cache
  in-process memory (`store`, `_retriever_cache`, `_chain_cache`). This resets on
  restart and won't share state across multiple backend workers/replicas. For real
  multi-instance deployment, move `store` to Redis and persist the Chroma vectorstores
  to disk (or a hosted vector DB) instead of rebuilding them at first use.
- **CORS**: set `ALLOWED_ORIGINS` in `backend/.env` to your real frontend domain(s).
- **Model loading**: `warm_up_model()` loads DINOv2 at FastAPI startup so the first
  request isn't slow — expect a slower cold start in exchange.
