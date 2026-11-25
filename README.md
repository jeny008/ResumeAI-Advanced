# ResumeAI Advanced — AI Career Copilot

Local learning project: React + FastAPI + LangChain + LangGraph + Hugging Face embeddings + Qdrant.

## Features
- Upload PDF, DOCX, TXT resumes.
- Split resume into chunks, create sentence-transformer embeddings and index in local Qdrant.
- Retrieve relevant resume evidence for a pasted job description.
- Run a LangGraph workflow: JD skill extraction → evidence comparison → recommendations.
- Optional Groq LLM assistance for recommendations and interview questions.
- React dashboard and FastAPI Swagger docs.

This is an advanced learning MVP, not a production ATS. Match score is an estimated skill overlap, not a hiring prediction. No user accounts or multi-tenant security are included. Use only resumes you have permission to process.

## Requirements
Python 3.10–3.12 recommended (Python 3.11 is safest), Node.js 18+, npm.

## Backend (Terminal 1)
```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
API: http://localhost:8000 | Docs: http://localhost:8000/docs | Health: http://localhost:8000/health

First run downloads `sentence-transformers/all-MiniLM-L6-v2`; allow a few minutes.

Optional Groq: add `GROQ_API_KEY=your_key` to `backend/.env` and restart backend. Without it, vector retrieval and deterministic LangGraph workflow still work; LLM-generated wording uses fallback.

## Frontend (Terminal 2, project root)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

Optional frontend API override: copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE`.

## API
- `POST /api/resumes/upload` (multipart field: `file`)
- `POST /api/analyze` JSON: `{"resume_id":"...","job_description":"..."}`
- `POST /api/interview/questions` same JSON
- `GET /api/resumes/{resume_id}`

Uploaded files and Qdrant data stay locally under `backend/data/`. Scanned PDFs need OCR, not included in this version. Stop duplicate backend processes if Qdrant local storage reports a lock error.

Next production milestones: PostgreSQL, authentication, secure file deletion, background jobs, Docker, RAGAS/LangSmith evaluation, automated tests and deployment.
