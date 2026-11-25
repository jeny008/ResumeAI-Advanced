import json
from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import UPLOAD_DIR, FRONTEND_ORIGIN
from app.schemas import AnalyzeRequest
from app.services.parser import parse_resume, ParseError
from app.services.vector_store import index_resume, retrieve
from app.agents.workflow import run_graph, groq_text
app=FastAPI(title="ResumeAI Advanced API",version="1.0.0",
 description="Resume Career Copilot using LangChain, Qdrant and LangGraph.")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173",FRONTEND_ORIGIN],
 allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
@app.get("/health")
def health(): return {"status":"ok","service":"resumeai-advanced-api"}
@app.post("/api/resumes/upload")
async def upload(file: UploadFile=File(...)):
    name=Path(file.filename or "resume.txt").name
    try: text=parse_resume(name,await file.read())
    except ParseError as e: raise HTTPException(status_code=400,detail=str(e))
    rid=str(uuid4()); (UPLOAD_DIR/f"{rid}_{name}").write_text(text,encoding="utf-8")
    try: chunks=index_resume(rid,text)
    except Exception as e: raise HTTPException(status_code=500,detail=f"Indexing failed: {str(e)[:220]}")
    return {"resume_id":rid,"filename":name,"word_count":len(text.split()),"chunk_count":chunks,
            "message":"Resume parsed, embedded and indexed in Qdrant."}
@app.post("/api/analyze")
async def analyze(p: AnalyzeRequest):
    try:
        evidence=retrieve(p.resume_id,p.job_description,6)
        if not evidence: raise HTTPException(status_code=404,detail="No resume passages found. Upload resume again.")
        s=run_graph(p.job_description,evidence); skills=s.get("skills",[]); matched=s.get("matched",[])
        score=round(100*len(matched)/max(1,len(skills)))
        prompt=("Return concise JSON with keys role_summary and recommendations (array). Use only resume evidence. "
        "Do not invent candidate skills or achievements.\nJOB DESCRIPTION:\n"+p.job_description[:6000]+
        "\nRESUME EVIDENCE:\n"+"\n".join(evidence)[:7000])
        answer=await groq_text(prompt); parsed=None
        if answer:
            try: parsed=json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
            except Exception: parsed={"role_summary":answer[:1500]}
        return {"resume_id":p.resume_id,"summary":(parsed or {}).get("role_summary",s.get("summary","")),
        "match_score":score,"matched_skills":matched,"missing_skills":s.get("missing",[]),
        "evidence":evidence,"recommendations":(parsed or {}).get("recommendations",s.get("recommendations",[])),
        "workflow":s.get("trace",[]),"llm_used":bool(parsed),"llm_analysis":parsed}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500,detail=f"Analysis failed: {str(e)[:250]}")
@app.post("/api/interview/questions")
async def questions(p: AnalyzeRequest):
    evidence=retrieve(p.resume_id,p.job_description,6)
    prompt=("Create 8 tailored interview questions. Return JSON arrays technical_questions, project_questions, behavioral_questions. "
    "Do not assume skills not evidenced.\nJD:\n"+p.job_description[:5000]+"\nRESUME:\n"+"\n".join(evidence)[:6000])
    answer=await groq_text(prompt)
    if answer:
        try: return json.loads(answer.strip().removeprefix("```json").removesuffix("```").strip())
        except Exception: pass
    return {"technical_questions":["Explain a core technology required by this role.","How would you design and test a reliable REST API?","How do you debug a production issue?"],
    "project_questions":["Walk through your most relevant project.","What trade-offs did you make in its architecture?"],
    "behavioral_questions":["Describe a difficult problem you solved.","How do you handle changing requirements.","What would you improve in your last project?"],
    "note":"Configure GROQ_API_KEY for personalized LLM-generated questions."}
