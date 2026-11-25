import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from app.core.config import GROQ_API_KEY, GROQ_MODEL
SKILLS = ["python","java","javascript","typescript","react","angular","node.js","fastapi","flask",
"django","spring boot","rest api","sql","mysql","postgresql","mongodb","redis","docker","kubernetes",
"aws","azure","git","jenkins","machine learning","deep learning","nlp","generative ai","llm","rag",
"langchain","langgraph","qdrant","chromadb","faiss","prompt engineering","agentic ai","microservices",
"system design","pytest","pandas","numpy","scikit-learn","tensorflow","pytorch"]
class State(TypedDict, total=False):
    jd: str; evidence: List[str]; skills: List[str]; matched: List[str]; missing: List[str]
    recommendations: List[str]; summary: str; trace: List[str]
def jd_node(s):
    jd=s["jd"].lower()
    found=[x for x in SKILLS if re.search(r"(?<![a-z0-9])"+re.escape(x)+r"(?![a-z0-9])",jd)]
    return {"skills":found,"trace":["JD Analyzer: extracted role skills"]}
def evidence_node(s):
    return {"trace":["Evidence Retriever: selected relevant resume passages from Qdrant"]}
def match_node(s):
    blob=" ".join(s.get("evidence",[])).lower()
    yes=[x for x in s.get("skills",[]) if x in blob]
    no=[x for x in s.get("skills",[]) if x not in blob]
    return {"matched":yes,"missing":no,"trace":["Skill Matcher: compared requirements with retrieved evidence"]}
def advice_node(s):
    rec=[]
    if s.get("missing"): rec.append("Not found in retrieved evidence: "+", ".join(s["missing"])+". Add only skills you genuinely possess.")
    if s.get("matched"): rec.append("Add specific project examples and verified outcomes for: "+", ".join(s["matched"])+".")
    rec.append("Use action + technical approach + verified result. Never invent metrics or responsibilities.")
    return {"recommendations":rec,"summary":f"Found {len(s.get('matched',[]))} of {len(s.get('skills',[]))} detected JD skills in retrieved resume evidence.",
            "trace":["Recommendation Agent: produced evidence-grounded guidance"]}


g = StateGraph(State)

# Register all four nodes
g.add_node("jd_analysis", jd_node)
g.add_node("retrieve_evidence", evidence_node)
g.add_node("skill_match", match_node)
g.add_node("recommendation_agent", advice_node)

# Define workflow sequence
g.set_entry_point("jd_analysis")

g.add_edge("jd_analysis", "retrieve_evidence")
g.add_edge("retrieve_evidence", "skill_match")
g.add_edge("skill_match", "recommendation_agent")
g.add_edge("recommendation_agent", END)

# Compile graph
GRAPH = g.compile()


def run_graph(jd, evidence):
    return GRAPH.invoke({
        "jd": jd,
        "evidence": evidence,
        "trace": []
    })


def run_graph(jd, evidence):
    return GRAPH.invoke({
        "jd": jd,
        "evidence": evidence,
        "trace": []
    })

def run_graph(jd,evidence): return GRAPH.invoke({"jd":jd,"evidence":evidence,"trace":[]})
async def groq_text(prompt):
    if not GROQ_API_KEY: return None
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        model=ChatGroq(model=GROQ_MODEL,api_key=GROQ_API_KEY,temperature=0.2)
        r=await model.ainvoke([SystemMessage(content="Be a factual career coach. Never invent candidate experience, skills or metrics."),HumanMessage(content=prompt)])
        return r.content
    except Exception: return None
