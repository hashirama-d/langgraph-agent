import os
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# LangGraph (simple linear graph)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangSmith tracing (decorator)
try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap


# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("seo_tasks_agent")


class AgentState(TypedDict, total=False):
    request_id: str
    question: str
    vanna_id: str
    sql: str
    records: List[Dict[str, Any]]
    tasks: List[Dict[str, str]]
    site: Optional[str]
    top_n: Optional[int]


class TasksRequest(BaseModel):
    question: Optional[str] = Field(
        default=None,
        description="Natural language question to send to Vanna. If omitted, will be generated from site/top_n.",
    )
    site: Optional[str] = Field(default=None, description="Site/domain context for the question")
    top_n: int = Field(default=10, ge=1, le=100, description="How many issues to ask for")


class TasksResponse(BaseModel):
    tasks: List[Dict[str, str]]


# LangSmith tracing (optional)
try:
    from langchain import set_tracing_v2_enabled

    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("1", "true", "yes", "y", "on"):
        set_tracing_v2_enabled(True)
except Exception:
    pass


app = FastAPI(title="SEO Tasks Agent", version="0.1.0")


def build_default_question(site: Optional[str], top_n: int) -> str:
    return (
        "Select the top 10 most important SEO issues. For each issue, provide a list of urls, where issue happens. Ensure the issue is present in overview table before searching"
    )


def vanna_base_url() -> str:
    return os.getenv("VANNA_BASE_URL", "http://localhost:8000")


def call_vanna_generate_sql(question: str) -> Dict[str, Any]:
    url = f"{vanna_base_url().rstrip('/')}/api/v0/generate_sql"
    params = {"question": question}
    r = requests.get(url, params=params, timeout=120)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid response from Vanna generate_sql")
    if data.get("type") == "error":
        raise HTTPException(status_code=502, detail=f"Vanna error (generate_sql): {data.get('error')}")
    if data.get("type") not in ("sql", "text"):
        raise HTTPException(status_code=502, detail="Unexpected Vanna response type for generate_sql")
    return data


def call_vanna_run_sql(vanna_id: str) -> Dict[str, Any]:
    url = f"{vanna_base_url().rstrip('/')}/api/v0/run_sql"
    params = {"id": vanna_id}
    r = requests.get(url, params=params, timeout=180)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid response from Vanna run_sql")
    if data.get("type") in ("error", "sql_error"):
        raise HTTPException(status_code=502, detail=f"Vanna error (run_sql): {data.get('error')}")
    if data.get("type") != "df":
        raise HTTPException(status_code=502, detail="Unexpected Vanna response type for run_sql")
    return data


@traceable(name="formulate")
def node_formulate(state: AgentState) -> AgentState:
    start_t = time.monotonic()
    if not state.get("request_id"):
        state["request_id"] = str(uuid.uuid4())
    if not state.get("question"):
        state["question"] = build_default_question(
            site=state.get("site"),  # type: ignore
            top_n=state.get("top_n", 10),  # type: ignore
        )
    dur_ms = int((time.monotonic() - start_t) * 1000)
    logger.info(
        "formulate completed",
        extra={
            "request_id": state.get("request_id"),
            "duration_ms": dur_ms,
            "site": state.get("site"),
            "top_n": state.get("top_n"),
            "question_preview": str(state.get("question", ""))[:200],
        },
    )
    return state


@traceable(name="vanna")
def node_call_vanna(state: AgentState) -> AgentState:
    start_t = time.monotonic()
    question = state["question"]
    gen = call_vanna_generate_sql(question)
    v_id = gen.get("id")
    sql_text = gen.get("text", "")
    if not v_id:
        raise HTTPException(status_code=502, detail="Vanna did not return an id for caching")
    state["vanna_id"] = v_id
    state["sql"] = sql_text

    df_resp = call_vanna_run_sql(vanna_id=v_id)
    df_json_str = df_resp.get("df")
    try:
        records = json.loads(df_json_str)
        if not isinstance(records, list):
            records = []
    except Exception:
        records = []
    state["records"] = records  # raw rows from Vanna (head up to 10)
    dur_ms = int((time.monotonic() - start_t) * 1000)
    logger.info(
        "vanna completed",
        extra={
            "request_id": state.get("request_id"),
            "duration_ms": dur_ms,
            "vanna_id": v_id,
            "sql_length": len(sql_text or ""),
            "records_count": len(records),
        },
    )
    return state


@traceable(name="tasks")
def node_generate_tasks(state: AgentState) -> AgentState:
    # Use OpenAI to transform records into tasks in English
    start_t = time.monotonic()
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Load system prompt JSON string from .txt file and pass it raw to the model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_txt_path = os.path.join(base_dir, "prompts", "tasks_system_prompt.txt")
    try:
        with open(prompt_txt_path, "r", encoding="utf-8") as f:
            instructions = f.read()
    except Exception:
        # Safe fallback (minimal)
        instructions = (
            "{\n  \"role\": \"system\",\n  \"goal\": \"Convert structured rows of SEO issues into tasks.\",\n  \"output\": {\n    \"format\": \"json_object\",\n    \"schema\": {\"type\": \"object\", \"properties\": {\"tasks\": {\"type\": \"array\"}}, \"required\": [\"tasks\"]}\n  }\n}"
        )

    content = {
        "question": state.get("question", ""),
        "sql": state.get("sql", ""),
        "rows": state.get("records", []),
    }

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": "Transform these rows into tasks (JSON only):\n" + json.dumps(content, ensure_ascii=False),
            },
        ],
        temperature=1,
        response_format={"type": "json_object"},
    )

    text = completion.choices[0].message.content or "{}"
    try:
        obj = json.loads(text)
        tasks = obj.get("tasks", [])
        # Normalize to only title/description strings
        norm_tasks: List[Dict[str, str]] = []
        for t in tasks:
            title = str(t.get("title", "")).strip()
            description = str(t.get("description", "")).strip()
            if title:
                norm_tasks.append({"title": title, "description": description})
        state["tasks"] = norm_tasks
    except Exception:
        state["tasks"] = []
    dur_ms = int((time.monotonic() - start_t) * 1000)
    logger.info(
        "tasks generation completed",
        extra={
            "request_id": state.get("request_id"),
            "duration_ms": dur_ms,
            "llm_model": model,
            "rows_count": len(state.get("records", [])),
            "tasks_count": len(state.get("tasks", [])),
        },
    )
    return state


def build_graph():
    sg = StateGraph(AgentState)
    sg.add_node("formulate", node_formulate)
    sg.add_node("vanna", node_call_vanna)
    sg.add_node("tasks", node_generate_tasks)

    sg.set_entry_point("formulate")
    sg.add_edge("formulate", "vanna")
    sg.add_edge("vanna", "tasks")
    sg.add_edge("tasks", END)
    checkpointer = MemorySaver()
    return sg.compile(checkpointer=checkpointer)


GRAPH = build_graph()


@app.post("/tasks", response_model=TasksResponse)
def generate_tasks(req: TasksRequest) -> TasksResponse:
    request_id = str(uuid.uuid4())
    logger.info("/tasks invoked", extra={"request_id": request_id})
    initial_state: AgentState = {
        "request_id": request_id,
        "question": req.question or "",
        "site": req.site,  # type: ignore
        "top_n": req.top_n,  # type: ignore
    }
    final_state = GRAPH.invoke(initial_state)
    tasks = final_state.get("tasks", [])  # type: ignore
    if not tasks:
        # Fallback: fabricate minimal tasks from rows if LLM failed
        rows = final_state.get("records", [])
        tasks = []
        for idx, row in enumerate(rows):
            title = f"Investigate issue #{idx + 1}"
            description = json.dumps(row, ensure_ascii=False)
            tasks.append({"title": title, "description": description})
    logger.info(
        "/tasks completed",
        extra={
            "request_id": request_id,
            "tasks_count": len(tasks),
        },
    )
    return TasksResponse(tasks=tasks)


