import os
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import requests
from langgraph.graph import StateGraph, END
try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("seo_tasks_agent.graph")


class AgentState(TypedDict, total=False):
    request_id: str
    question: str
    vanna_id: str
    sql: str
    records: List[Dict[str, Any]]
    tasks: List[Dict[str, str]]
    site: Optional[str]
    top_n: Optional[int]
    selection_id: Optional[str]


def vanna_base_url() -> str:
    url = os.getenv("VANNA_BASE_URL")
    if not url:
        raise RuntimeError("VANNA_BASE_URL is not set")
    return url


def call_vanna_generate_sql(question: str) -> Dict[str, Any]:
    url = f"{vanna_base_url().rstrip('/')}/api/v0/generate_sql"
    params = {"question": question}
    r = requests.get(url, params=params, timeout=120)
    data = r.json()
    if data.get("type") == "error":
        raise RuntimeError(f"Vanna error (generate_sql): {data.get('error')}")
    if data.get("type") not in ("sql", "text"):
        raise RuntimeError("Unexpected Vanna response type for generate_sql")
    return data


def call_vanna_run_sql(vanna_id: str) -> Dict[str, Any]:
    url = f"{vanna_base_url().rstrip('/')}/api/v0/run_sql"
    params = {"id": vanna_id}
    r = requests.get(url, params=params, timeout=180)
    data = r.json()
    if data.get("type") in ("error", "sql_error"):
        raise RuntimeError(f"Vanna error (run_sql): {data.get('error')}")
    if data.get("type") != "df":
        raise RuntimeError("Unexpected Vanna response type for run_sql")
    return data


@traceable(name="vanna")
def node_call_vanna(state: AgentState) -> AgentState:
    start_t = time.monotonic()
    question = state["question"]
    gen = call_vanna_generate_sql(question)
    v_id = gen.get("id")
    sql_text = gen.get("text", "")
    if not v_id:
        raise RuntimeError("Vanna did not return an id for caching")
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
    state["records"] = records
    # Persist selection to Supabase if available (REST)
    try:
        from persistence import save_selection

        sel_id = save_selection(
            {
                "source_type": "vanna_sql",
                "input_params": {
                    "question": state.get("question"),
                    "vanna_id": v_id,
                    "sql": sql_text,
                    "site": state.get("site"),
                    "top_n": state.get("top_n"),
                },
                "items": records,
                "notes": "auto-ingested",
                "version": 1,
            }
        )
        if sel_id:
            state["selection_id"] = sel_id
    except Exception as e:
        logger.warning("Supabase save_selection failed", extra={"error": str(e)})
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
    start_t = time.monotonic()
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_txt_path = os.path.join(base_dir, "prompts", "tasks_system_prompt.txt")
    try:
        with open(prompt_txt_path, "r", encoding="utf-8") as f:
            instructions = f.read()
    except Exception:
        instructions = (
            "{\n  \"role\": \"system\",\n  \"goal\": \"Convert structured rows of SEO issues into tasks.\",\n  \"output\": {\n    \"format\": \"json_object\",\n    \"schema\": {\"type\": \"object\", \"properties\": {\"tasks\": {\"type\": \"array\"}}, \"required\": [\"tasks\"]}\n  }\n}"
        )

    # RAG: pull SEO context from Pinecone (rag_seo)
    seo_context = ""
    hits_count = 0
    try:
        from retrievers import get_retriever

        retr = get_retriever("seo")
        rows = state.get("records", [])
        issue_hints = []
        for row in rows[:10]:
            if isinstance(row, dict):
                for key in ("issue", "category", "status_code", "robots", "canonical"):
                    val = row.get(key)
                    if val:
                        issue_hints.append(str(val))
        query = ("; ".join(issue_hints) or state.get("question", "SEO issues"))[:512]
        hits = retr.search(collection="rag_seo", query=query, k=6)
        hits_count = len(hits)
        seo_context = "\n\n---\n\n".join([h.get("text", "") for h in hits if h.get("text")])
        # DEBUG
        try:
            preview = ((hits[0].get("text", "")) if hits else "")[:160].replace("\n", " ")
            logger.debug("rag_seo hits", extra={"count": hits_count, "preview": preview})
        except Exception:
            pass
    except Exception as e:
        logger.debug("rag_seo retrieval failed", extra={"error": str(e)})

    content = {
        "question": state.get("question", ""),
        "sql": state.get("sql", ""),
        "rows": state.get("records", []),
        "seo_context": seo_context,
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
        norm_tasks: List[Dict[str, str]] = []
        for t in tasks:
            title = str(t.get("title", "")).strip()
            description = str(t.get("description", "")).strip()
            if title:
                norm_tasks.append({"title": title, "description": description})
        state["tasks"] = norm_tasks
        # Persist tasks to Supabase if available
        try:
            from persistence import save_tasks

            save_tasks(selection_id=state.get("selection_id"), tasks=norm_tasks, raw_output=obj)
        except Exception as e:
            logger.warning("Supabase save_tasks failed", extra={"error": str(e)})
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
            "rag_seo_hits": hits_count,
        },
    )
    return state


@traceable(name="tasks")
def node_generate_tasks(state: AgentState) -> AgentState:
    start_t = time.monotonic()
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_txt_path = os.path.join(base_dir, "prompts", "tasks_system_prompt.txt")
    try:
        with open(prompt_txt_path, "r", encoding="utf-8") as f:
            instructions = f.read()
    except Exception:
        instructions = (
            "{\n  \"role\": \"system\",\n  \"goal\": \"Convert structured rows of SEO issues into tasks.\",\n  \"output\": {\n    \"format\": \"json_object\",\n    \"schema\": {\"type\": \"object\", \"properties\": {\"tasks\": {\"type\": \"array\"}}, \"required\": [\"tasks\"]}\n  }\n}"
        )

    # RAG: pull SEO context from Pinecone (rag_seo)
    seo_context = ""
    hits_count = 0
    try:
        from retrievers import get_retriever

        retr = get_retriever("seo")
        rows = state.get("records", [])
        issue_hints = []
        for row in rows[:10]:
            if isinstance(row, dict):
                for key in ("issue", "category", "status_code", "robots", "canonical"):
                    val = row.get(key)
                    if val:
                        issue_hints.append(str(val))
        query = ("; ".join(issue_hints) or state.get("question", "SEO issues"))[:512]
        hits = retr.search(collection="rag_seo", query=query, k=6)
        hits_count = len(hits)
        seo_context = "\n\n---\n\n".join([h.get("text", "") for h in hits if h.get("text")])
        # DEBUG
        try:
            preview = ((hits[0].get("text", "")) if hits else "")[:160].replace("\n", " ")
            logger.debug("rag_seo hits", extra={"count": hits_count, "preview": preview})
        except Exception:
            pass
    except Exception as e:
        logger.debug("rag_seo retrieval failed", extra={"error": str(e)})

    content = {
        "question": state.get("question", ""),
        "sql": state.get("sql", ""),
        "rows": state.get("records", []),
        "seo_context": seo_context,
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
        norm_tasks: List[Dict[str, str]] = []
        for t in tasks:
            title = str(t.get("title", "")).strip()
            description = str(t.get("description", "")).strip()
            if title:
                norm_tasks.append({"title": title, "description": description})
        state["tasks"] = norm_tasks
        # Persist tasks to Supabase if available
        try:
            from persistence import save_tasks

            save_tasks(selection_id=state.get("selection_id"), tasks=norm_tasks, raw_output=obj)
        except Exception as e:
            logger.warning("Supabase save_tasks failed", extra={"error": str(e)})
    except Exception:
        state["tasks"] = []
    return state


def build_graph():
    sg = StateGraph(AgentState)
    sg.add_node("vanna", node_call_vanna)
    sg.add_node("tasks", node_generate_tasks)

    sg.set_entry_point("vanna")
    sg.add_edge("vanna", "tasks")
    sg.add_edge("tasks", END)
    return sg.compile()

GRAPH = build_graph()


