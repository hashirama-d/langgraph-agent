import os
import json
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import requests
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict, total=False):
    question: str
    vanna_id: str
    sql: str
    records: List[Dict[str, Any]]
    tasks: List[Dict[str, str]]
    site: Optional[str]
    top_n: Optional[int]


def build_default_question(site: Optional[str], top_n: int) -> str:
    base = f"Select the top {top_n} most important SEO issues"
    if site:
        base += f" for the site {site}"
    base += (
        ". Return a table with columns like issue, category, severity, pages_affected, and recommendation. "
        "Order by severity DESC, pages_affected DESC. Limit to " + str(top_n)
    )
    return base


def vanna_base_url() -> str:
    return os.getenv("VANNA_BASE_URL", "http://localhost:8000")


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


def node_formulate(state: AgentState) -> AgentState:
    if not state.get("question"):
        state["question"] = build_default_question(
            site=state.get("site"),  # type: ignore
            top_n=state.get("top_n", 10),  # type: ignore
        )
    return state


def node_call_vanna(state: AgentState) -> AgentState:
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
    return state


def node_generate_tasks(state: AgentState) -> AgentState:
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
        norm_tasks: List[Dict[str, str]] = []
        for t in tasks:
            title = str(t.get("title", "")).strip()
            description = str(t.get("description", "")).strip()
            if title:
                norm_tasks.append({"title": title, "description": description})
        state["tasks"] = norm_tasks
    except Exception:
        state["tasks"] = []
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
    return sg.compile(checkpointer=CHECKPOINTER)


CHECKPOINTER = MemorySaver()

GRAPH = build_graph()


