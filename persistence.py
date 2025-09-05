import os
import uuid
import logging
import json
from typing import Any, Dict, List, Optional

logger = logging.getLogger("persistence")


def get_supabase() -> None:
    # SDK disabled: always use REST in this project.
    return None


def _rest_headers() -> Optional[dict]:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not key:
        return None
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation, resolution=merge-duplicates",
    }


def _rest_base_url() -> Optional[str]:
    base = os.getenv("SUPABASE_URL")
    if not base:
        return None
    return base.rstrip("/") + "/rest/v1"


def _rest_upsert(table: str, rows: list) -> bool:
    import requests

    headers = _rest_headers()
    base = _rest_base_url()
    if not headers or not base:
        logger.warning("REST fallback not configured: missing headers/base URL")
        return False
    url = f"{base}/{table}?on_conflict=id"
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(rows), timeout=30)
        if not resp.ok:
            logger.error("Supabase REST upsert failed", extra={"status": resp.status_code, "text": resp.text[:500]})
            return False
        return True
    except Exception as e:
        logger.error("Supabase REST request error", extra={"error": str(e)})
        return False


def save_selection(data: Dict[str, Any]) -> Optional[str]:
    sb = get_supabase()
    # Payload
    payload = {
        "id": data.get("id") or str(uuid.uuid4()),
        "source_type": data.get("source_type", "unknown"),
        "input_params": data.get("input_params", {}),
        "items": data.get("items", []),
        "notes": data.get("notes", ""),
        "version": int(data.get("version", 1)),
    }
    # SDK path
    if sb:
        try:
            sb.table("selections").upsert(payload).execute()
            logger.info("Saved selection (SDK)", extra={"selection_id": payload["id"], "items_len": len(payload.get("items", []))})
            return payload["id"]
        except Exception as e:
            logger.error("Failed to save selection via SDK", extra={"error": str(e)})
    # REST
    ok = _rest_upsert("selections", [payload])
    if ok:
        logger.info("Saved selection (REST)", extra={"selection_id": payload["id"], "items_len": len(payload.get("items", []))})
        return payload["id"]
    return None


def save_tasks(selection_id: Optional[str], tasks: List[Dict[str, Any]], raw_output: Optional[Dict[str, Any]] = None) -> None:
    sb = get_supabase()
    rows = []
    for t in tasks:
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "selection_id": selection_id,
                "title": t.get("title", ""),
                "description": t.get("description", ""),
                "category": t.get("category"),
                "priority": t.get("priority"),
                "raw_output": raw_output or {},
            }
        )
    if not rows:
        logger.info("No tasks to save")
        return
    # SDK path
    if sb:
        try:
            sb.table("tasks").upsert(rows).execute()
            logger.info("Saved tasks (SDK)", extra={"count": len(rows), "selection_id": selection_id})
            return
        except Exception as e:
            logger.error("Failed to save tasks via SDK", extra={"error": str(e), "count": len(rows)})
    # REST
    ok = _rest_upsert("tasks", rows)
    if ok:
        logger.info("Saved tasks (REST)", extra={"count": len(rows), "selection_id": selection_id})
    else:
        logger.error("Failed to save tasks via REST", extra={"count": len(rows)})


