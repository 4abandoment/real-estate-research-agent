"""Robust JSON extraction for model responses."""

import json
import re


def load_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def parse_scope(text: str) -> dict:
    try:
        data = load_json(text)
    except (ValueError, json.JSONDecodeError):
        return {"needs_clarification": False, "questions": [], "confidence": 1.0, "assumption": ""}
    questions = [str(item) for item in data.get("questions", []) if str(item).strip()]
    try:
        confidence = min(max(float(data.get("confidence", 1.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 1.0
    return {
        "needs_clarification": bool(data.get("needs_clarification")) and bool(questions),
        "questions": questions[:3],
        "confidence": confidence,
        "assumption": str(data.get("assumption", "")).strip(),
    }


def parse_decision(text: str, fallback: str) -> dict:
    answer_match = re.search(
        r"(?:\A|\n)\s*ANSWER:\s*(.*?)(?=\n\s*(?:CONFIDENCE|NEEDS_HUMAN|REASON):|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        confidence_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        human_match = re.search(r"NEEDS_HUMAN:\s*(true|false|yes|no)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        return {
            "answer": answer_match.group(1).strip() or fallback.strip() or text.strip(),
            "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
            "needs_human": human_match is not None
            and human_match.group(1).lower() in {"true", "yes"},
            "reason": reason_match.group(1).strip() if reason_match else "",
        }

    try:
        data = load_json(text)
    except (ValueError, json.JSONDecodeError):
        if text.strip():
            return {
                "answer": text.strip(),
                "confidence": 0.5,
                "needs_human": False,
                "reason": "",
            }
        return {
            "answer": fallback.strip(),
            "confidence": 0.0,
            "needs_human": True,
            "reason": "empty model response",
        }
    return {
        "answer": str(data.get("answer", fallback)).strip(),
        "confidence": float(data.get("confidence", 0.5)),
        "needs_human": bool(data.get("needs_human", False)),
        "reason": str(data.get("reason", "")).strip(),
    }
