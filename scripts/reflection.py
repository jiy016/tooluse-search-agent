# reflection.py

from typing import Dict, Any, Optional, List
import re
import json

from vllm import SamplingParams

from prompts import get_search_o1_reflection_instruction_v2

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"


def _extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _extract_boxed(text: str) -> Optional[str]:
    # Extract \boxed{...} including nested newlines (best-effort)
    m = re.search(r"\\boxed\{([\s\S]*?)\}", text)
    if m:
        return m.group(0).strip()
    return None


def _build_results_preview(search_results: Any, max_items: int = 3, max_chars: int = 1500) -> str:
    """
    search_results can be:
    - list[dict] (already extracted_relevant_info preview)
    - dict (raw bing result)
    This function is defensive; it won't crash.
    """
    try:
        if isinstance(search_results, list):
            items = search_results[:max_items]
            lines: List[str] = []
            for i, it in enumerate(items, 1):
                title = str(it.get("title", "")) if isinstance(it, dict) else ""
                url = str(it.get("url", "")) if isinstance(it, dict) else ""
                snippet = str(it.get("snippet", "")) if isinstance(it, dict) else ""
                lines.append(f"[{i}] title={title}\nurl={url}\nsnippet={snippet}")
            preview = "\n\n".join(lines).strip()
        elif isinstance(search_results, dict):
            # best effort for bing schema: search_results.get("webPages", {}).get("value", [])
            values = []
            wp = search_results.get("webPages", {})
            if isinstance(wp, dict):
                values = wp.get("value", []) or []
            values = values[:max_items]
            lines = []
            for i, it in enumerate(values, 1):
                if not isinstance(it, dict):
                    continue
                lines.append(
                    f"[{i}] title={it.get('name','')}\n"
                    f"url={it.get('url','')}\n"
                    f"snippet={it.get('snippet','')}"
                )
            preview = "\n\n".join(lines).strip()
        else:
            preview = str(search_results)[:max_chars]
    except Exception:
        preview = "(failed to build preview)"

    if len(preview) > max_chars:
        preview = preview[:max_chars] + " ..."
    return preview


def run_reflection(
    llm,
    tokenizer,
    judge_prompt: str,
    *,
    question: str,
    current_reasoning: str,
    search_query: str,
    search_results_preview: Any,
    seq: Dict[str, Any],
    remaining_searches: int,
    max_suggested_queries: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 20,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    LLM-powered reflection.

    Returns a dict with optional fields:
    - "new_search_query": Optional[str]
    - "reflection_note": str
    - "should_append_to_history": bool
    - "history_item": Optional[dict]
    - "stop": bool
    - "final_answer": Optional[str] (boxed)
    """

    preview_text = _build_results_preview(search_results_preview)

    # Build prompt from prompts.py
    user_prompt = get_search_o1_reflection_instruction_v2(
        question=question,
        current_reasoning=current_reasoning,
        judge_prompt=judge_prompt,
        last_search_query=search_query,
        search_results_preview=preview_text,
        remaining_searches=remaining_searches,
    )

    # vLLM expects plain text prompts; we use chat template like the rest of your script
    chat = [{"role": "user", "content": user_prompt}]
    prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    out = llm.generate(
        [prompt_str],
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.05,
            stop=[END_SEARCH_QUERY, tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
    )

    raw_text = out[0].outputs[0].text if out and out[0].outputs else ""
    new_q = _extract_between(raw_text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

    final_boxed = _extract_boxed(raw_text)

    # Build reflection note (for debug)
    reflection_note = (
        "[Reflection-LLM]\n"
        f"Judge said:\n{judge_prompt}\n\n"
        f"Last query: {search_query}\n"
        f"Raw reflection output:\n{raw_text}\n"
    )

    result: Dict[str, Any] = {
        "new_search_query": None,
        "final_answer": None,
        "reflection_note": reflection_note,
        "should_append_to_history": False,
        "history_item": {"role": "system", "content": reflection_note},
        "stop": False,
    }

    # Priority: if boxed exists and no new query tag -> stop
    if final_boxed and not new_q:
        result["final_answer"] = final_boxed
        result["stop"] = True
        return result

    # If model proposes a query, enforce max_suggested_queries and basic sanity
    if new_q and max_suggested_queries > 0:
        new_q = new_q.strip()
        # Avoid empty / same query loops
        if new_q and new_q != search_query:
            result["new_search_query"] = new_q

    return result


def apply_reflection_to_seq(seq: Dict[str, Any], reflection_out: Dict[str, Any]) -> None:
    """
    Apply reflection output to seq state (mutates seq).
    """
    seq.setdefault("reflection_trace", [])
    seq["reflection_trace"].append({
        "reflection_note": reflection_out.get("reflection_note"),
        "new_search_query": reflection_out.get("new_search_query"),
        "final_answer": reflection_out.get("final_answer"),
        "stop": bool(reflection_out.get("stop", False)),
    })

    if reflection_out.get("should_append_to_history") and reflection_out.get("history_item") is not None:
        seq.setdefault("messages", [])
        seq["messages"].append(reflection_out["history_item"])
