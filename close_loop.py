from __future__ import annotations

import argparse
import ast
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from gemini import GeminiInterface
from chatgpt import ChatGPTInterface
from deepseek import DeepSeekInterface
# from claude import ClaudeInterface
# from qwen import QwenInterface


# =============================================================================
# Environment variables
# =============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# =============================================================================
# Model initialization
# =============================================================================
def _require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}. "
            f"Export it first, e.g. `export {name}=...`"
        )
    return value


gemini = GeminiInterface(
    api_key=_require_env("GEMINI_API_KEY", GEMINI_API_KEY),
    model_name="gemini-3-pro-preview",
    max_tokens=9600,
)

chatgpt = ChatGPTInterface(
    api_key=_require_env("CHATGPT_API_KEY", CHATGPT_API_KEY),
    model_name="gpt-5.2",
    max_tokens=9600,
)

deepseek = DeepSeekInterface(
    api_key=_require_env("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
    model_name="deepseek-reasoner",
    max_tokens=9600,
)

# claude = ClaudeInterface(...)
# qwen = QwenInterface(...)


# =============================================================================
# Prompt loading / utilities
# =============================================================================
def load_prompt(prefix: Optional[str] = None, output_lang: str = "zh") -> List[str]:
    """
    Load the user prompt from disk and optionally enforce the output language.

    Args:
        prefix: If provided, load `prompt_{prefix}.md`. Otherwise load `prompt.md`.
        output_lang: "zh" for Simplified Chinese, "en" for English.

    Returns:
        A list containing one prompt string (kept as a list for interface compatibility).
    """
    prompt_file = f"prompt_{prefix}.md" if prefix else "prompt.md"

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_str = f.read().strip()

    if output_lang == "zh":
        lang_hint = "Please answer the following in Simplified Chinese:\n"
    elif output_lang == "en":
        lang_hint = "Please answer the following in English:\n"
    else:
        raise ValueError("output_lang must be 'zh' or 'en'")

    return [lang_hint + prompt_str]


def parse_list_response(text: str) -> List[Any]:
    """
    Parse a model response in the required format: a Python list [bool, str].

    Strategy:
      1) Try ast.literal_eval directly.
      2) If that fails, strip a single code fence block and try again.
      3) If still fails, fall back to a simple heuristic:
         - If 'true' appears (and not 'false') in the first 200 chars -> True
         - Else -> False
         The answer field will be the raw text.

    Returns:
        [agree: bool, answer: str]
    """
    try:
        data = ast.literal_eval(text)
        if (
            isinstance(data, list)
            and len(data) == 2
            and isinstance(data[0], bool)
            and isinstance(data[1], str)
        ):
            return data
    except Exception:
        pass

    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()

    try:
        data = ast.literal_eval(cleaned)
        if (
            isinstance(data, list)
            and len(data) == 2
            and isinstance(data[0], bool)
            and isinstance(data[1], str)
        ):
            return data
    except Exception:
        pass

    head = cleaned[:200].lower()
    if "true" in head and "false" not in head:
        agree = True
    elif "false" in head and "true" not in head:
        agree = False
    else:
        agree = False

    return [agree, text]


def write_log_to_file(log_lines: List[str], filename: str) -> None:
    """Write a Markdown log file to disk."""
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for entry in log_lines:
            f.write(entry.rstrip() + "\n\n")


def write_final_answers_to_file(
    final_answers: Dict[str, str],
    filename: str,
    output_lang: str,
) -> None:
    """Write each model's final answer to a Markdown file."""
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    title = (
        "# Final Answers (Re-answered Original Prompt)\n"
        if output_lang == "en"
        else "# 最终答案（重新回答原始问题）\n"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(title)
        for model_name, ans in final_answers.items():
            f.write(f"## {model_name}\n{ans}\n\n")


# =============================================================================
# Debate loop
# =============================================================================
ModelState = Dict[str, Any]


def close_loop_ask(
    prompt: List[str],
    max_loops: int = 20,
    log_filename: str = "logs/default/dialog_log.md",
    final_answers_filename: str = "logs/default/final_answers.md",
    output_lang: str = "zh",
) -> Tuple[str, str, str, str, str, str]:
    """
    Run a multi-model debate loop until consensus is reached (or max_loops is hit).

    Consensus criterion:
      - Among models that successfully return in a round, all output agree=True.

    Args:
        prompt: The initial user prompt (list of one string).
        max_loops: Maximum debate rounds after the initial round.
        log_filename: Where to write the full dialogue log (Markdown).
        final_answers_filename: Where to write final long answers (Markdown).
        output_lang: Final output language ("zh" or "en"). Debate prompts are always English.

    Returns:
        (gemini_final, chatgpt_final, deepseek_final,
         last_gemini_struct, last_chatgpt_struct, last_deepseek_struct)
    """

    def get_model(models: List[ModelState], mid: str) -> Optional[ModelState]:
        for m in models:
            if m["id"] == mid:
                return m
        return None

    def build_debate_prompt(target_model: ModelState, participants: List[ModelState]) -> str:
        """
        Build the per-round debate prompt for a specific model.

        Output format requirement:
          - Must return a Python list of length 2: [bool, str]
          - bool indicates whether the successful models have converged on key conclusions.
          - str is the model's message to the other models (debate content, in English).
        """
        mid = target_model["id"]

        others = [m for m in participants if m["id"] != mid]
        if others:
            other_desc_lines = []
            for m in others:
                other_desc_lines.append(
                    f"[{m['name']} - Latest Answer]\n{m['last_answer']}\n"
                    "------------------------------\n"
                )
            others_block = "".join(other_desc_lines)
        else:
            others_block = "(Only you returned a valid answer in the previous round.)\n"

        if mid in ("gemini", "chatgpt", "claude"):
            header = (
                "You are participating in a multi-model debate on the same user question.\n"
                "You have access to web search tools in this environment.\n"
                "If other models request a fact check or up-to-date info, you may search and bring back evidence.\n\n"
            )
        else:
            header = (
                "You are participating in a multi-model debate on the same user question.\n"
                "You do NOT have direct web search access in this environment.\n"
                "If you need up-to-date facts, explicitly ask the web-enabled models to verify specific claims.\n\n"
            )

        meta = (
            "You will see other models' latest outputs below.\n"
            "------------------------------\n"
            f"{others_block}\n"
            "Your tasks:\n"
            "1) Critically judge whether you agree with the other models' key conclusions and reasoning.\n"
            "2) Group viewpoints: which model aligns with you, and which is incorrect or missing key points?\n"
            "3) If you change your mind, explicitly state what convinced you and how your stance updated.\n"
            "4) Focus on convergence:\n"
            "   - Identify established consensus.\n"
            "   - For disagreements, strengthen arguments or propose a resolution path.\n"
            "   - Avoid rewriting a full final answer; prioritize deltas, corrections, and persuasion.\n"
            "5) Write directly to the other models: debate, challenge, reconcile.\n"
        )

        web_part = (
            "\nWeb search guidance:\n"
            "  - If you can search, do so when factual accuracy or recency matters.\n"
            "  - If you use web search, include a short 'References:' list with key URLs.\n"
            "  - If you do not use web search, write 'References: none'.\n"
            if target_model["supports_web"]
            else
            "\nWeb search guidance:\n"
            "  - You cannot browse the web directly.\n"
            "  - You may cite links surfaced by other models; label them as 'via <model name>'.\n"
        )

        tail = (
            "\nOutput format (strict):\n"
            "  - Return a Python list of length 2:\n"
            "    [<agree_bool>, <message_str>]\n"
            "  - agree_bool = True only if you believe all successful models have converged on key conclusions.\n"
            "  - message_str must be English and should be addressed to other models.\n"
        )

        return header + meta + web_part + tail

    def build_final_prompt(target_model: ModelState) -> str:
        """
        Build the final long-form answer prompt for one model.
        The final answer language is controlled by output_lang.
        """
        lang_line = (
            "Write the final answer in Simplified Chinese."
            if output_lang == "zh"
            else "Write the final answer in English."
        )

        head = (
            "All online models have substantially converged on the question.\n"
            f"You are {target_model['name']}. Now write a final long-form answer for the user.\n"
            "The user does not care about the debate process. Produce a standalone, polished report.\n\n"
        )

        body = (
            f"Language requirement: {lang_line}\n\n"
            "Writing requirements:\n"
            "1) Do NOT mention models, debate, rounds, or 'another model said...'.\n"
            "2) Start with a concise overview (1–3 paragraphs) of key conclusions.\n"
            "3) Expand in sections with clear reasoning and concrete examples.\n"
            "4) Include uncertainty, limitations, and common misinterpretations.\n"
            "5) End with a practical summary and actionable suggestions.\n"
            "6) Length: be thorough. If the topic benefits from detail, write a long answer.\n"
            "7) If you used web search or relied on external links earlier, integrate them naturally\n"
            "   and add a 'References:' section with key URLs. Otherwise write 'References: none'.\n\n"
            "Now output the full final answer.\n"
        )

        if not target_model["supports_web"]:
            extra = (
                "Note: You cannot browse the web directly. You may include references previously surfaced\n"
                "by web-enabled participants; label them as 'via <model name>: <url>'.\n\n"
            )
            return head + extra + body

        return head + body

    # =========================
    # Start
    # =========================
    log: List[str] = []
    prompt_text = " ".join(str(p) for p in prompt)
    log.append(
        "=== Initial User Prompt ===\n"
        f"{prompt_text}\n"
        f"(timestamp: {datetime.now().isoformat()})"
    )

    models: List[ModelState] = [
        {
            "id": "gemini",
            "name": "Google Gemini",
            "interface": gemini,
            "supports_web": True,
            "temporarily_down": False,
            "last_answer": "",
            "last_struct": "",
            "last_agree": False,
        },
        {
            "id": "chatgpt",
            "name": "OpenAI ChatGPT",
            "interface": chatgpt,
            "supports_web": True,
            "temporarily_down": False,
            "last_answer": "",
            "last_struct": "",
            "last_agree": False,
        },
        {
            "id": "deepseek",
            "name": "DeepSeek",
            "interface": deepseek,
            "supports_web": False,
            "temporarily_down": False,
            "last_answer": "",
            "last_struct": "",
            "last_agree": False,
        },
        # {
        #     "id": "claude",
        #     "name": "Anthropic Claude",
        #     "interface": claude,
        #     "supports_web": True,
        #     "temporarily_down": False,
        #     "last_answer": "",
        #     "last_struct": "",
        #     "last_agree": False,
        # },
        # {
        #     "id": "qwen",
        #     "name": "Alibaba Qwen",
        #     "interface": qwen,
        #     "supports_web": False,
        #     "temporarily_down": False,
        #     "last_answer": "",
        #     "last_struct": "",
        #     "last_agree": False,
        # },
    ]

    loop_idx = 0

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # Initial round: each model answers the user prompt (parallel).
        print("=== Initial Round: All models answer the original prompt ===")
        init_futs = {}

        for m in models:
            mid = m["id"]
            if m["supports_web"]:
                init_futs[mid] = executor.submit(m["interface"].ask, prompt, True)
            else:
                init_futs[mid] = executor.submit(m["interface"].ask, prompt)

        for m in models:
            mid = m["id"]
            try:
                raw = init_futs[mid].result()
                m["last_answer"] = raw
                log.append(
                    f"=== Initial {m['name']} Answer ===\n"
                    f"{raw}\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )
            except Exception as e:
                m["temporarily_down"] = True
                m["last_answer"] = f"[{m['name']} initial call failed: {repr(e)}]"
                log.append(
                    f"=== Initial {m['name']} Error ===\n"
                    f"Error: {repr(e)}\n"
                    "This model failed to return in the initial round; it will be retried in later rounds.\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )

        print("=== Initial Round Complete. Starting Debate Loops ===")

        # Debate loops
        while loop_idx < max_loops:
            loop_idx += 1
            print(f"\n=== Loop {loop_idx} ===")

            participants = [m for m in models if not m["temporarily_down"]]

            if len(participants) < 2:
                log.append(
                    f"=== Loop {loop_idx}: Not enough participants (n={len(participants)}). Stopping. ===\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )
                break

            prompts: Dict[str, str] = {}
            for m in participants:
                prompts[m["id"]] = build_debate_prompt(m, participants)

            futs = {}
            for m in participants:
                mid = m["id"]
                if m["supports_web"]:
                    futs[mid] = executor.submit(m["interface"].ask, [prompts[mid]], True)
                else:
                    futs[mid] = executor.submit(m["interface"].ask, [prompts[mid]])

            # Assume everyone is eligible next round; mark down only on failures in this round.
            for m in models:
                m["temporarily_down"] = False

            for m in participants:
                mid = m["id"]
                try:
                    raw = futs[mid].result()
                    m["last_struct"] = raw
                    agree, answer = parse_list_response(raw)
                    m["last_agree"] = bool(agree)
                    m["last_answer"] = str(answer)

                    print(f"{m['name']} parsed agree={agree} (answer length={len(m['last_answer'])})")
                    log.append(
                        f"=== Loop {loop_idx}: {m['name']} Evaluation ===\n"
                        f"Prompt to {m['name']}:\n{prompts[mid]}\n\n"
                        f"Raw output:\n{raw}\n\n"
                        f"Parsed -> agree: {agree}, answer length: {len(m['last_answer'])}\n"
                        f"(timestamp: {datetime.now().isoformat()})"
                    )
                except Exception as e:
                    m["temporarily_down"] = True
                    m["last_struct"] = f"[{m['name']} call failed in loop {loop_idx}: {repr(e)}]"
                    print(m["last_struct"])
                    log.append(
                        f"=== Loop {loop_idx}: {m['name']} Error ===\n"
                        f"Prompt to {m['name']}:\n{prompts[mid]}\n\n"
                        f"Error: {repr(e)}\n"
                        "This model is considered offline for this round, but will be retried next round.\n"
                        f"(timestamp: {datetime.now().isoformat()})"
                    )

            successful_models = [m for m in models if (not m["temporarily_down"]) and m["last_struct"]]
            if successful_models and all(m["last_agree"] for m in successful_models):
                print("\n✅ All successful models returned agree=True. Consensus reached.")
                log.append(
                    "=== Final Agreement (All successful models True) ===\n"
                    + "\n".join(
                        [
                            f"{m['name']} agree={m['last_agree']} answer_len={len(m['last_answer'])}"
                            for m in successful_models
                        ]
                    )
                    + f"\n(timestamp: {datetime.now().isoformat()})"
                )
                break

        # Final long-form answers (for any model that returned something at least once)
        final_futs = {}
        for m in models:
            if not m["last_answer"] and not m["last_struct"]:
                continue

            final_prompt = [build_final_prompt(m)]
            if m["supports_web"]:
                final_futs[m["id"]] = executor.submit(m["interface"].ask, final_prompt, True)
            else:
                final_futs[m["id"]] = executor.submit(m["interface"].ask, final_prompt)

        final_answers_map: Dict[str, str] = {}
        for m in models:
            mid = m["id"]
            if mid not in final_futs:
                continue
            try:
                ans = final_futs[mid].result()
                final_answers_map[m["name"]] = ans
                log.append(
                    f"=== Final Long Answer from {m['name']} ===\n"
                    f"{ans}\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )
            except Exception as e:
                final_answers_map[m["name"]] = f"[{m['name']} final answer failed: {repr(e)}]"
                log.append(
                    f"=== Final Long Answer Error from {m['name']} ===\n"
                    f"Error: {repr(e)}\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )

    # Summarize and write outputs
    def pick_final(mid: str) -> str:
        m = get_model(models, mid)
        if m is None:
            return f"[Unknown model id: {mid}]"
        if m["name"] in final_answers_map:
            return final_answers_map[m["name"]]
        if m["last_answer"]:
            return m["last_answer"]
        if m["last_struct"]:
            return m["last_struct"]
        return f"[{m['name']} produced no usable output]"

    gemini_final = pick_final("gemini")
    gpt_final = pick_final("chatgpt")
    deepseek_final = pick_final("deepseek")

    log.append(
        "=== Final Summary ===\n"
        + "\n".join(
            [
                f"{m['name']} temporarily_down={m['temporarily_down']} has_last_answer={bool(m['last_answer'])}"
                for m in models
            ]
        )
        + f"\n(timestamp: {datetime.now().isoformat()})"
    )

    write_log_to_file(log, log_filename)
    write_final_answers_to_file(final_answers_map, final_answers_filename, output_lang)

    print(f"\n📝 Dialog log exported to {log_filename}")
    print(f"📝 Final answers exported to {final_answers_filename}")

    last_gemini_struct = (get_model(models, "gemini") or {}).get("last_struct", "")
    last_gpt_struct = (get_model(models, "chatgpt") or {}).get("last_struct", "")
    last_deepseek_struct = (get_model(models, "deepseek") or {}).get("last_struct", "")

    return (
        gemini_final,
        gpt_final,
        deepseek_final,
        last_gemini_struct,
        last_gpt_struct,
        last_deepseek_struct,
    )


# =============================================================================
# CLI entrypoint
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-model debate loop.")
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prompt prefix (e.g., 'wzw' -> prompt_wzw.md). Logs go to logs/<prefix>/.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "en"],
        default="zh",
        help="Final output language: 'zh' (Simplified Chinese) or 'en' (English).",
    )
    parser.add_argument(
        "--max_loops",
        type=int,
        default=10,
        help="Maximum debate loops after the initial round.",
    )
    args = parser.parse_args()

    prompt = load_prompt(prefix=args.prefix, output_lang=args.lang)

    if args.prefix:
        output_dir = f"logs/{args.prefix}"
        log_filename = f"{output_dir}/dialog_log_{args.prefix}.md"
        final_answers_filename = f"{output_dir}/final_answers_{args.prefix}.md"
    else:
        output_dir = "logs/default"
        log_filename = f"{output_dir}/dialog_log.md"
        final_answers_filename = f"{output_dir}/final_answers.md"

    os.makedirs(output_dir, exist_ok=True)

    (
        final_gemini,
        final_gpt,
        final_deepseek,
        gemini_debug,
        gpt_debug,
        deepseek_debug,
    ) = close_loop_ask(
        prompt,
        max_loops=args.max_loops,
        log_filename=log_filename,
        final_answers_filename=final_answers_filename,
        output_lang=args.lang,
    )

    if args.lang == "en":
        print("\n=== Final Answers (Re-answered Original Prompt) ===")
    else:
        print("\n=== 最终答案（重新回答原始问题） ===")

    print("Gemini:", final_gemini)
    print("ChatGPT:", final_gpt)
    print("DeepSeek:", final_deepseek)
