import re
import ast
import argparse
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# get API keys from system environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

from gemini import GeminiInterface
from chatgpt import ChatGPTInterface
from deepseek import DeepSeekInterface
from claude import ClaudeInterface
from qwen import QwenInterface


# Initialize model interfaces
gemini = GeminiInterface(
    api_key=GEMINI_API_KEY,
    model_name="gemini-3-pro-preview",
    max_tokens=9600,
)

chatgpt = ChatGPTInterface(
    api_key=CHATGPT_API_KEY,
    model_name="gpt-5.2",
    max_tokens=9600,
)

deepseek = DeepSeekInterface(
    api_key=DEEPSEEK_API_KEY,
    model_name="deepseek-reasoner",
    max_tokens=9600,
)

# claude = ClaudeInterface(
#     api_key=CLAUDE_API_KEY,
#     model_name="claude-opus-4-5-20251101",
#     max_tokens=9600,
# )

# qwen = QwenInterface(
#     api_key=QWEN_API_KEY_SG,
#     model_name="qwen3-max-preview",
#     max_tokens=9600,
# )


def load_prompt(prefix: str = None) -> list:
    """
    Load prompt from file based on prefix.
    
    Args:
        prefix: Optional prefix for the prompt file (e.g., "wzw" -> "prompt_wzw.md")
        
    Returns:
        List containing the prompt string
    """
    if prefix:
        prompt_file = f"prompt_{prefix}.md"
    else:
        prompt_file = "prompt.md"
    
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_str = f.read().strip()
    
    # Optional: wrap with Chinese language instruction
    prompt_str = "请用简体中文回答下面的问题：\n" + prompt_str
    return [prompt_str]


def parse_list_response(text: str):
    """
    1. Try strict ast.literal_eval (能解析就最好).
    2. If that fails, try stripping ``` fences and ast again.
    3. If still fails, cheap heuristic with 'true'/'false' in head.
    """
    # 先试一次“干净”的 ast
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

    # 如果 raw 里面有 ``` ```，先把 code fence 去掉一层
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
        cleaned = cleaned.strip()

    # 再试一次 ast.literal_eval
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

    # 暴力方案：看前 200 个字符有没有 true/false
    head = cleaned[:200].lower()
    if "true" in head and "false" not in head:
        agree = True
    elif "false" in head and "true" not in head:
        agree = False
    else:
        agree = False

    # 答案直接用原文（你只是要传给下一轮引用）
    answer = text
    return [agree, answer]


def write_log_to_file(log_lines, filename: str = "dialog_log.md"):
    """
    Write dialog log to file.
    
    Args:
        log_lines: List of log entries to write
        filename: Output filename (can include directory path)
    """
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for entry in log_lines:
            f.write(entry.rstrip() + "\n\n")


def write_final_answers_to_file(
    gemini_answer: str,
    gpt_answer: str,
    deepseek_answer: str,
    # claude_answer: str,
    # qwen_answer: str,
    filename: str = "final_answers.md",
):
    """
    Write final answers to file.
    
    Args:
        gemini_answer: Gemini's final answer
        gpt_answer: ChatGPT's final answer
        deepseek_answer: DeepSeek's final answer
        filename: Output filename (can include directory path)
    """
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Final Answers (Re-answered Original Prompt)\n")
        f.write(f"## Gemini final answer:\n{gemini_answer}\n\n")
        f.write(f"## ChatGPT final answer:\n{gpt_answer}\n\n")
        f.write(f"## DeepSeek final answer:\n{deepseek_answer}\n\n")
        # f.write(f"## Claude final answer:\n{claude_answer}\n\n")
        # f.write(f"## Qwen final answer:\n{qwen_answer}\n")


def close_loop_ask(
    prompt,
    max_loops=20,
    log_filename="dialog_log.md",
    final_answers_filename="final_answers.md",
    prefix=None,
):
    """
    Multi-model debate loop until all successfully returned models have agree=True.
    Models are not permanently offline; they are only marked as temporarily_down in the round where an error occurs.
    They will be retried in the next round.
    
    Args:
        prompt: List containing the prompt string
        max_loops: Maximum number of debate loops
        log_filename: Filename for the dialog log
        final_answers_filename: Filename for the final answers
        prefix: Optional prefix used for file naming (if None, uses default filenames)
    """

    def get_model(models, mid):
        for m in models:
            if m["id"] == mid:
                return m
        return None

    def build_debate_prompt(target_model, participants):
        name = target_model["name"]
        mid = target_model["id"]

        others = [m for m in participants if m["id"] != mid]
        if others:
            other_desc_lines = []
            for m in others:
                other_desc_lines.append(
                    f"【{m['name']} 当前答案】\n{m['last_answer']}\n------------------------------\n"
                )
            others_block = "".join(other_desc_lines)
        else:
            others_block = "（当前只有你一个模型返回了上一轮的答案。）\n"

        # Header by model id /能力说明
        if mid == "gemini":
            header = (
                "你是 Google Gemini 模型，并且具备联网检索 (web search) 功能。\n\n"
                "现在有若干个模型在就同一个用户问题进行多轮中文辩论，你是其中之一。\n"
                "有的模型支持 web search，有的模型不支持。\n"
                "如果你看到了其他模型提出的互联网搜索请求，请替它们完成搜索。\n\n"
            )
        elif mid == "chatgpt":
            header = (
                "你是 OpenAI ChatGPT 模型，并且具备联网检索 (web search) 功能。\n\n"
                "现在有若干个模型在就同一个用户问题进行多轮中文辩论，你是其中之一。\n"
                "有的模型支持 web search，有的模型不支持。\n"
                "如果你看到了其他模型提出的互联网搜索请求，请替它们完成搜索。\n\n"
            )
        elif mid == "claude":
            header = (
                "你是 Anthropic Claude 模型，并且具备联网检索 (web search) 功能。\n\n"
                "现在有若干个模型在就同一个用户问题进行多轮中文辩论，你是其中之一。\n"
                "有的模型支持 web search，有的模型不支持。\n"
                "如果你看到了其他模型提出的互联网搜索请求，请替它们完成搜索。\n\n"
            )
        elif mid == "qwen":
            header = (
                "你是阿里通义千问（Qwen）大模型，目前通过兼容 OpenAI 协议被调用，\n"
                "在本环境下不支持主动联网检索。\n\n"
                "现在有若干个模型在就同一个用户问题进行多轮中文辩论，你是其中之一。\n"
                "其它部分模型可以通过 web search 获取外部信息，你可以参考它们在答案中给出的引用和链接。\n"
                "如果你有不确定的地方需要访问互联网搜索信息，你可以在输出回答中请求其他模型替你搜索。\n\n"
            )
        else:  # deepseek
            header = (
                "你是 DeepSeek 模型 deepseek-reasoner，目前不支持主动联网检索。\n\n"
                "现在有若干个模型在就同一个用户问题进行多轮中文辩论，你是其中之一。\n"
                "其它部分模型可以通过 web search 获取外部信息，你可以参考它们在答案中给出的引用和链接。\n"
                "如果你有不确定的地方需要访问互联网搜索信息，你可以在输出回答中请求其他模型替你搜索。\n\n"
            )
        meta = (
            "本轮你将看到其它模型最新的输出。\n"
            "------------------------------\n"
            f"{others_block}\n"
            "请你在充分理解这些内容的前提下，完成以下任务：\n"
            "1. 批判性地判断你是否同意其它模型当前答案的全部结论与推理过程。\n"
            "2. 尝试对不同观点进行归类：谁和你更接近？谁出现了明显错误或遗漏？\n"
            "3. 如果你发现自己原始判断有错误或不够严谨，请坦然承认并明确写出你被说服/更新立场的地方。\n"
            "4. 在给出本轮新的观点时，请尽可能：\n"
            "   - 指出哪些结论已经达成共识；\n"
            "   - 对尚有分歧的点，给出更强的论证，或者提出方案以帮助其它模型向你靠拢；\n"
            "   - 避免重复整篇重写原答案，重点写“分歧点 + 共识的收束方式”。\n"
            "5. 请直接输出你想对其他模型说的话，试图辩论、说服对方。\n"
            "6. 把篇幅更多用在：修正、说服、整合分歧上。\n"
        )
        # meta = (
        #     "本轮你将看到其它模型当前最新的答案版本。你的目标不是死守自己的原始观点，\n"
        #     "而是：在保证逻辑严谨和事实可靠的前提下，**尽可能促成所有模型在关键结论上的收敛和共识**。\n"
        #     "如果你认为某个模型的论证比你原来更严谨、更有说服力，可以明确写出你在哪些点上被说服、愿意修正立场。\n"
        #     "如果你与某个模型观点相近，而有第三个（或更多）模型观点明显有问题，你可以有意识地与观点相近的一方形成“联盟”，\n"
        #     "一起用更系统的论证去说服那一方修改观点。\n\n"
        #     "下面是其它模型当前的答案（它们都在回答同一个原始问题）：\n"
        #     "------------------------------\n"
        #     f"{others_block}\n"
        #     "请你在充分理解这些内容的前提下，完成以下任务：\n"
        #     "1. 批判性地判断你是否严格意义上完全同意其它模型当前答案的全部关键结论与推理过程。\n"
        #     "2. 尝试对不同观点进行归类：谁和你更接近？谁出现了明显错误或遗漏？\n"
        #     "3. 如果你发现自己原始判断有错误或不够严谨，请坦然承认并明确写出你被说服/更新立场的地方。\n"
        #     "4. 在给出本轮新的观点时，请尽可能：\n"
        #     "   - 指出哪些结论已经达成共识；\n"
        #     "   - 对尚有分歧的点，给出更强的论证，或者提出折中方案，以帮助其它模型向你靠拢；\n"
        #     "   - 避免重复整篇重写原答案，重点写“分歧点 + 共识的收束方式”。\n"
        #     "5. 输出时仍然需要给出你认为当前最正确、最完整的中文答案，但可以适当简化对基础背景的重复描述，\n"
        #     "   把篇幅更多用在：修正、说服、整合分歧上。\n"
        # )

        if target_model["supports_web"]:
            web_part = (
                "\n⚠️ 关于联网检索：\n"
                "  - 你具备 web search 能力，请在需要事实信息或最新资料时主动使用搜索。\n"
                "  - 如果本轮使用了 web search，请在答案末尾列出“参考链接：”并给出关键 URL；\n"
                "    如果没有使用任何外部网页，可以写“参考链接：无”。\n"
            )
        else:
            web_part = (
                "\n⚠️ 关于联网检索：\n"
                "  - 你不能主动访问互联网，但可以依赖你自身的知识和其它模型给出的参考链接。\n"
                "  - 如果你在本轮论证中参考了其它模型给出的链接，可以在答案末尾列出“参考链接：”并标注“转引自某模型”。\n"
            )

        tail = (
            "\n⚠️ 输出格式要求：\n"
            "  - 你必须返回一个 Python 列表，长度为 2：\n"
            "    第一个元素是布尔值 True 或 False，表示你是否认为“目前所有在线并成功返回的模型的答案已经在关键结论上足够一致，可以视为达成共识”。\n"
            "    第二个元素是你的中文答案字符串（必须是简体中文，不要使用英文）。\n"
        )

        return header + meta + web_part + tail

    def build_final_prompt(target_model):
        name = target_model["name"]
        supports_web = target_model["supports_web"]

        head = (
            f"现在，你和其它在线模型已经就该问题达成了实质性的共识。\n"
            f"你是 {name}，接下来请你面向用户，给出一份“最终长篇解读”。\n"
            "用户不关心模型之间的辩论过程，只关心最后整合后的、尽可能完整而有深度的答案。\n\n"
        )

        body = (
            "【写作要求】\n"
            "1. 使用**简体中文**，风格可以自然、有一点你的“个性”，但要保证逻辑清晰。\n"
            "2. 不要再提及“模型”“辩论”“谁说过什么”等过程性信息，\n"
            "   就像是你独立思考后给出的终极报告。\n"
            "3. 请尽量写得**详细、充实、有想象力**：\n"
            "   - 总览/结论综述：先用 1–3 段话概括核心结论和整体印象；\n"
            "   - 分章节展开：按若干维度拆解问题，每一部分都要有清楚的推理和例子；\n"
            "   - 风险与局限：指出哪些地方存在不确定性、容易被误读；\n"
            "   - 总结与行动建议：从宏观上再收束一次，并给出下一步可以如何理解/行动。\n"
            "4. 字数上不要吝啬，只要信息是有用的、推理是有价值的，可以写得很长（例如 3000 字以上）。\n"
            "5. 如果你在之前轮次中使用过 web search 或参考过链接，请在正文中自然吸收这些信息，\n"
            "   并在答案末尾列出“参考链接：”部分，逐条给出你认为关键的 URL；\n"
            "   如果没有使用任何外部网页，可以写“参考链接：无”。\n\n"
            "请直接输出这一份长篇最终答案的完整内容。\n"
        )

        if not supports_web:
            extra = (
                "你不能主动访问互联网，但可以引用你在之前轮次中从其它模型看到的链接或外部信息，\n"
                "如果有参考这些内容，请在“参考链接：”部分注明“转引自其它模型：URL”。\n\n"
            )
            return head + extra + body
        else:
            return head + body

    # ================== 正式开始 ==================

    log = []
    prompt_text = " ".join(str(p) for p in prompt)
    log.append(
        "=== Initial User Prompt ===\n"
        f"{prompt_text}\n"
        f"(timestamp: {datetime.now().isoformat()})"
    )

    # 初始化模型状态（不再有永久 active，仅有 temporarily_down）
    models = [
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
        #     "supports_web": False,  # 你的接口里也没有 web search
        #     "temporarily_down": False,
        #     "last_answer": "",
        #     "last_struct": "",
        #     "last_agree": False,
        # },
    ]

    loop_idx = 0

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # ===== 首轮：所有模型各自回答原问题（并行） =====
        print("=== Initial Round: All models answer the original prompt ===")
        init_futs = {}
        for m in models:
            if m["supports_web"]:
                init_futs[m["id"]] = executor.submit(
                    m["interface"].ask, prompt, True
                )
            else:
                init_futs[m["id"]] = executor.submit(
                    m["interface"].ask, prompt
                )

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
                m["last_answer"] = f"[{m['name']} 在首轮回答中发生错误: {repr(e)}]"
                log.append(
                    f"=== Initial {m['name']} Error ===\n"
                    f"Error: {repr(e)}\n"
                    f"该模型在首轮未能成功返回，将被视为本轮掉线，但在后续轮次仍会尝试重新加入。\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )
        
        print("=== Initial Round Complete. Starting Debate Loops ===")
        # ===== 主循环：多轮辩论 =====
        while loop_idx < max_loops:
            loop_idx += 1
            print(f"\n=== Loop {loop_idx} ===")

            # 本轮参与者：上一轮没有 temporarily_down 的模型
            participants = [m for m in models if not m["temporarily_down"]]

            if len(participants) < 2:
                log.append(
                    f"=== Loop {loop_idx}: Not enough participants (n={len(participants)}), stop debating. ===\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )
                break

            prompts = {}
            for m in participants:
                prompts[m["id"]] = build_debate_prompt(m, participants)

            futs = {}
            for m in participants:
                mid = m["id"]
                if m["supports_web"]:
                    futs[mid] = executor.submit(
                        m["interface"].ask, [prompts[mid]], True
                    )
                else:
                    futs[mid] = executor.submit(
                        m["interface"].ask, [prompts[mid]]
                    )

            # 默认下一轮大家都“有资格”参与；这一轮报错的再标记 temporarily_down=True
            for m in models:
                m["temporarily_down"] = False

            for m in participants:
                mid = m["id"]
                try:
                    raw = futs[mid].result()
                    m["last_struct"] = raw
                    agree, answer = parse_list_response(raw)
                    m["last_agree"] = agree
                    m["last_answer"] = answer

                    print(f"{m['name']} parsed:", agree, "(answer length:", len(answer), ")")
                    log.append(
                        f"=== Loop {loop_idx}: {m['name']} Evaluation ===\n"
                        f"Prompt to {m['name']}:\n{prompts[mid]}\n\n"
                        f"Raw output:\n{raw}\n\n"
                        f"Parsed -> agree: {agree}, answer length: {len(answer)}\n"
                        f"(timestamp: {datetime.now().isoformat()})"
                    )

                except Exception as e:
                    m["temporarily_down"] = True
                    m["last_struct"] = f"[{m['name']} 在第 {loop_idx} 轮中调用失败: {repr(e)}]"
                    msg = f"[{m['name']} 在第 {loop_idx} 轮中调用失败: {repr(e)}]"
                    print(msg)
                    log.append(
                        f"=== Loop {loop_idx}: {m['name']} Error ===\n"
                        f"Prompt to {m['name']}:\n{prompts[mid]}\n\n"
                        f"Error: {repr(e)}\n"
                        f"该模型在本轮被视为掉线，但下一轮仍会尝试重新加入。\n"
                        f"(timestamp: {datetime.now().isoformat()})"
                    )

            successful_models = [
                m for m in models
                if not m["temporarily_down"] and m["last_struct"]
            ]
            if successful_models and all(m["last_agree"] for m in successful_models):
                print("\n✅ 本轮所有成功返回的模型 agree=True，认为已达成共识。")
                log.append(
                    "=== Final Agreement (All successful models True) ===\n"
                    + "\n".join(
                        [
                            f"{m['name']} agree flag: {m['last_agree']}, answer length: {len(m['last_answer'])}"
                            for m in successful_models
                        ]
                    )
                    + f"\n(timestamp: {datetime.now().isoformat()})"
                )
                break

        # ===== 结束：让所有“最近曾经成功返回过”的模型写最终长篇 =====
        final_futs = {}
        for m in models:
            if not m["last_answer"] and not m["last_struct"]:
                continue
            final_prompt = [build_final_prompt(m)]
            if m["supports_web"]:
                final_futs[m["id"]] = executor.submit(
                    m["interface"].ask, final_prompt, True
                )
            else:
                final_futs[m["id"]] = executor.submit(
                    m["interface"].ask, final_prompt
                )

        final_answers_map = {}
        for m in models:
            mid = m["id"]
            if mid not in final_futs:
                continue
            try:
                ans = final_futs[mid].result()
                final_answers_map[mid] = ans
                log.append(
                    f"=== Final Long Answer from {m['name']} ===\n"
                    f"{ans}\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )
            except Exception as e:
                final_answers_map[mid] = f"[{m['name']} 在最终长篇回答阶段出错: {repr(e)}]"
                log.append(
                    f"=== Final Long Answer Error from {m['name']} ===\n"
                    f"Error: {repr(e)}\n"
                    f"(timestamp: {datetime.now().isoformat()})"
                )

    # ===== 收尾：汇总结果 =====

    def pick_final(mid):
        m = get_model(models, mid)
        if m is None:
            return f"[{mid} 模型不存在]"
        if mid in final_answers_map:
            return final_answers_map[mid]
        if m["last_answer"]:
            return m["last_answer"]
        if m["last_struct"]:
            return m["last_struct"]
        return f"[{m['name']} 未能给出有效答案]"

    gemini_final = pick_final("gemini")
    gpt_final = pick_final("chatgpt")
    deepseek_final = pick_final("deepseek")
    # claude_final = pick_final("claude")
    # qwen_final = pick_final("qwen")

    log.append(
        "=== Final Summary ===\n"
        + "\n".join(
            [
                f"{m['name']} temporarily_down (last loop): {m['temporarily_down']}, "
                f"has_last_answer: {bool(m['last_answer'])}"
                for m in models
            ]
        )
        + f"\n(timestamp: {datetime.now().isoformat()})"
    )

    write_log_to_file(log, log_filename)
    write_final_answers_to_file(
        gemini_final,
        gpt_final,
        deepseek_final,
        # claude_final,
        # qwen_final,
        filename=final_answers_filename,
    )
    print(f"\n📝 Dialog log exported to {log_filename}")
    print(f"📝 Final answers exported to {final_answers_filename}")

    last_gemini_struct = get_model(models, "gemini")["last_struct"]
    last_gpt_struct = get_model(models, "chatgpt")["last_struct"]
    last_deepseek_struct = get_model(models, "deepseek")["last_struct"]
    # last_claude_struct = get_model(models, "claude")["last_struct"]
    # last_qwen_struct = get_model(models, "qwen")["last_struct"]

    return (
        gemini_final,
        gpt_final,
        deepseek_final,
        # claude_final,
        # qwen_final,
        last_gemini_struct,
        last_gpt_struct,
        last_deepseek_struct,
        # last_claude_struct,
        # last_qwen_struct,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-model debate loop with optional prefix for file naming"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for input/output files (e.g., 'wzw' -> prompt_wzw.md, logs/wzw/final_answers_wzw.md, logs/wzw/dialog_log_wzw.md)"
    )
    args = parser.parse_args()
    
    # Load prompt based on prefix
    prompt = load_prompt(prefix=args.prefix)
    
    # Set output directory and filenames based on prefix
    if args.prefix:
        output_dir = f"logs/{args.prefix}"
        log_filename = f"{output_dir}/dialog_log_{args.prefix}.md"
        final_answers_filename = f"{output_dir}/final_answers_{args.prefix}.md"
    else:
        output_dir = "logs/default"
        log_filename = f"{output_dir}/dialog_log.md"
        final_answers_filename = f"{output_dir}/final_answers.md"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    (
        final_gemini,
        final_gpt,
        final_deepseek,
        # final_claude,
        # final_qwen,
        gemini_debug,
        gpt_debug,
        deepseek_debug,
        # claude_debug,
        # qwen_debug,
    ) = close_loop_ask(
        prompt,
        max_loops=30,
        log_filename=log_filename,
        final_answers_filename=final_answers_filename,
        prefix=args.prefix,
    )
    print("\n=== 最终中文答案（重答原始任务） ===")
    print("Gemini:", final_gemini)
    print("ChatGPT:", final_gpt)
    print("DeepSeek:", final_deepseek)
    # print("Claude:", final_claude)
    # print("Qwen:", final_qwen)
