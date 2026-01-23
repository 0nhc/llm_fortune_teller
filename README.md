# 🔮 LLM Fortune Teller (Multi-Agent Debate)

A multi-agent LLM debating system wrapped around BaZi (八字) / Four Pillars fortune reading. Inspired by evidence that debate can improve factuality and reasoning in language models (see: [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://openreview.net/pdf?id=zj7YuTE4t8)), I found the same “argue it out, then converge” loop works surprisingly well for fortune-teller style narratives too.

At its core is `close_loop.py`: a reusable debate-and-refine engine that can be adapted to other applications beyond a fortune teller.

Given a birth datetime and gender, the program computes structured BaZi signals (pillars, relations, DaYun, etc.), then uses the debate loop to produce a clearer, more consistent final report.



> ⚠️ **Disclaimer**: This project is for entertainment/learning only. Not professional advice.

---

## ✨ Features
* ✅ Multi-LLM support (DeepSeek / Qwen / Kimi) via API keys
* ✅ Automatic retry mechanism with configurable max retries for robust API calls
* ✅ Web search capability (Kimi model supports internet search during debates)
* ✅ Multi-agent debate loop that converges to consensus
* ✅ CLI-first workflow, easy to automate

---

## 📦 Installation

It’s recommended to use a virtual environment (`venv`, `conda`, `mamba`, etc.). Recommended: **Python 3.10+**.

```bash
git clone https://github.com/0nhc/llm_fortune_teller.git
cd llm_fortune_teller
pip install -r requirements.txt
```

---

## 🔑 Configure LLM API Keys

Export keys in each new terminal, or add them to your `~/.bashrc`, `~/.zshrc`, etc.

```bash
export DEEPSEEK_API_KEY="<your-deepseek-api-key>"  # https://api-docs.deepseek.com/
export QWEN_API_KEY="<your-qwen-api-key>"  # https://help.aliyun.com/zh/model-studio/get-api-key
export KIMI_API_KEY="<your-kimi-api-key>"  # https://platform.moonshot.cn/console/api-keys
```

---

## 🚀 Quick Start

```bash
python ./main.py \
  --name test \
  --year 1900 --month 1 --day 1 \
  --hour 12 --minute 00 \
  --gender male
```

---

## 🧰 Usage

### Required Arguments

> Timezone note: Please provide the birth datetime in **China Standard Time (CST, UTC+8)**.

| Flag       | Type   | Example           | Notes                                 |
| ---------- | ------ | ----------------- | ------------------------------------- |
| `--name`   | string | `John`            | Used for labeling/report              |
| `--year`   | int    | `1900`            | Birth year                            |
| `--month`  | int    | `1`               | Birth Month (`1`–`12`)                |
| `--day`    | int    | `1`               | Birth Day (`1`–`31`)                  |
| `--hour`   | int    | `12`              | Birth Hour (`0`–`23`)                |
| `--minute` | int    | `00`              | Birth Minute (`0`–`59`)               |
| `--gender` | string | `male`            | Gender (`male` or `female`)           |

### Examples

**Basic**

```bash
python ./main.py --name John --year 1900 --month 1 --day 1 --hour 15 --minute 0 --gender male
```

**Another person**

```bash
python ./main.py --name Alice --year 1998 --month 8 --day 2 --hour 00 --minute 45 --gender female
```

### Advanced Usage

The debate loop (`close_loop.py`) supports additional options:

| Flag          | Type | Default | Description                                    |
| ------------- | ---- | ------- | ---------------------------------------------- |
| `--max_loops` | int  | `10`    | Maximum debate rounds after the initial round |
| `--max_retries` | int | `3`     | Maximum retry attempts for each API call      |
| `--lang`      | str  | `zh`    | Final output language (`zh` or `en`)           |
| `--prefix`    | str  | `None`  | Prompt prefix for custom prompts               |

**Example with custom options:**

```bash
python ./close_loop.py --prefix custom --lang en --max_loops 15 --max_retries 5
```

### How It Works

1. **Initial Round**: All models independently answer the original prompt in parallel
2. **Debate Loops**: Models see each other's responses and debate until consensus is reached
   - Each model evaluates whether it agrees with others' conclusions
   - Models can challenge, reconcile, or update their stance
   - Loop continues until all successful models return `agree=True` or `max_loops` is reached
3. **Final Answers**: Each model produces a polished, standalone final answer
4. **Output**: Results are saved to `logs/<prefix>/` directory:
   - `dialog_log_<prefix>.md`: Full debate transcript
   - `final_answers_<prefix>.md`: Final answers from each model

### Retry Mechanism

The system includes automatic retry logic for API calls:
- Each API call will retry up to `max_retries` times (default: 3) on failure
- Retries help handle transient network issues, rate limits, etc.
- Models that fail after all retries are marked as temporarily down but can retry in the next round

---

## 📄 License

```
MIT License

Copyright (c) 2025 Zhengxiao Han

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## 🙏 Acknowledgements

* `lunar_python` for calendar + BaZi computations
* LLM providers: DeepSeek, Alibaba Qwen, Moonshot Kimi
