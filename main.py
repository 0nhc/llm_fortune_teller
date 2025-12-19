from close_loop import close_loop_ask
from bazi import BaZiAutomation
import argparse


def generate_bazi_analysis(year, month, day, hour, minute, gender_str, name):
    automator = BaZiAutomation()
    profile = automator.generate_prompt(year, month, day, hour, minute, gender_str)
    prompt = ""
    # open .txt as string
    with open("prompt_template/prompt_1.txt", "r", encoding="utf-8") as f:
        prompt_1 = f.read()
    with open("prompt_template/prompt_2.txt", "r", encoding="utf-8") as f:
        prompt_2 = f.read()
    prompt += prompt_1
    prompt += profile
    prompt += prompt_2

    # Set output directory and filenames based on prefix
    if name:
        output_dir = f"logs/{name}"
        log_filename = f"{output_dir}/dialog_log_{name}.txt"
        final_answers_filename = f"{output_dir}/final_answers_{name}.md"
    else:
        output_dir = "logs/default"
        log_filename = f"{output_dir}/dialog_log.md"
        final_answers_filename = f"{output_dir}/final_answers.md"

    response = close_loop_ask(prompt,
                              max_loops=10,
                              log_filename=log_filename,
                              final_answers_filename=final_answers_filename,
                              prefix=name)

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model debate loop with optional prefix for file naming"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the user profile for input/output files (e.g., 'hzx' -> prompt_hzx.txt, logs/hzx/final_answers_hzx.md, logs/hzx/dialog_log_hzx.txt)"
    )
    parser.add_argument("--year", type=int, required=True, help="Year of birth")
    parser.add_argument("--month", type=int, required=True, help="Month of birth")
    parser.add_argument("--day", type=int, required=True, help="Day of birth")
    parser.add_argument("--hour", type=int, required=True, help="Hour of birth")
    parser.add_argument("--minute", type=int, required=True, help="Minute of birth")
    parser.add_argument("--gender", type=str, required=True, help="Gender (e.g., 'male' or 'female')")
    args = parser.parse_args()

    generate_bazi_analysis(args.year, args.month, args.day, args.hour, args.minute, args.gender, args.name)

if __name__ == "__main__":
    main()