from __future__ import annotations

import argparse
import os
from typing import Optional

from bazi import BaZiAutomation
from close_loop import close_loop_ask


def _read_text(path: str) -> str:
    """
    Read a UTF-8 text file and return its content.

    Args:
        path: Path to the text file.

    Returns:
        The file content as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def build_full_prompt(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    gender: str,
) -> str:
    """
    Build the full analysis prompt by combining:
      1) prompt_template/prompt_1.txt
      2) a generated BaZi profile block
      3) prompt_template/prompt_2.txt

    Args:
        year, month, day, hour, minute: Birth datetime (Gregorian).
        gender: "male" or "female" (passed through to BaZiAutomation).

    Returns:
        A single concatenated prompt string.
    """
    automator = BaZiAutomation()
    profile = automator.generate_prompt(year, month, day, hour, minute, gender)

    prompt_1 = _read_text("prompt_template/prompt_1.txt")
    prompt_2 = _read_text("prompt_template/prompt_2.txt")

    return f"{prompt_1}{profile}{prompt_2}"


def get_output_paths(profile_name: Optional[str]) -> tuple[str, str, str]:
    """
    Compute output directory and filenames for logs and final answers.

    Args:
        profile_name: Optional profile name used as a folder and filename suffix.

    Returns:
        (output_dir, log_filename, final_answers_filename)
    """
    if profile_name:
        output_dir = os.path.join("logs", profile_name)
        log_filename = os.path.join(output_dir, f"dialog_log_{profile_name}.txt")
        final_answers_filename = os.path.join(output_dir, f"final_answers_{profile_name}.md")
    else:
        output_dir = os.path.join("logs", "default")
        log_filename = os.path.join(output_dir, "dialog_log.txt")
        final_answers_filename = os.path.join(output_dir, "final_answers.md")

    return output_dir, log_filename, final_answers_filename


def generate_bazi_analysis(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    gender: str,
    name: Optional[str] = None,
    max_loops: int = 10,
) -> None:
    """
    Generate a BaZi report using the multi-model debate loop.

    Args:
        year, month, day, hour, minute: Birth datetime (Gregorian).
        gender: "male" or "female".
        name: Optional profile name used for output folder and filenames.
        max_loops: Maximum debate loops.
    """
    full_prompt = build_full_prompt(year, month, day, hour, minute, gender)

    output_dir, log_filename, final_answers_filename = get_output_paths(name)
    _ensure_dir(output_dir)

    close_loop_ask(
        [full_prompt] if not isinstance(full_prompt, list) else full_prompt,
        max_loops=max_loops,
        log_filename=log_filename,
        final_answers_filename=final_answers_filename,
        output_lang="zh",
    )


def main() -> None:
    """CLI entrypoint for generating a BaZi report."""
    parser = argparse.ArgumentParser(
        description="Generate a BaZi/Four Pillars report using a multi-model debate loop."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Profile name for output files. Example: 'hzx' -> logs/hzx/ "
            "dialog_log_hzx.txt and final_answers_hzx.md"
        ),
    )
    parser.add_argument("--year", type=int, required=True, help="Birth year (Gregorian)")
    parser.add_argument("--month", type=int, required=True, help="Birth month (1-12)")
    parser.add_argument("--day", type=int, required=True, help="Birth day (1-31)")
    parser.add_argument("--hour", type=int, required=True, help="Birth hour (0-23)")
    parser.add_argument("--minute", type=int, required=True, help="Birth minute (0-59)")
    parser.add_argument(
        "--gender",
        type=str,
        required=True,
        choices=["male", "female"],
        help="Gender ('male' or 'female')",
    )
    parser.add_argument(
        "--max_loops",
        type=int,
        default=10,
        help="Maximum debate loops after the initial round.",
    )
    args = parser.parse_args()

    generate_bazi_analysis(
        year=args.year,
        month=args.month,
        day=args.day,
        hour=args.hour,
        minute=args.minute,
        gender=args.gender,
        name=args.name,
        max_loops=args.max_loops,
    )


if __name__ == "__main__":
    main()
