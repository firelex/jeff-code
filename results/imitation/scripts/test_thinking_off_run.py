"""Tests of the generation-loop detector in thinking_off_run.py. Run: python -m pytest test_thinking_off_run.py"""

from thinking_off_run import looping

C_FILE = "#include <stdio.h>\n\n" + "".join(
    f"int check_{i}(int x) {{\n    if (x > {i}) {{\n        return x - {i};\n    }}\n    return {i};\n}}\n\n"
    for i in range(12)
)

THINKING = """The test failed because the parser does not handle empty lines. I will look at parse_line in main.py and
check how it splits fields. After that I will rerun the tests with pytest -q to see whether the fix holds, and then
check the output file /app/out.csv for the header row and the number of rows, which should be 1,204 per the task.
"""


def test_repeated_piece_is_a_loop():
    text = THINKING + "Let me check the file again. " * 20
    assert looping(text).startswith("the last 400 characters repeat")


def test_repeated_line_is_a_loop():
    text = THINKING + "".join(f"Step {i}: wait.\nI need to check the config file again.\n" for i in range(9))
    assert "occurs" in looping(text)


def test_ordinary_thinking_is_not_a_loop():
    assert looping(THINKING * 1) is None


def test_code_with_many_short_repeated_lines_is_not_a_loop():
    # `}` and `return` lines recur many times in the last 60 lines; lines under 10 characters do not count.
    assert C_FILE.split("\n")[-60:].count("    }") >= 8
    assert looping(C_FILE) is None


def test_short_text_is_not_a_loop():
    assert looping("ab" * 50) is None
