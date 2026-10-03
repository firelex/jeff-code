from imitation.splitter import Part, first_word, has_file_write, split_command


def heads(command: str) -> list[str]:
    return [part.head for part in split_command(command)]


def test_splits_at_and_or_semicolon_and_newline():
    assert heads("cd /app && ls -la; cat a.py || echo missing\npwd") == [
        "cd /app",
        "ls -la",
        "cat a.py",
        "echo missing",
        "pwd",
    ]


def test_pipe_keeps_the_left_most_stage_and_the_rest_as_filters():
    parts = split_command("grep -rn 'def main' /app | head -20 | sort")
    assert parts == [Part(head="grep -rn 'def main' /app", filters=("head -20", "sort"))]


def test_background_ampersand_ends_a_part_but_redirections_do_not():
    assert heads("python3 -m http.server 8000 > /tmp/log 2>&1 & sleep 1") == [
        "python3 -m http.server 8000 > /tmp/log 2>&1 &",
        "sleep 1",
    ]


def test_quotes_and_command_substitution_are_not_split():
    assert heads("echo \"a; b && c | d\" && X=$(ls /app | wc -l); echo 'e || f'") == [
        'echo "a; b && c | d"',
        "X=$(ls /app | wc -l)",
        "echo 'e || f'",
    ]


def test_heredoc_body_is_kept_out_of_the_parts():
    command = "cat > /app/fix.py <<'EOF'\nimport sys; print(1)\nx = a && b\nEOF\npython3 /app/fix.py"
    parts = split_command(command)
    assert [part.head for part in parts] == ["cat > /app/fix.py <<'EOF' __HEREDOC0__", "python3 /app/fix.py"]
    assert parts[0].heredoc == "import sys; print(1)\nx = a && b"


def test_control_flow_keywords_are_stripped_and_conditions_kept():
    assert heads("if [ -f a ]; then cat a; fi") == ["cat a"]
    assert heads("for f in *.py; do wc -l $f; done") == ["wc -l $f"]
    assert heads("if grep -q foo a.txt; then echo yes; fi") == ["grep -q foo a.txt", "echo yes"]


def test_comments_and_line_continuations():
    assert heads("# look around\nls -la \\\n  /app") == ["ls -la    /app"]


def test_python_heredoc_from_a_real_terminus_turn():
    command = (
        'FILE=$(find /app -path \'*/CQuery/QueryInfo.cs\' -print -quit)\n'
        'if [ -z "$FILE" ]; then FILE=$(find /app -name QueryInfo.cs -print -quit); fi\n'
        'python3 - "$FILE" <<\'PY\'\nimport sys\nif len(sys.argv) < 2:\n    sys.exit(1)\nPY\n'
    )
    assert heads(command) == [
        "FILE=$(find /app -path '*/CQuery/QueryInfo.cs' -print -quit)",
        "FILE=$(find /app -name QueryInfo.cs -print -quit)",
        "python3 - \"$FILE\" <<'PY' __HEREDOC0__",
    ]


def test_first_word_skips_sudo_timeout_and_assignments():
    assert first_word("sudo timeout 10 FOO=1 python3 x.py")[0] == "python3"
    assert first_word("env A=1 B=2 node app.js")[0] == "node"


def test_has_file_write():
    assert has_file_write("echo hi > out.txt")
    assert has_file_write("cat a >> b")
    assert not has_file_write("ls 2>/dev/null")
    assert not has_file_write("make > /dev/null 2>&1")
    assert not has_file_write("cmd 2>&1")
