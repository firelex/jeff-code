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


def test_heredoc_belongs_to_the_part_that_opens_it_not_the_last_part_of_the_line():
    parts = split_command("cat > /app/run.sh <<'EOF' && chmod +x /app/run.sh\necho hi\nEOF")
    assert [(part.head, part.heredoc) for part in parts] == [
        ("cat > /app/run.sh <<'EOF' __HEREDOC0__", "echo hi"),
        ("chmod +x /app/run.sh", None),
    ]
    parts = split_command("sqlite3 /app/db <<EOF | head -5\n.tables\nEOF")
    assert parts == [Part(head="sqlite3 /app/db <<EOF __HEREDOC0__", filters=("head -5",), heredoc=".tables")]


def test_heredoc_marker_inside_quotes_or_a_here_string_is_not_a_heredoc():
    parts = split_command("echo \"use <<EOF to start\" && ls\ncat a.txt")
    assert [(part.head, part.heredoc) for part in parts] == [
        ('echo "use <<EOF to start"', None),
        ("ls", None),
        ("cat a.txt", None),
    ]
    parts = split_command("python3 -c 'print(1)\n# <<EOF'\nls")
    assert [part.heredoc for part in parts] == [None, None]
    assert [part.head for part in split_command("grep x <<< \"$TEXT\"\nls")] == ['grep x <<< "$TEXT"', "ls"]


def test_two_heredocs_on_one_line_each_go_to_their_own_part():
    parts = split_command("cat > a <<A && cat > b <<B\none\nA\ntwo\nB")
    assert [(part.head, part.heredoc) for part in parts] == [
        ("cat > a <<A __HEREDOC0__", "one"),
        ("cat > b <<B __HEREDOC1__", "two"),
    ]


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


def test_each_part_records_the_operator_before_it_and_whether_control_flow_was_stripped():
    parts = split_command("echo a; ls && cat b || echo c\npwd")
    assert [part.joiner for part in parts] == ["", ";", "&&", "||", "\n"]
    assert not any(part.in_control_flow for part in parts)
    assert all(part.in_control_flow for part in split_command("if [ -f a ]; then cat a; fi; ls"))
    assert all(part.in_control_flow for part in split_command("{ ls; cat a; } > out"))
    assert not any(part.in_control_flow for part in split_command("ls -la /out 2>/dev/null || true"))
