import pytest

from imitation.events import Shell, backfill, command_events, resolve

SHELL = Shell("/app", "/root")


def events(command, output):
    return command_events(command, output, SHELL)[0]


def test_resolve():
    assert resolve("a.py", SHELL) == "/app/a.py"
    assert resolve("../etc/x", SHELL) == "/etc/x"
    assert resolve("~/notes", SHELL) == "/root/notes"
    assert resolve("*.py", SHELL) is None
    assert resolve("$HOME/x", SHELL) is None
    assert resolve("a.py", Shell(None, "/root")) is None


def test_long_listing_gives_kinds_and_sizes():
    output = (
        "total 12\n"
        "drwxr-xr-x. 2 root root 4096 Aug 26 10:46 .\n"
        "drwxr-xr-x. 1 root root   62 Aug 27 10:40 ..\n"
        "-rw-r--r--. 1 root root  120 Aug 26 10:46 a.py\n"
        "drwxr-xr-x. 2 root root 4096 Aug 26 10:46 src\n"
        "lrwxrwxrwx. 1 root root    7 Apr 22  2024 bin -> usr/bin"
    )
    assert events("ls -la", output) == [
        {
            "type": "listing",
            "folder": "/app",
            "entries": [{"name": "a.py", "kind": "file", "size": 120}, {"name": "src", "kind": "folder"}, {"name": "bin"}],
            "showsHidden": True,
        }
    ]


def test_short_listing_and_listings_that_give_nothing():
    assert events("cd /app && ls src", "a.py  b.py\nc.txt") == [
        {"type": "listing", "folder": "/app/src", "entries": [{"name": "a.py"}, {"name": "b.py"}, {"name": "c.txt"}], "showsHidden": False}
    ]
    assert events("ls | head -3", "a\nb\nc") == []
    assert events("ls -R", "x") == []
    assert events("ls /app/*.py", "/app/a.py") == []
    assert events("ls -la", "garbage line") == []


def test_cat_gives_the_whole_text_and_head_only_existence():
    assert events("cat a.py", "print(1)") == [{"type": "read", "path": "/app/a.py", "content": "print(1)\n"}]
    assert events("head -n 5 a.py", "x") == [{"type": "read", "path": "/app/a.py"}]
    # A filtered or cut-off output still shows the file exists, but not its whole text.
    assert events("cat a.py | grep x", "x") == [{"type": "read", "path": "/app/a.py"}]
    assert events("cat a.py", "line\n[... output limited to 10000 bytes; 70 interior bytes omitted ...]\nend") == [
        {"type": "read", "path": "/app/a.py"}
    ]


def test_errors_say_missing_and_commands_not_found():
    out = "cat: /app/x.py: No such file or directory\nls: cannot access 'gone': No such file or directory\nbash: rg: command not found"
    assert events("cat /app/x.py; ls gone; rg foo", out) == [
        {"type": "missing", "path": "/app/x.py"},
        {"type": "missing", "path": "/app/gone"},
        {"type": "program", "name": "rg", "found": False},
    ]
    assert events("python3 run.py", "python3: can't open file '/app/run.py': [Errno 2] No such file or directory") == [
        {"type": "missing", "path": "/app/run.py"}
    ]


def test_writes_moves_removals_and_programs():
    assert events("cat > /app/n.py <<'EOF'\nprint(2)\nEOF", "") == [{"type": "written", "path": "/app/n.py", "content": "print(2)\n"}]
    assert events("python3 gen.py > out.csv", "") == [{"type": "deleted", "path": "/app/out.csv"}, {"type": "read", "path": "/app/out.csv"}]
    assert events("sed -i 's/a/b/' a.py", "") == [{"type": "deleted", "path": "/app/a.py"}, {"type": "read", "path": "/app/a.py"}]
    assert events("rm -rf build", "") == [{"type": "deleted", "path": "/app/build"}]
    assert events("mv a.py src/", "") == [{"type": "moved", "from": "/app/a.py", "to": "/app/src/a.py"}]
    assert events("mv a.py b.py", "") == [{"type": "moved", "from": "/app/a.py", "to": "/app/b.py"}]
    assert events("which gcc rustc", "/usr/bin/gcc") == [
        {"type": "program", "name": "gcc", "found": True},
        {"type": "program", "name": "rustc", "found": False},
    ]


def test_cd_moves_the_folder_and_cd_minus_makes_it_unknown():
    found, shell = command_events("cd src && cat a.py", "x", SHELL)
    assert found == [{"type": "read", "path": "/app/src/a.py", "content": "x\n"}]
    assert shell.cwd == "/app/src"
    found, shell = command_events("cd - && rm a.py", "", SHELL)
    assert found == [] and shell.cwd is None


def test_mixed_commands_give_only_self_naming_evidence():
    assert events("ls && cat a.py", "a.py\nprint(1)") == [{"type": "read", "path": "/app/a.py"}]
    assert events("ls && rm a.py", "a.py") == [{"type": "deleted", "path": "/app/a.py"}]


LISTING = {"type": "listing", "folder": "/app", "entries": [{"name": "a.py"}], "showsHidden": False}


def looking(*steps: list[dict]) -> list[tuple[bool, list[dict]]]:
    """Later commands that only gather information, with their events."""
    return [(False, step) for step in steps]


def test_backfill_adds_what_later_information_commands_showed_before_any_action():
    # Nothing acted between the point and these commands, so what they showed was already true at the point.
    later = looking([{"type": "missing", "path": "/app/other"}], [{"type": "read", "path": "/app/b.py", "content": "x\n" * 10}])
    assert backfill([LISTING], later) == [
        LISTING,
        {"type": "missing", "path": "/app/other"},
        {"type": "read", "path": "/app/b.py", "content": "x\n" * 10},
    ]
    listing = {"type": "listing", "folder": "/output", "entries": [], "showsHidden": True}
    assert backfill([], looking([listing])) == [listing]


def test_backfill_stops_at_any_acting_command_even_one_that_gives_no_events():
    # `cp other.py a.py` acts but is not recognised as a write: nothing seen after it may be back-filled.
    later = [(True, []), (False, [{"type": "read", "path": "/app/a.py", "content": "y\n"}])]
    assert backfill([LISTING], later) == [LISTING]
    removed = [(True, [{"type": "deleted", "path": "/app/a.py"}]), (False, [{"type": "read", "path": "/app/a.py", "content": "y\n"}])]
    assert backfill([LISTING], removed) == [LISTING]


def test_backfill_keeps_only_the_existence_of_a_text_whose_length_differs_from_a_known_size():
    sized = {"type": "listing", "folder": "/app", "entries": [{"name": "a.py", "kind": "file", "size": 40}], "showsHidden": False}
    assert backfill([sized], looking([{"type": "read", "path": "/app/a.py", "content": "x\n"}])) == [sized, {"type": "read", "path": "/app/a.py"}]
    later_size = {"type": "listing", "folder": "/app", "entries": [{"name": "a.py", "kind": "file", "size": 40}], "showsHidden": True}
    later = looking([{"type": "read", "path": "/app/a.py", "content": "\tx\n"}], [later_size])
    assert backfill([LISTING], later) == [LISTING, {"type": "read", "path": "/app/a.py"}, later_size]
    fits = {"type": "read", "path": "/app/a.py", "content": "y" * 39 + "\n"}
    assert backfill([sized], looking([fits])) == [sized, fits]
def test_find_output_lines_of_a_file_search_are_files():
    output = "/app/a.py\n/app/src/b.py"
    assert events("find /app -type f -name '*.py'", output) == [
        {"type": "read", "path": "/app/a.py"},
        {"type": "read", "path": "/app/src/b.py"},
    ]
    assert events("find . -name '*.py' -type f 2>/dev/null | head -20", "./a.py\n./src/b.py") == [
        {"type": "read", "path": "/app/a.py"},
        {"type": "read", "path": "/app/src/b.py"},
    ]
    assert events("find /app /lib -type f | sort", "/app/a.py\n/lib/c.so") == [
        {"type": "read", "path": "/app/a.py"},
        {"type": "read", "path": "/lib/c.so"},
    ]
    assert events("find /app -type f", "find: '/app/secret': Permission denied\n/app/a.py") == [{"type": "read", "path": "/app/a.py"}]


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ("find /app -type f -o -type l", "/app/a.py"),
        ("find /app -name '*.py'", "/app/a.py"),
        ("find /app -type d", "/app/src"),
        ("find /app -type f -printf '%s %p\\n'", "12 /app/a.py"),
        ("find /app -type f | wc -l", "3"),
        ("find /app -type f | sed 's/a/b/'", "/bpp/a.py"),
        ("find /app -type f", "/app/a.py\nsomething else"),
        ("find /app -type f", "/app/" + "x" * 160),
        ("find /app -type f | head", "/app/a.py\n[... output limited to 10000 bytes; 70 interior bytes omitted ...]"),
    ],
)
def test_find_output_that_does_not_prove_files_gives_nothing(command, output):
    assert events(command, output) == []


def test_a_read_in_a_command_of_several_parts_proves_the_file_when_no_error_names_it():
    assert events("ls -la && cat a.py", "a.py\nprint(1)") == [{"type": "read", "path": "/app/a.py"}]
    assert events("echo '--- a ---'; head -n 5 a.py; echo; sed -n '1,3p' b.py", "x") == [
        {"type": "read", "path": "/app/a.py"},
        {"type": "read", "path": "/app/b.py"},
    ]
    assert events("ls; cat a.py", "cat: a.py: No such file or directory") == [{"type": "missing", "path": "/app/a.py"}]
    assert events("ls; cat a.py 2>/dev/null", "") == []
    assert events("ls; cat a.py 2>&1 | grep x", "") == []
    assert events("ls; cat a.py", "x\n[... output limited to 10000 bytes; 70 interior bytes omitted ...]") == []


LONG_OUTPUT = "total 4\n-rw-r--r--. 1 root root 12 Aug 26 10:46 a.py"
LONG_ENTRIES = [{"name": "a.py", "kind": "file", "size": 12}]


def test_literal_echo_lines_around_one_information_part_are_set_aside():
    found = events("echo '=== files ===' && ls -l", "=== files ===\n" + LONG_OUTPUT)
    assert found == [{"type": "listing", "folder": "/app", "entries": LONG_ENTRIES, "showsHidden": False}]
    found = events("ls -l 2>/dev/null || echo 'no such folder'", LONG_OUTPUT)
    assert found == [{"type": "listing", "folder": "/app", "entries": LONG_ENTRIES, "showsHidden": False}]
    # The fallback message ran, so the listing failed: nothing about the folder.
    assert events("ls -l missing 2>/dev/null || echo 'no such folder'", "no such folder") == []
    assert events("printf '== a ==\\n'; pwd; ls", "== a ==\n/app\na.py  b.py") == [
        {"type": "listing", "folder": "/app", "entries": [{"name": "a.py"}, {"name": "b.py"}], "showsHidden": False}
    ]


def test_literal_separators_attribute_the_output_of_several_information_parts():
    command = "echo '--- list ---'; ls -l; echo '--- text ---'; cat a.py"
    output = "--- list ---\n" + LONG_OUTPUT + "\n--- text ---\nprint(1)"
    assert events(command, output) == [
        {"type": "listing", "folder": "/app", "entries": LONG_ENTRIES, "showsHidden": False},
        {"type": "read", "path": "/app/a.py", "content": "print(1)\n"},
    ]


@pytest.mark.parametrize(
    ("command", "output"),
    [
        # Two information parts with no separator between them: their outputs cannot be told apart.
        ("ls -l; cat a.py", LONG_OUTPUT + "\nprint(1)"),
        # A separator that also occurs inside a later output is ambiguous.
        ("echo '---'; ls; echo '---'; cat a.py", "---\na.py\n---\nx\n---\ny"),
        # Output that is not computable from the text: a variable, a printing command outside the table.
        ('echo "$NAME"; ls', "jeff\na.py"),
        ("apt-get update; ls", "Reading package lists... Done\na.py"),
        ("echo -n 'x'; ls", "xa.py"),
    ],
)
def test_output_that_cannot_be_attributed_gives_no_listing_or_text(command, output):
    assert [event for event in events(command, output) if event["type"] == "listing" or "content" in event] == []
