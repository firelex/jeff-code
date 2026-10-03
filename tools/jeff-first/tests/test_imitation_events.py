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
    assert events("ls && cat a.py", "a.py\nprint(1)") == []
    assert events("ls && rm a.py", "a.py") == [{"type": "deleted", "path": "/app/a.py"}]


LISTING = {"type": "listing", "folder": "/app", "entries": [{"name": "a.py"}], "showsHidden": False}


def test_backfill_takes_a_later_whole_text_of_an_unchanged_file():
    later = [[{"type": "missing", "path": "/app/other"}], [{"type": "read", "path": "/app/a.py", "content": "x\n" * 10}]]
    assert backfill([LISTING], later) == [LISTING, {"type": "read", "path": "/app/a.py", "content": "x\n" * 10}]


def test_backfill_ignores_text_seen_after_a_change_or_of_unknown_files():
    changed = [[{"type": "deleted", "path": "/app/a.py"}, {"type": "read", "path": "/app/a.py"}], [{"type": "read", "path": "/app/a.py", "content": "y\n"}]]
    assert backfill([LISTING], changed) == [LISTING]
    unknown = [[{"type": "read", "path": "/app/b.py", "content": "y\n"}]]
    assert backfill([LISTING], unknown) == [LISTING]


def test_backfill_takes_a_later_size_into_the_listing():
    later = [[{"type": "listing", "folder": "/app", "entries": [{"name": "a.py", "kind": "file", "size": 40}, {"name": "new.py", "kind": "file", "size": 5}], "showsHidden": True}]]
    assert backfill([LISTING], later) == [{**LISTING, "entries": [{"name": "a.py", "size": 40}]}]
