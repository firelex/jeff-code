"""Split each bash command into parts and classify each part's intent."""
import json, re, collections, sys

HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")


def strip_heredocs(cmd):
    """Replace heredoc bodies with a marker. Returns (text, bodies)."""
    lines = cmd.split('\n')
    out = []; bodies = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEREDOC_RE.search(line)
        if m and not line.strip().startswith('#'):
            tag = m.group(2)
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != tag:
                body.append(lines[i]); i += 1
            bodies.append('\n'.join(body))
            out.append(line + f' __HEREDOC{len(bodies)-1}__')
            i += 1
            continue
        out.append(line); i += 1
    return '\n'.join(out), bodies


def split_parts(cmd):
    """Split on ; && || newline | (outside quotes/parens/braces). Returns list of pipelines (list of segments)."""
    text, bodies = strip_heredocs(cmd)
    pipelines = []; cur = []; seg = ''
    q = None; depth = 0; i = 0
    def flush_seg():
        nonlocal seg
        if seg.strip(): cur.append(seg.strip())
        seg = ''
    def flush_pipe():
        nonlocal cur
        flush_seg()
        if cur: pipelines.append(cur)
        cur = []
    while i < len(text):
        c = text[i]
        if q:
            if c == '\\' and q == '"' and i + 1 < len(text):
                seg += text[i:i+2]; i += 2; continue
            if c == q: q = None
            seg += c; i += 1; continue
        if c in ('"', "'"):
            q = c; seg += c; i += 1; continue
        if c == '\\' and i + 1 < len(text):
            if text[i+1] == '\n': i += 2; seg += ' '; continue
            seg += text[i:i+2]; i += 2; continue
        if c == '$' and text[i+1:i+2] == '(':
            depth += 1; seg += '$('; i += 2; continue
        if c == '(' and depth > 0:
            depth += 1; seg += c; i += 1; continue
        if c == ')' and depth > 0:
            depth -= 1; seg += c; i += 1; continue
        if depth == 0:
            if c == '#' and (not seg or seg[-1] in ' \t\n;'):
                # comment to end of line
                j = text.find('\n', i)
                i = len(text) if j < 0 else j
                continue
            if text.startswith('&&', i) or text.startswith('||', i):
                flush_pipe(); i += 2; continue
            if c in ';\n':
                flush_pipe(); i += 1; continue
            if c == '&' and text[i+1:i+2] != '>' and (not seg or seg[-1] != '>'):
                seg += ' &'; flush_pipe(); i += 1; continue
            if c == '|' and text[i+1:i+2] != '|':
                flush_seg(); i += 1; continue
        seg += c; i += 1
    flush_pipe()
    # strip control-flow keywords
    res = []
    for p in pipelines:
        p2 = []
        for s in p:
            s = re.sub(r'^(then|do|else|\{|\()\s+', '', s).strip()
            s = re.sub(r'^\(+', '', s).strip()
            if re.fullmatch(r'(done|fi|esac|\}|\)+|then|do|else|wait|true|:)', s): continue
            if re.match(r'^(for|while|if|elif|until)\b', s):
                # keep the condition of if/while as a part (it is a check)
                m = re.match(r'^(if|elif|while|until)\s+!?\s*(.*)$', s)
                if m and m.group(2) and not m.group(2).startswith('['):
                    s = m.group(2)
                else:
                    continue
            if s: p2.append(s)
        if p2: res.append(p2)
    return res, bodies


# ---------------- classification ----------------
INFO = 'info'; ACT = 'act'; NEUTRAL = 'neutral'; PROBE = 'compute'

def first_word(s):
    s = re.sub(r'^(sudo|time|timeout\s+\S+|env\s+(\w+=\S+\s+)*|nice|stdbuf\s+\S+)\s+', '', s)
    s = re.sub(r'^(\w+=("[^"]*"|\'[^\']*\'|\S*)\s+)+', '', s)
    m = re.match(r'\s*(\S+)', s)
    return (m.group(1) if m else ''), s

WRITE_REDIR = re.compile(r'(?<![0-9&])>{1,2}\s*(?!&|/dev/null)(\S+)')

def has_file_write(s):
    for m in WRITE_REDIR.finditer(s):
        tgt = m.group(1)
        if tgt.startswith('/dev/'): continue
        return True
    return False

DATA_EXT = r'\.(csv|tsv|json|jsonl|txt|log|fasta|fa|ttl|xml|yaml|yml|sql|sqlite|db|bpe|ckpt|pth|bin|png|jpg|pdf|parquet|npy|html|conf|cfg|ini|md|dat)\b'


def classify(seg, rest_of_pipe, bodies):
    """Return (kind, category). rest_of_pipe: the filters after this segment."""
    w, s = first_word(seg)
    base = w.split('/')[-1]
    sl = s.lower()
    hd = re.search(r'__HEREDOC(\d+)__', s)
    if base in ('cd', 'pushd', 'popd', 'export', 'set', 'source', '.', 'sleep', 'echo', 'printf', 'clear', 'true', 'unset', 'trap', 'local', 'shopt', 'alias', 'PID=$!', 'exit', 'return'):
        if base in ('echo', 'printf') and has_file_write(s):
            return ACT, 'write.file'
        if base == 'echo' and re.search(r'\$\(', s):
            return INFO, 'env.var'  # echo of computed value
        if base in ('echo', 'printf') and re.search(r'\$[A-Z_]{2,}', s) and not re.search(r'\$\?', s):
            return INFO, 'env.var'
        return NEUTRAL, 'shell'
    m_assign = re.match(r'^\w+=\$\((.*)\)\s*$', s)
    if m_assign:
        return classify(m_assign.group(1), rest_of_pipe, bodies)
    if re.match(r'^\w+=', w):
        return NEUTRAL, 'shell'
    if base in ('mount', 'pkg-config', 'postconf', 'locale', 'getent', 'ip', 'ifconfig'):
        return INFO, 'env.system'
    if base in ('kill', 'pkill', 'killall'):
        return ACT, 'process.control'
    # writing via heredoc
    if hd and base in ('cat', 'tee') and (has_file_write(s) or base == 'tee'):
        return ACT, 'write.file'
    if base in ('mkdir', 'cp', 'mv', 'rm', 'touch', 'chmod', 'chown', 'ln', 'tar', 'unzip', 'gunzip', 'zip', 'truncate', 'install', 'rmdir', 'dd', 'patch', 'git-apply'):
        if base == 'tar' and re.search(r'\s-?t[vzf]*\b', s):
            return INFO, 'file.list'
        if base == 'unzip' and ' -l' in s:
            return INFO, 'file.list'
        return ACT, 'write.file'
    if base == 'sed':
        if re.search(r'\s-i', s): return ACT, 'edit.file'
        return INFO, 'file.read'
    if base == 'perl' and re.search(r'\s-[a-z]*i', s) and '-e' in s and 'binmode' not in s and re.search(r'-p?i', s):
        return ACT, 'edit.file'
    if base in ('apt-get', 'apt', 'yum', 'dnf', 'apk', 'brew'):
        if re.search(r'\b(list|search|show|policy|info)\b', s) and not re.search(r'\binstall\b', s):
            return INFO, 'env.packages'
        return ACT, 'install'
    if base in ('apt-cache', 'dpkg', 'dpkg-query', 'rpm'):
        if base == 'dpkg' and re.search(r'\s-i\b', s): return ACT, 'install'
        return INFO, 'env.packages'
    if base in ('pip', 'pip3', 'conda', 'uv', 'poetry') or re.match(r'python[\d.]*\s+-m\s+pip', s):
        if re.search(r'\b(list|show|freeze|check|--version|-V|index|download\s+--no-deps\s+-d\s+/tmp)\b', s) and not re.search(r'\binstall\b', s):
            return INFO, 'env.packages'
        return ACT, 'install'
    if base in ('npm', 'npx', 'yarn', 'pnpm'):
        if re.search(r'\b(ls|list|view|info|search|show|outdated|-v|--version|root|config get)\b', s) and not re.search(r'\b(install|i|add|ci)\b', s):
            return INFO, 'env.packages'
        if re.search(r'\b(test|run\s+test)\b', s): return ACT, 'test'
        if re.search(r'\b(run|start|exec)\b', s) or base == 'npx': return ACT, 'run'
        return ACT, 'install'
    if base in ('cargo', 'go', 'mvn', 'gradle', 'dotnet', 'rustc', 'javac', 'tsc', 'gcc', 'g++', 'cc', 'clang', 'clang++', 'make', 'cmake', 'ninja', 'gfortran', 'cobc', 'ghc', 'ld', 'as', 'nasm', 'rustup'):
        if re.search(r'(--version|-v$|-V$|\s-dumpversion|--help|-print-search-dirs)', s) and not re.search(r'\s-o\s', s):
            return INFO, 'env.tool'
        if base in ('cargo', 'go', 'dotnet') and re.search(r'\btest\b', s): return ACT, 'test'
        if base == 'make' and re.search(r'\btest|check\b', s): return ACT, 'test'
        return ACT, 'build'
    if base in ('which', 'whereis', 'type', 'command', 'hash'):
        return INFO, 'env.tool'
    if re.search(r'\s(--version|-version|-V)\s*$', s) or re.search(r'\s--version\b', s) and len(s) < 80:
        return INFO, 'env.tool'
    if re.search(r'\s(--help|-h|-help)\b', s) or base in ('man', 'info', 'help'):
        return INFO, 'api.help'
    if base in ('uname', 'nproc', 'free', 'df', 'lscpu', 'env', 'printenv', 'id', 'whoami', 'hostname', 'pwd', 'ulimit', 'date', 'locale', 'arch', 'lsb_release', 'getconf', 'umask', 'tty', 'ldconfig', 'ldd', 'nvidia-smi', 'groups'):
        if base == 'date' and re.search(r'\s-s\b', s): return ACT, 'process.control'
        return INFO, 'env.system'
    if base in ('ps', 'pgrep', 'pidof', 'top', 'htop', 'lsof', 'netstat', 'ss', 'jobs', 'fuser', 'systemctl', 'service', 'supervisorctl', 'nc', 'netcat', 'ping', 'dig', 'nslookup', 'host'):
        if base in ('systemctl', 'service', 'supervisorctl') and not re.search(r'\b(status|is-active|list|--status-all)\b', s):
            return ACT, 'service.control'
        if base in ('nc', 'netcat') and ' -l' in s: return ACT, 'service.control'
        return INFO, 'service.check'
    if base in ('curl', 'wget', 'http'):
        if re.search(r'localhost|127\.0\.0\.1|0\.0\.0\.0|:\d{2,5}/', s) and not re.search(r'-o\s+(?!/dev/null)\S', s) and not re.search(r'-X\s*(POST|PUT|DELETE)|--data|-d\s', s):
            return INFO, 'service.check'
        if re.search(r'-[sSfLI]*I\b|--head', s): return INFO, 'net.check'
        return ACT, 'download'
    if base == 'git':
        if not re.search(r'\b(clone|checkout|pull|fetch|reset|commit|apply)\b', s) and re.search(r'\b(status|log|diff|show|branch|remote|ls-files|rev-parse|blame|describe|tag\s*$|config\s+--get|stash\s+list|reflog)\b', s):
            return INFO, 'git.state'
        return ACT, 'git.change'
    if base in ('ls', 'tree', 'du', 'stat', 'realpath', 'readlink', 'dirname', 'basename', 'test', '[', '[['):
        if base in ('test', '[', '[['):
            return INFO, 'file.exists'
        return INFO, 'file.list'
    if base in ('find', 'locate', 'fd'):
        if re.search(r'-delete|-exec\s+(rm|mv|cp|chmod|sed)', s): return ACT, 'write.file'
        if re.search(r'-exec\s+(grep|cat|head|ls|wc)', s) or 'xargs grep' in ' '.join(rest_of_pipe):
            return INFO, 'file.search'
        return INFO, 'file.find'
    if base in ('grep', 'egrep', 'fgrep', 'rg', 'ag', 'ack', 'zgrep'):
        return INFO, 'file.search'
    if base in ('cat', 'head', 'tail', 'less', 'more', 'nl', 'tac', 'zcat', 'bat', 'column', 'strings', 'cut', 'sort', 'uniq', 'awk', 'gawk', 'tr', 'fold', 'paste', 'comm', 'diff', 'cmp', 'md5sum', 'sha256sum', 'sha1sum', 'cksum', 'jq', 'yq', 'xmllint', 'iconv'):
        if has_file_write(s) and base not in ('diff',):
            return ACT, 'write.file'
        if base == 'tail' and re.search(r'\s-f\b', s): return INFO, 'logs'
        if re.search(r'/var/log|\.log\b|_logs/', s) and base in ('tail', 'cat', 'head', 'grep'):
            return INFO, 'logs'
        if base in ('diff', 'cmp', 'md5sum', 'sha256sum', 'sha1sum', 'cksum'):
            return INFO, 'file.compare'
        if base in ('cat', 'head', 'tail', 'less', 'more', 'nl', 'tac', 'zcat', 'bat') and re.search(r'\s-c\s*\d|\s-c\d', s):
            return INFO, 'data.bytes'
        if base in ('cat', 'head', 'tail', 'less', 'more', 'nl', 'tac', 'zcat', 'bat'):
            if re.search(r'node_modules|site-packages|/usr/(lib|share|include)|dist-packages|\.d\.ts|README', s):
                return INFO, 'api.source'
            return INFO, 'file.read'
        if base in ('awk', 'gawk', 'cut', 'sort', 'uniq', 'jq', 'yq', 'column', 'tr', 'strings', 'xmllint'):
            return INFO, 'data.inspect'
        return INFO, 'file.read'
    if base in ('wc',):
        return INFO, 'data.inspect'
    if base in ('od', 'xxd', 'hexdump', 'file', 'identify', 'exiftool', 'readelf', 'objdump', 'nm', 'size', 'mediainfo', 'ffprobe', 'pdfinfo'):
        return INFO, 'data.bytes'
    if base in ('pdftotext', 'tesseract', 'pdftoppm'):
        if re.search(r'\s-\s*(2>|$|\|)', s + ' ') or re.search(r'\s-$', s): return INFO, 'data.extract'
        return ACT, 'run'
    if base in ('sqlite3', 'psql', 'mysql', 'duckdb') and (re.match(r'^\s*time\s', seg) or len(s) > 350):
        return ACT, 'run'
    if base in ('sqlite3', 'psql', 'mysql', 'duckdb', 'redis-cli', 'mongo', 'mongosh'):
        if re.search(r'\.(schema|tables|indexes|indices|dbinfo|header)|PRAGMA|EXPLAIN|sqlite_master|information_schema|\\d', s, re.I) and not re.search(r'\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|VACUUM)\b', s, re.I):
            return INFO, 'data.schema'
        if '<' in s and not re.search(r'<<', s):
            return ACT, 'run'  # running a query file
        if re.search(r'\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|VACUUM|\.import|\.restore)\b', s, re.I):
            return ACT, 'write.db'
        if re.search(r'\bSELECT\b', s, re.I):
            return INFO, 'data.query'
        return INFO, 'data.query'
    if base in ('nginx', 'apache2', 'httpd', 'apachectl', 'redis-server', 'mysqld', 'postgres', 'php-fpm', 'gunicorn', 'uvicorn', 'flask', 'mailman', 'postfix', 'dovecot', 'sshd', 'cron'):
        if re.search(r'\s-t\b|\s-T\b|configtest|-V|-v$', s): return INFO, 'service.config_test'
        return ACT, 'service.control'
    if base in ('pytest', 'py.test', 'tox', 'nosetests', 'unittest') or re.search(r'-m\s+(pytest|unittest)', s):
        return ACT, 'test'
    # interpreters
    if re.match(r'(python[\d.]*|node|perl|ruby|php|java|bash|sh|nginx|sqlite3)$', base) and re.search(r'\s(-v|-V|--version|-version)\s*$', re.sub(r'\s*(2>&1|2>/dev/null)\s*', ' ', s).strip()):
        return INFO, 'env.tool'
    if re.match(r'(python[\d.]*|node|perl|ruby|php|bash|sh|Rscript|julia|lua|java|deno|bun)$', base):
        if hd or re.search(r'\s-c\s|\s-e\s|\s-\s*$|\s-\s+__|\s-\s*<', s) or re.search(r'\s-(c|e)\s*["\']', s):
            body = ''
            if hd: body = bodies[int(hd.group(1))]
            else: body = s
            # simple import check
            s_core = re.sub(r'\s*(2>&1|2>/dev/null|>/dev/null)\s*', ' ', s).strip()
            if re.search(r'^\s*(python[\d.]*)\s+-c\s+["\']\s*(import|from)\s+[\w., ]+(\s+import\s+[\w., *]+)?\s*(;\s*print\([^)]*\))?\s*;?\s*["\']\s*$', s_core):
                return INFO, 'env.packages'
            if re.search(r"node\s+-e\s+[\"'].*require\(['\"][^'\"]+['\"]\)", s) and re.search(r'Object\.keys|typeof|console\.log\(require', s) and len(s) < 200:
                return INFO, 'api.probe'
            if re.search(r'\b(inspect\.(signature|getsource)|help\(|dir\(|__doc__|__file__|__version__|\.__all__)', body) and len(body) < 400:
                return INFO, 'api.probe'
            if re.search(r'(open\(|write|\.to_csv|json\.dump|fs\.writeFileSync|os\.(remove|rename)|shutil|subprocess|torch\.save|np\.save)', body) and re.search(r"(open\([^)]*['\"][wa]b?['\"]|\.write\(|to_csv|json\.dump\(|writeFileSync|os\.remove|os\.rename|shutil\.(copy|move|rmtree)|torch\.save|np\.save)", body):
                return ACT, 'script.writes'
            if re.search(r'^\s*(from|import)\s+(run|model|solution|main|app|cli\w*|\w*_tool)\b', body, re.M) or re.search(r'subprocess\.(run|check_output|call|Popen)\(\s*\[?\s*[\'"]\./', body):
                return PROBE, 'script.own_code'
            return PROBE, 'script.inline'
        # running a script file
        m = re.match(r'\S+\s+(?:-\S+\s+)*(\S+)', s)
        if base in ('bash', 'sh') and m and m.group(1).endswith('.sh'):
            return ACT, 'run'
        if re.search(r'\b(test_\w+|tests?/)', s):
            return ACT, 'test'
        if base.startswith('python') and re.search(r'-m\s+http\.server', s):
            return ACT, 'service.control'
        if base.startswith('python') and re.match(r'python[\d.]*\s*$', s):
            return NEUTRAL, 'shell'
        return ACT, 'run'
    if w.startswith('./') or w.startswith('/app/') or w.startswith('/tmp/') or w.startswith('/usr/local/bin/'):
        if re.search(r'\s(--help|-h)\b', s): return INFO, 'api.help'
        return ACT, 'run'
    if base in ('xargs',):
        return INFO, 'data.inspect'
    if base in ('seq', 'yes', 'expr', 'bc', 'let', 'read'):
        return NEUTRAL, 'shell'
    if base in ('tee',):
        return ACT, 'write.file'
    if base in ('oligotm', 'primer3_core', 'tesseract', 'ffmpeg', 'convert', 'gs', 'qemu-system-x86_64', 'qemu', 'java', 'mono', 'cobc', 'gnucobol', 'mmseqs', 'blastn', 'samtools'):
        return ACT, 'run'
    return ACT, 'other:' + base


def classify_command(cmd):
    pipes, bodies = split_parts(cmd)
    parts = []
    for p in pipes:
        head = p[0]
        kind, cat = classify(head, p[1:], bodies)
        # if a later pipe stage writes a file (tee / > file), the pipeline acts
        for later in p[1:]:
            lw, ls_ = first_word(later)
            if lw.split('/')[-1] == 'tee' or (has_file_write(later) and not re.search(r'>\s*/dev/null', later)):
                if kind == INFO:
                    kind, cat = ACT, 'write.file'
            # `| python3 -c ...` or `| bash` executing
            if lw.split('/')[-1] in ('sh', 'bash') and kind == INFO:
                kind, cat = ACT, 'run'
        parts.append(dict(text=' | '.join(p)[:400], head=head[:300], kind=kind, cat=cat, filters=p[1:]))
    return parts


if __name__ == '__main__':
    for c in sys.argv[1:]:
        for p in classify_command(c): print(p['kind'], p['cat'], '::', p['head'][:120])
