"""Estimate which information-gathering parts the scout could have pre-empted.

For every Qwen turn we rebuild what the scout could see before that turn:
  task text, all earlier tool calls (Qwen and scout) and their outputs,
  files Qwen wrote/edited/created via bash.
Then each information part is matched against the option lists the current
scout tools and the proposed new tools would build.
"""
import json, re, sys, collections, os
sys.path.insert(0, '/private/tmp/claude-501/qwen-mining/scripts')
from classify import classify_command, INFO, ACT, NEUTRAL, PROBE, first_word

S = json.load(open('/private/tmp/claude-501/qwen-mining/sessions.json'))
RECENT = 6  # how many earlier assistant turns count as "recent outputs"

PATH_RE = re.compile(r'(?:(?<=[\s\'"=(:,])|^)((?:/|\./|~/)?[\w.@+-]+(?:/[\w.@+-]+)*\.[A-Za-z0-9]{1,8}|/(?:[\w.@+-]+/)*[\w.@+-]+)')
WORD_RE = re.compile(r'[A-Za-z_][\w.+-]{1,}')

DATA_EXT = {'csv', 'tsv', 'json', 'jsonl', 'txt', 'log', 'fasta', 'fa', 'ttl', 'xml', 'yaml', 'yml', 'sql', 'sqlite', 'db', 'bpe', 'ckpt', 'pth', 'pt', 'bin', 'png', 'jpg', 'jpeg', 'pdf', 'parquet', 'npy', 'npz', 'dat', 'gz', 'zip', 'tar', 'h5', 'pkl', 'wav', 'mp3', 'mp4', 'cbl', 'conf'}
EXT_TOOLS = {  # file type -> tools the toolchain probe adds
    'pdf': ['pdftotext', 'pdfinfo'], 'jpg': ['tesseract', 'convert', 'identify'], 'jpeg': ['tesseract', 'convert'], 'png': ['tesseract', 'convert', 'identify'],
    'ttl': ['rdflib', 'arq', 'sparql', 'riot', 'java'], 'sparql': ['arq', 'rdflib'], 'db': ['sqlite3'], 'sqlite': ['sqlite3'], 'sql': ['sqlite3', 'psql'],
    'cpp': ['g++', 'gcc', 'cc', 'clang', 'make', 'cmake'], 'c': ['gcc', 'cc', 'clang', 'make'], 'h': ['gcc', 'cc'], 'py': ['python3', 'python', 'pip', 'pip3'],
    'pth': ['torch', 'numpy'], 'pt': ['torch'], 'ckpt': ['tensorflow', 'torch', 'numpy'], 'js': ['node', 'npm'], 'ts': ['node', 'npm', 'tsc'],
    'fasta': ['python3', 'perl', 'oligotm', 'primer3'], 'cbl': ['cobc'], 'java': ['java', 'javac'], 'go': ['go'], 'rs': ['cargo', 'rustc'], 'conf': ['nginx'],
}
BASE_TOOLS = {'python3', 'python', 'pip', 'pip3', 'node', 'npm', 'gcc', 'g++', 'cc', 'clang', 'make', 'cmake', 'perl', 'java', 'curl', 'wget', 'git', 'sqlite3',
              'apt', 'apt-get', 'dnf', 'yum', 'apk', 'sys', 'gdb', 'strace', 'objdump', 'systemctl', 'service', 'od', 'xxd', 'file', 'tcc', 'ruby', 'go', 'cargo', 'numpy', 'pandas', 'PIL', 'pillow', 'torch', 'awk', 'jq'}


def norm_path(p, cwd='/app'):
    p = p.strip('\'"')
    if p.startswith('./'): p = p[2:]
    if not p.startswith('/') and not p.startswith('~'):
        p = cwd.rstrip('/') + '/' + p
    return os.path.normpath(p)


class State:
    def __init__(self, task_text):
        self.task = task_text or ''
        self.task_words = set(w.lower() for w in WORD_RE.findall(self.task))
        self.task_paths = set(PATH_RE.findall(self.task))
        self.history = []  # list of (text_of_commands_and_outputs)
        self.written = set()
        self.installed = set()
        self.cwd = '/app'
        self.seen_dirs = set()

    def recent_text(self, n=RECENT):
        return '\n'.join(self.history[-n:])

    def all_text(self):
        return '\n'.join(self.history)

    def named(self, path, recent=True):
        """Is the file named in the task, in recent outputs/commands, or written by Qwen?"""
        base = os.path.basename(path)
        full = norm_path(path, self.cwd)
        if full in self.written: return 'written'
        if base and (base in self.task or full in self.task): return 'task'
        txt = self.recent_text() if recent else self.all_text()
        if base and base in txt: return 'recent'
        return None

    def add_turn(self, calls):
        chunks = []
        for c in calls:
            a = c.get('args') or {}
            if c['name'] == 'bash':
                cmd = a.get('command', '')
                chunks.append(cmd)
                for m in re.finditer(r'(?:cat|tee)\s+>{1,2}\s*(\S+)\s*<<', cmd): self.written.add(norm_path(m.group(1), self.cwd))
                for m in re.finditer(r'(?<![<>0-9&])>{1,2}\s*([\w./-]+\.\w+)', cmd):
                    if not m.group(1).startswith('/dev/'): self.written.add(norm_path(m.group(1), self.cwd))
                for m in re.finditer(r'(?:pip3?|npm|apt-get|apt)\s+(?:install|add)\s+([^;&|\n]+)', cmd):
                    for tok in m.group(1).split():
                        if not tok.startswith('-'): self.installed.add(tok.split('==')[0].split('@')[0] if not tok.startswith('@') else '@' + tok[1:].split('@')[0])
                m = re.match(r'\s*cd\s+(\S+)', cmd)
                if m: self.cwd = norm_path(m.group(1), self.cwd)
            elif c['name'] in ('write', 'edit'):
                self.written.add(norm_path(a.get('path', ''), self.cwd))
                chunks.append(a.get('path', ''))
            else:
                chunks.append(json.dumps(a))
            chunks.append((c.get('result') or '')[:20000])
        self.history.append('\n'.join(chunks))


def targets_of(part_text):
    """File-ish targets of a command part (skip options)."""
    toks = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', part_text)
    out = []
    for t in toks[1:]:
        if t.startswith('-') or t.startswith('2>') or t in ('|', '>', '<'): continue
        t2 = t.strip('\'"')
        if re.search(r'[/.]', t2) and not re.match(r'^\d+(,\d+)?p?$', t2) and not re.match(r'^[\d.]+$', t2):
            out.append(t2)
    return out


def match_part(p, st, call):
    """Return (covered_by_current, covered_by_new, new_tool_name, note)."""
    cat = p['cat']; text = p['head']
    task_l = st.task.lower(); recent = st.recent_text(); recent_l = recent.lower()
    # ----- pi tools
    if cat == 'pi.read' or cat == 'file.read' or cat == 'api.source':
        paths = [call['args'].get('path')] if cat == 'pi.read' else targets_of(text)
        paths = [x for x in paths if x]
        if not paths: return False, False, 'Read', 'no-path'
        ok = [st.named(x) for x in paths]
        if all(ok):
            if cat == 'api.source':
                return True, True, 'Read', 'named'
            return True, True, 'Read', 'named'
        # new: "Peek data file" covers data files that exist in listings seen anywhere
        if all(st.named(x, recent=False) for x in paths):
            return False, True, 'Read (wider memory)', 'named-earlier'
        if cat == 'api.source' or any('node_modules' in x or 'site-packages' in x for x in paths):
            pk = [x for x in paths if any(i.strip('@') in x for i in st.installed if len(i) > 2)]
            if pk: return False, True, 'Package docs', 'installed-pkg'
        if any(x.startswith('/etc/') for x in paths):
            return False, True, 'System config', 'etc'
        if len(paths) > 1 and sum(bool(o) for o in ok) >= 1:
            return False, False, 'Read', 'bundle-partial'
        return False, False, 'Read', 'unnamed'
    if cat in ('pi.ls', 'file.list', 'file.exists'):
        paths = [call['args'].get('path', '.')] if cat == 'pi.ls' else (targets_of(text) or ['.'])
        if re.search(r'/usr/(local/)?bin|/bin\b', text):
            return False, True, 'Toolchain check', 'bin-listing'
        res = []
        for x in paths:
            full = norm_path(x.rstrip('*'), st.cwd)
            if full in ('/app', st.cwd) or x in ('.', './'):
                res.append(True); continue
            d = full if not re.search(r'\.\w{1,6}$', full) else os.path.dirname(full)
            if d in ('/app', st.cwd): res.append(True); continue
            res.append(bool(st.named(os.path.basename(d.rstrip('/'))) or st.named(x)))
        if all(res): return True, True, 'List', 'cwd-or-named'
        if any('node_modules' in x or 'site-packages' in x for x in paths):
            return False, True, 'Package docs', 'pkg-dir'
        if any(x.startswith('/etc/') or x.startswith('/var/') for x in paths):
            return False, True, 'Service check' if '/var/log' in text else 'System config', 'sys-dir'
        return False, False, 'List', 'unnamed-dir'
    if cat in ('pi.grep', 'file.search'):
        if cat == 'pi.grep':
            pats = [call['args'].get('pattern', '')]
        else:
            m = re.search(r'(?:grep|rg)\s+(?:-\S+\s+)*("[^"]*"|\'[^\']*\'|\S+)', text)
            pats = [m.group(1).strip('\'"')] if m else []
        alts = [a for pat in pats for a in re.split(r'\\\||\|', pat) if a]
        words = [w for a in alts for w in WORD_RE.findall(a) if len(w) > 2]
        if words and all(w.lower() in task_l or w.lower() in recent_l for w in words):
            # current Search builds names from task text and error messages only
            if all(w.lower() in task_l for w in words):
                return True, True, 'Search', 'task-names'
            return False, True, 'Search (names from recent output)', 'recent-names'
        if re.search(r'/var/log|\.log\b', text):
            return False, True, 'Service check', 'log-grep'
        return False, False, 'Search', 'novel-pattern'
    if cat in ('pi.find', 'file.find'):
        words = [w for w in WORD_RE.findall(text.replace('find', '')) if len(w) > 2 and w not in ('name', 'type', 'dev', 'null', 'head', 'path', 'proc', 'iname')]
        if words and any(w.lower().strip('*') in task_l for w in words):
            return True, True, 'Find', 'task-name'
        if words and any(w.lower().strip('*') in recent_l for w in words):
            return False, True, 'Find (names from recent output)', 'recent-name'
        return False, False, 'Find', 'novel'
    if cat in ('env.tool', 'env.packages', 'env.system', 'env.var'):
        # names queried
        names = set()
        for m in re.finditer(r'(?:which|command -v|type|whereis)\s+([^;&|>\n]+)', text):
            names.update(t for t in m.group(1).split() if not t.startswith('-') and not t.startswith('2'))
        for m in re.finditer(r'import\s+([\w.]+)', text): names.add(m.group(1).split('.')[0])
        for m in re.finditer(r'grep\s+(?:-\S+\s+)*["\']([^"\']+)["\']', ' '.join([text] + p.get('filters', []))):
            names.update(x for x in re.split(r'\\?\||\s', m.group(1)) if x)
        m = re.match(r'\s*(\S+)\s+(--version|-V|-v|version)', text)
        if m: names.add(m.group(1).split('/')[-1])
        if cat in ('env.system', 'env.var') or not names:
            return False, True, 'Toolchain check', 'fixed-probe'
        exts = set(re.findall(r'\.([A-Za-z0-9]{1,6})\b', st.task + ' ' + st.all_text()[:5000]))
        probe = set(BASE_TOOLS)
        for e in exts: probe.update(EXT_TOOLS.get(e.lower(), []))
        probe.update(w for w in st.task_words)
        for m in re.finditer(r"(?:command not found|No module named) ?'?:? ?'?([\w.+-]+)", st.all_text()):
            probe.add(m.group(1))
        for m in re.finditer(r'bash: (?:line \d+: )?([\w.+-]+): command not found', st.all_text()):
            probe.add(m.group(1))
        probe.update(st.installed)
        probe_l = {x.lower() for x in probe}
        names_l = {n.lower().strip('\'"$') for n in names if n.strip('\'"$')}
        hit = {n for n in names_l if n in probe_l or n.rstrip('-') in probe_l}
        if names_l and len(hit) >= max(1, int(0.75 * len(names_l))):
            return False, True, 'Toolchain check', 'probe-set'
        return False, False, 'Toolchain check', 'names-missing:' + ','.join(sorted(names_l - hit))[:80]
    if cat in ('data.bytes', 'data.inspect', 'data.schema', 'data.extract', 'data.query'):
        paths = [x for x in targets_of(text) if re.search(r'\.\w{1,8}$', x) or x.startswith('/')]
        if cat == 'data.query' and len(text) > 200:
            return False, False, 'Data peek', 'custom-query'
        if cat == 'data.extract' and '$f' in text:
            # loop over a folder of documents
            return False, True, 'Data peek', 'folder-loop'
        if not paths:
            toks = [t.strip('\'"') for t in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', text)[1:] if not t.startswith('-')]
            paths = [t for t in toks if re.fullmatch(r'[\w.*/-]+', t) and (t in st.all_text() or t in st.task or '*' in t)]
        if re.search(r'\bfile\s+(\S*/)?\*', text):
            return False, True, 'Data peek', 'workdir-glob'
        if paths and all(st.named(x, recent=False) or norm_path(x, st.cwd).startswith('/app/') or '*' in x for x in paths):
            return False, True, 'Data peek', 'named-or-workdir-file'
        if not paths:
            return False, False, 'Data peek', 'no-target'
        return False, False, 'Data peek', 'unnamed'
    if cat in ('service.check', 'service.config_test', 'logs', 'net.check'):
        if cat == 'net.check':
            return False, False, 'Service check', 'external-net'
        if re.search(r'localhost|127\.0\.0\.1', text):
            port = re.search(r':(\d{2,5})', text)
            if port and (port.group(1) in st.task or port.group(1) in recent):
                return False, True, 'Service check', 'task-port'
            if not port and ('localhost' in task_l or 'port' in task_l):
                return False, True, 'Service check', 'task-port'
            return False, False, 'Service check', 'port-unknown'
        if cat == 'logs':
            paths = targets_of(text)
            if paths and all(st.named(x, recent=False) for x in paths):
                return False, True, 'Service check', 'named-log'
            if '*' in text: return False, True, 'Data peek', 'glob-logs'
            return False, False, 'Service check', 'unnamed-log'
        return False, True, 'Service check', 'fixed-probe'
    if cat in ('api.probe', 'api.help'):
        names = re.findall(r"require\(['\"]([^'\"]+)['\"]\)|import\s+([\w.]+)|from\s+([\w.]+)\s+import", text)
        names = [x for t in names for x in t if x]
        m = re.match(r'\s*(\S+)\s+(--help|-h)', text)
        if m: names.append(m.group(1).split('/')[-1])
        if names and all(n.split('.')[0] in st.installed or n.split('.')[0].lower() in task_l or norm_path(n.split('.')[0] + '.py', st.cwd) in st.written or n.split('.')[0] in recent for n in names):
            return False, True, 'Package docs', 'installed-or-named'
        return False, False, 'Package docs', 'unknown-pkg'
    if cat == 'file.compare':
        paths = targets_of(text)
        if paths and all(st.named(x, recent=False) for x in paths):
            return False, True, 'Compare files', 'named'
        return False, False, 'Compare files', 'unnamed'
    if cat == 'git.state':
        return False, True, 'Git state', 'fixed-probe'
    return False, False, 'other', cat


def run(models):
    rows = []
    for s in S:
        st = State(s['task_text'])
        qi = 0
        for t in s['turns']:
            if not t['scout'] and t['model'] in models:
                qi += 1
                parts_info = []
                kinds = []
                for c in t['calls']:
                    a = c['args'] or {}
                    if c['name'] == 'bash':
                        parts = classify_command(a.get('command', ''))
                    elif c['name'] in ('read', 'ls', 'grep', 'find'):
                        parts = [dict(kind=INFO, cat='pi.' + c['name'], head=json.dumps(a), text=json.dumps(a), filters=[])]
                    else:
                        parts = [dict(kind=ACT, cat='pi.' + c['name'], head='', text='', filters=[])]
                    for p in parts:
                        if p['kind'] == NEUTRAL: continue
                        kinds.append(p['kind'])
                        if p['kind'] == INFO:
                            cur, new, tool, note = match_part(p, st, dict(args=a))
                            parts_info.append(dict(cat=p['cat'], text=p['text'][:200], cur=cur, new=new, tool=tool, note=note, bash=c['name'] == 'bash'))
                info_only = bool(kinds) and all(k == INFO for k in kinds)
                rows.append(dict(task=s['task'], group=s['group'], file=s['file'], idx=qi, info_only=info_only, kinds=kinds,
                                 parts=parts_info, has_bash=any(c['name'] == 'bash' for c in t['calls'])))
            st.add_turn(t['calls'])
    return rows


if __name__ == '__main__':
    import pickle
    out = {}
    for name, mf in {'27b': ['qwen3.8-27b'], 'flash': ['qwen3.8-flash-next'], 'glm': ['scissero-glm-5.3']}.items():
        out[name] = run(mf)
    pickle.dump(out, open('/private/tmp/claude-501/qwen-mining/coverage.pkl', 'wb'))
    print('ok', {k: len(v) for k, v in out.items()})
