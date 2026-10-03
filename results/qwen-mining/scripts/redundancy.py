import json, collections, re, os
S=json.load(open('/private/tmp/claude-501/qwen-mining/sessions.json'))
def sig(c):
    a=c['args'] or {}
    if c['name']=='bash': return ('bash',a.get('command','').strip())
    if c['name'] in ('read','ls','grep','find'): return (c['name'], json.dumps({k:v for k,v in a.items()},sort_keys=True))
    return None
def readpath(c):
    a=c['args'] or {}
    if c['name']=='read': return os.path.normpath(a.get('path',''))
    if c['name']=='bash':
        m=re.match(r'\s*(?:cd \S+ && )?(?:cat|head|tail|sed -n \S+)\s+(?:-\S+\s+)*(\S+)\s*$', a.get('command',''))
        if m: return m.group(1)
    return None
tot=collections.Counter()
ex=[]
for s in S:
    teacher = any(t['scout'] for t in s['turns'])
    model=None
    since_scout=[]; seen=collections.Counter(); lastedit=-1
    for i,t in enumerate(s['turns']):
        if t['scout']:
            for c in t['calls']: since_scout.append(c)
            continue
        model=t['model']
        for c in t['calls']:
            sg=sig(c)
            if sg is None:
                seen.clear(); continue  # an edit/write resets "same state"
            if c['name']=='bash' and re.search(r'(>|sed -i|apt|pip|npm|mv |cp |rm |mkdir|cat >)', sg[1]):
                pass
            tot[(model,'qwen-info-or-bash-calls')]+=1
            if seen[sg]>0:
                tot[(model,'exact-repeat-of-own-call-since-last-write')]+=1
            seen[sg]+=1
            if teacher:
                tot[(model,'calls-in-scout-sessions')]+=1
                rp=readpath(c)
                ssig=[sig(x) for x in since_scout]
                srp=[readpath(x) for x in since_scout]
                if sg in ssig or (rp and rp in srp):
                    tot[(model,'repeats-scout-step-just-before')]+=1
                    if len(ex)<12: ex.append((s['task'],sg[0],sg[1][:100]))
        since_scout=[]
for k,v in sorted(tot.items()): print(k,v)
for e in ex: print(e)
