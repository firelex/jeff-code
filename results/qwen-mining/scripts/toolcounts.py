import json, collections
S=json.load(open('/private/tmp/claude-501/qwen-mining/sessions.json'))
by=collections.defaultdict(collections.Counter); nturn=collections.Counter(); ncall=collections.Counter()
for s in S:
    for t in s['turns']:
        if t['scout']: continue
        m=t['model']; nturn[m]+=1
        if len(t['calls'])>1: ncall[m]+=1
        for c in t['calls']: by[m][c['name']]+=1
        if not t['calls']: by[m]['<none>']+=1
for m in by: print(m, nturn[m], 'multi-call turns',ncall[m], dict(by[m]))
# scout tool use
sc=collections.Counter()
for s in S:
    for t in s['turns']:
        if t['scout']:
            for c in t['calls']: sc[c['name']]+=1
print('scout',sc)
