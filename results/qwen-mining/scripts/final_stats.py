import pickle, collections, json, re, random
D=pickle.load(open('/private/tmp/claude-501/qwen-mining/coverage.pkl','rb'))
S=json.load(open('/private/tmp/claude-501/qwen-mining/sessions.json'))
LOOP={'runs-gate0-teacher/dna-insert','runs-gate0-teacher/pytorch-model-cli'}
def key(r): return r['group']+'/'+r['task']
TOOLMAP={'Read (wider memory)':'Read','Search (names from recent output)':'Search+','Find (names from recent output)':'Find+'}
random.seed(5)
for name in ('27b','flash','glm'):
    for excl in (False,True):
        rows=[r for r in D[name] if not (excl and key(r) in LOOP)]
        if excl and len(rows)==len(D[name]): continue
        bt=[r for r in rows if r['has_bash']]
        io=[r for r in rows if r['info_only']]
        iob=[r for r in io if r['has_bash']]
        print(f'#### {name} excl_loops={excl}: turns={len(rows)} bash_turns={len(bt)} info_only={len(io)} (bash {len(iob)}) tasks={len(set(r["task"] for r in rows))}')
        for label,rs in (('all',io),('bash',iob)):
            cf=sum(1 for r in rs if all(p['cur'] for p in r['parts'])); nf=sum(1 for r in rs if all(p['new'] for p in r['parts']))
            print(f'   {label} info-only n={len(rs)} current={cf} ({100*cf/max(1,len(rs)):.0f}%) new={nf} ({100*nf/max(1,len(rs)):.0f}%)')
        # per-tool over all turns
        T=collections.defaultdict(lambda: dict(parts=0,turns=set(),io=set(),tasks=set(),cur=0,new=0,bash_parts=0,ex=[]))
        for r in rows:
            for p in r['parts']:
                t=TOOLMAP.get(p['tool'],p['tool'])
                d=T[t]; d['parts']+=1; d['turns'].add((r['file'],r['idx'])); d['tasks'].add(r['task']); d['cur']+=p['cur']; d['new']+=p['new']; d['bash_parts']+=p['bash']
                if r['info_only']: d['io'].add((r['file'],r['idx']))
                if p['bash']: d['ex'].append(p['text'][:140])
        print('   tool | parts(bash) | turns | info-only turns | %bash turns | tasks | cur% | new%')
        for t,d in sorted(T.items(), key=lambda kv:-len(kv[1]['io'])):
            print(f"   {t:18s} {d['parts']:4d}({d['bash_parts']:4d}) {len(d['turns']):4d} {len(d['io']):4d} {100*len(d['turns'])/len(bt):5.1f}% {len(d['tasks']):3d} {100*d['cur']/d['parts']:4.0f}% {100*d['new']/d['parts']:4.0f}%")
        if not excl and name!='glm':
            for t,d in T.items():
                ex=list(dict.fromkeys(d['ex']))
                print('     EX',t, random.sample(ex,min(4,len(ex))))
