import pickle, collections
D=pickle.load(open('/private/tmp/claude-501/qwen-mining/coverage.pkl','rb'))
for name,rows in D.items():
    io=[r for r in rows if r['info_only']]
    iob=[r for r in io if r['has_bash']]
    print(f'==== {name}: turns={len(rows)} info-only={len(io)} (bash {len(iob)}, pi-tools-only {len(io)-len(iob)})')
    def cov(rs, key):
        full=sum(1 for r in rs if r['parts'] and all(p[key] for p in r['parts']))
        part=sum(1 for r in rs if r['parts'] and any(p[key] for p in r['parts']))
        return full, part
    for label, rs in (('all info-only',io),('bash info-only',iob)):
        c=cov(rs,'cur'); n=cov(rs,'new')
        print(f'  {label}: n={len(rs)} current full={c[0]} ({100*c[0]/max(1,len(rs)):.0f}%) any={c[1]} | with new full={n[0]} ({100*n[0]/max(1,len(rs)):.0f}%) any={n[1]} ({100*n[1]/max(1,len(rs)):.0f}%)')
    # tools
    tc=collections.Counter(); tcov=collections.Counter(); tcur=collections.Counter(); tturn=collections.defaultdict(set); ttask=collections.defaultdict(set)
    for r in io:
        for p in r['parts']:
            tc[p['tool']]+=1; tcov[p['tool']]+=p['new']; tcur[p['tool']]+=p['cur']; tturn[p['tool']].add((r['file'],r['idx'])); ttask[p['tool']].add(r['task'])
    print('  tool demand in info-only turns: parts / covered-now / covered-with-new / turns / tasks')
    for k,v in tc.most_common(): print(f'    {k:36s} {v:4d} {tcur[k]:4d} {tcov[k]:4d} {len(tturn[k]):4d} {len(ttask[k]):3d}')
    notes=collections.Counter((p['tool'],p['note'].split(':')[0]) for r in io for p in r['parts'] if not p['new'])
    print('  uncovered reasons:', notes.most_common(15))
    # position
    pos=collections.Counter(); posall=collections.Counter()
    for r in rows:
        b='1-3' if r['idx']<=3 else '4-10' if r['idx']<=10 else '11-30' if r['idx']<=30 else '31+'
        posall[b]+=1; pos[b]+=r['info_only']
    print('  position (info-only/all):', {b:(pos[b],posall[b],f"{100*pos[b]/posall[b]:.0f}%") for b in ['1-3','4-10','11-30','31+'] if posall[b]})
