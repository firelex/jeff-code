import json, sys
S=json.load(open('/private/tmp/claude-501/qwen-mining/sessions.json'))
model=sys.argv[1]; lim=int(sys.argv[2])
for s in S:
    q=[t for t in s['turns'] if not t['scout'] and t['model']==model]
    if not q: continue
    print('='*20, s['group'], s['task'])
    i=0
    for t in s['turns']:
        if t['scout']:
            for c in t['calls']: print('   [scout]', c['name'], json.dumps(c['args'])[:150])
            continue
        i+=1
        for c in t['calls']:
            a=c['args'] or {}
            if c['name']=='bash': s_=a.get('command','')
            elif c['name'] in ('write',): s_='WRITE '+a.get('path','')
            elif c['name']=='edit': s_='EDIT '+a.get('path','')
            else: s_=c['name'].upper()+' '+json.dumps(a)
            print(f'T{i:03d}', s_[:lim].replace('\n','\n      '))
            if c['result'] is not None and c['name']=='bash':
                r=c['result'].strip().splitlines()
                print('      -> %d lines%s: %s'%(len(r),' ERR' if c['isError'] else '', ' | '.join(r[:2])[:160]))
