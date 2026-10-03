import json, glob, os, collections
ROOT='/private/tmp/claude-501/qwen-mining/data'
out=[]
for f in sorted(glob.glob(ROOT+'/runs*/*/*__*/agent/pi/sessions/*.jsonl')):
    rel=os.path.relpath(f,ROOT); parts=rel.split('/')
    group=parts[0]; task=parts[2].split('__')[0]
    msgs=[json.loads(l) for l in open(f)]
    msgs=[e['message'] for e in msgs if e.get('type')=='message']
    task_text=None; turns=[]; results={}; models=collections.Counter()
    for m in msgs:
        if m['role']=='user' and task_text is None:
            c=m['content']; task_text=c if isinstance(c,str) else ' '.join(p.get('text','') for p in c if p.get('type')=='text')
        if m['role']=='toolResult':
            c=m.get('content'); txt=c if isinstance(c,str) else '\n'.join(p.get('text','') for p in (c or []) if p.get('type')=='text')
            results[m['toolCallId']]=dict(text=txt,isError=m.get('isError'))
    ti=0
    for m in msgs:
        if m['role']!='assistant': continue
        prov=m.get('provider'); models[(prov,m.get('model'))]+=1
        calls=[p for p in m['content'] if isinstance(p,dict) and p.get('type')=='toolCall']
        turns.append(dict(scout=(prov=='jeff-first'),model=m.get('model'),
            calls=[dict(id=c['id'],name=c['name'],args=c.get('arguments')) for c in calls],
            text=' '.join(p.get('text','') for p in m['content'] if isinstance(p,dict) and p.get('type')=='text')[:500]))
    for t in turns:
        for c in t['calls']:
            r=results.get(c['id']); c['result']=(r['text'] if r else None); c['isError']=(r['isError'] if r else None)
    out.append(dict(file=rel,group=group,task=task,task_text=task_text,models=dict((f'{k[0]}/{k[1]}',v) for k,v in models.items()),turns=turns))
json.dump(out,open('/private/tmp/claude-501/qwen-mining/sessions.json','w'))
for s in out: print(s['group'][:28].ljust(28),s['task'][:28].ljust(28),len(s['turns']),s['models'])
