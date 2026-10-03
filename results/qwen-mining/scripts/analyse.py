import json, collections, sys, re
sys.path.insert(0, '/private/tmp/claude-501/qwen-mining/scripts')
from classify import classify_command, INFO, ACT, NEUTRAL, PROBE

S = json.load(open('/private/tmp/claude-501/qwen-mining/sessions.json'))
PI_INFO = {'read': 'pi.read', 'ls': 'pi.ls', 'grep': 'pi.grep', 'find': 'pi.find'}


def turn_records(model_filter):
    recs = []
    for s in S:
        qi = 0
        prev_out = ''
        for t in s['turns']:
            if t['scout']:
                continue
            if t['model'] not in model_filter:
                continue
            qi += 1
            calls = []
            for c in t['calls']:
                a = c['args'] or {}
                if c['name'] == 'bash':
                    parts = classify_command(a.get('command', ''))
                    calls.append(dict(tool='bash', cmd=a.get('command', ''), parts=parts, result=c['result'], isError=c['isError']))
                elif c['name'] in PI_INFO:
                    calls.append(dict(tool=c['name'], args=a, parts=[dict(kind=INFO, cat=PI_INFO[c['name']], head=json.dumps(a), text=json.dumps(a), filters=[])], result=c['result'], isError=c['isError']))
                else:
                    calls.append(dict(tool=c['name'], args={k: v for k, v in a.items() if k == 'path'}, parts=[dict(kind=ACT, cat='pi.' + c['name'], head=a.get('path', ''), text=a.get('path', ''), filters=[])], result=None, isError=c['isError']))
            kinds = [p['kind'] for c in calls for p in c['parts'] if p['kind'] != NEUTRAL]
            has_bash = any(c['tool'] == 'bash' for c in calls)
            if not kinds:
                tclass = 'empty'
            elif all(k == INFO for k in kinds):
                tclass = 'info-only'
            elif all(k in (INFO, PROBE) for k in kinds):
                tclass = 'info+compute' if INFO in kinds else 'compute-only'
            elif any(k == INFO for k in kinds):
                tclass = 'mixed'
            else:
                tclass = 'act-only'
            recs.append(dict(group=s['group'], task=s['task'], file=s['file'], idx=qi, calls=calls, tclass=tclass, has_bash=has_bash, prev_out=prev_out))
            prev_out = '\n'.join((c['result'] or '')[:300] for c in calls)
    return recs


if __name__ == '__main__':
    models = {'27b': ['qwen3.8-27b'], 'flash': ['qwen3.8-flash-next'], 'glm': ['scissero-glm-5.3']}
    for name, mf in models.items():
        R = turn_records(mf)
        bash_turns = [r for r in R if r['has_bash']]
        print(f'===== {name}: turns={len(R)} bash_turns={len(bash_turns)} tasks={len(set(r["task"] for r in R))} sessions={len(set(r["file"] for r in R))}')
        print(' turn classes (all turns):', collections.Counter(r['tclass'] for r in R))
        print(' turn classes (bash turns):', collections.Counter(r['tclass'] for r in bash_turns))
        pc = collections.Counter(); tc = collections.Counter(); tk = collections.defaultdict(set)
        for r in R:
            seen = set()
            for c in r['calls']:
                if c['tool'] != 'bash': continue
                for p in c['parts']:
                    key = (p['kind'], p['cat'])
                    pc[key] += 1
                    seen.add(key)
                    tk[key].add(r['task'])
            for k in seen: tc[k] += 1
        tot = sum(v for k, v in pc.items() if k[0] != NEUTRAL)
        print(' bash parts (non-neutral):', tot)
        for k, v in pc.most_common():
            print(f'   {k[0]:8s} {k[1]:24s} parts={v:4d} turns={tc[k]:4d} ({100*tc[k]/max(1,len(bash_turns)):.1f}% bash turns) tasks={len(tk[k])}')
