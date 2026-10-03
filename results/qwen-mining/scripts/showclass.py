import sys
sys.path.insert(0,'/private/tmp/claude-501/qwen-mining/scripts')
from analyse import turn_records
model=sys.argv[1]; cls=sys.argv[2]; n=int(sys.argv[3])
R=[r for r in turn_records([model]) if r['tclass']==cls and r['has_bash']]
import random; random.seed(1)
for r in (random.sample(R,min(n,len(R)))):
    print(f"--- {r['task']} T{r['idx']} [{r['group']}]")
    for c in r['calls']:
        for p in c['parts']:
            if p['kind']!='neutral': print(f"   {p['kind']:7s} {p['cat']:18s} {p['text'][:160]!r}")
