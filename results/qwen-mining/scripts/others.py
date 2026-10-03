import sys
sys.path.insert(0,'.')
from analyse import turn_records
import collections
R=turn_records(['qwen3.8-flash-next','qwen3.8-27b'])
seen=collections.Counter()
for r in R:
    for c in r['calls']:
        if c['tool']!='bash': continue
        for p in c['parts']:
            if p['cat'].startswith('other:'):
                if seen[p['cat']]>=1: continue
                seen[p['cat']]+=1
                print('###',p['cat'],'::',repr(p['head'][:150]))
                print('   CMD:',repr(c['cmd'][:300]))
