path = "/home/jupyter/project/robustness_tests/sweep_retailhero.log"
with open(path) as f:
    lines = f.readlines()

idx = None
for i, line in enumerate(lines):
    if '"model": "causalpfn#ctx1024"' in line and '"status": "error"' in line:
        idx = i
        break

print("Error JSON line number:", idx)
start = max(0, idx - 50)
print("---- RAW LOG CONTEXT BEFORE ERROR ----")
for l in lines[start:idx+1]:
    print(l.rstrip())
