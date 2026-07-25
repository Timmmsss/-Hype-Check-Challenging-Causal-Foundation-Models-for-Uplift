import json

path = "/home/jupyter/project/robustness_tests/sweep_retailhero.log"
n_errors = 0

with open(path) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("status") == "error":
            n_errors += 1
            if n_errors <= 3:
                print(json.dumps(d, indent=2, ensure_ascii=False))
                print("---")

print("Total errors so far:", n_errors)
