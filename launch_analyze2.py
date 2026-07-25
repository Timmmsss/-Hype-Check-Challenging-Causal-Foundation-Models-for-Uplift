import subprocess
cmd = [
    "python3", "analyze_robustness.py",
    "--results-root", "/home/jupyter/project/robustness_tests/results_robustness",
    "--n-bootstrap", "50",
    "--bootstrap-max-rows", "5000",
]
log = open("/home/jupyter/project/robustness_tests/analyze2.log", "w")
proc = subprocess.Popen(cmd, cwd="/home/jupyter/project/robustness_tests", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Launched PID:", proc.pid)
