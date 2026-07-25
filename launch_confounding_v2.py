import subprocess
cmd = ["python3", "/home/jupyter/project/run_confounding_sweep_v2.py"]
log = open("/home/jupyter/project/robustness_tests/sweep_confounding_v2.log", "w")
proc = subprocess.Popen(cmd, cwd="/home/jupyter/project/robustness_tests", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Launched PID:", proc.pid)
