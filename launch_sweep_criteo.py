import subprocess
cmd = [
    "python3", "run_robustness.py",
    "--cleaned-root", "/home/jupyter/project/data_A_cleaned",
    "--datasets", "criteo",
    "--axes", "scale,control_share,conversion",
    "--models", "t_learner,x_learner,dr_learner",
    "--seeds", "0,1,2",
    "--equalize-rows", "40000",
    "--max-rows", "0",
]
log = open("/home/jupyter/project/robustness_tests/sweep_criteo.log", "w")
proc = subprocess.Popen(cmd, cwd="/home/jupyter/project/robustness_tests", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Launched PID:", proc.pid)
