import subprocess
cmd = [
    "python3", "run_robustness.py",
    "--cleaned-root", "/home/jupyter/project/data_A_cleaned",
    "--datasets", "lzd",
    "--axes", "scale,control_share,conversion",
    "--models", "t_learner,x_learner,dr_learner,dragonnet,causalpfn",
    "--seeds", "0,1,2,3,4",
    "--equalize-rows", "40000",
    "--max-rows", "0",
    "--skip-existing",
]
log = open("/home/jupyter/project/robustness_tests/sweep_lzd3.log", "w")
proc = subprocess.Popen(cmd, cwd="/home/jupyter/project/robustness_tests", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Launched PID:", proc.pid)
