import subprocess
cmd = [
    "python3", "run_robustness.py",
    "--cleaned-root", "/home/jupyter/project/data_A_cleaned",
    "--datasets", "retailhero",
    "--axes", "scale,control_share,conversion",
    "--seeds", "0,1,2,3,4,5,6,7,8,9",
    "--equalize-rows", "40000",
    "--max-rows", "0",
    "--gp-component-variants", "10,20,40",
    "--q-min-converters-variants", "50,100",
]
log = open("/home/jupyter/project/robustness_tests/sweep_hparam.log", "w")
proc = subprocess.Popen(cmd, cwd="/home/jupyter/project/robustness_tests", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("Launched PID:", proc.pid)
