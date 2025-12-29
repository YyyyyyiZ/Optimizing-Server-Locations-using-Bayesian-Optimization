import subprocess
import sys

print(sys.executable)


loc_list = [20, 30, 30, 50, 100, 100]
amb_list = [10, 5, 15, 20, 30, 50]
iter_list = [50, 80, 100, 120, 150, 180]
# iter_list = [21,21,21,21,21,21]
num_config = len(loc_list)

# exe =  f"/home/sjtu/.conda/envs/BO4Loc/python "
exe =  "E:\\Anaconda3\\envs\\BO4Location\\python "

base_command = exe + "main.py -p MRT_binary -opt 0 -st_paul 0 -p_median 1 -GA 1 -BOCS 1 -BOCS_sub 1 -CAS 1 --ard --seed 0 --n_trials 3 --n_init 20"
# base_command = "E:\\Anaconda3\\envs\\BO4Location\\python main_mercer.py -p MRT_binary -Mercer 1 --ard --seed 0 --n_trials 10 --n_init 20"


for i in range(num_config):
    command = f"{base_command} -loc {loc_list[i]} -ambulance {amb_list[i]} --max_iters {iter_list[i]}"
    print(f"Executing: {command}")
    subprocess.run(command, shell=True)

