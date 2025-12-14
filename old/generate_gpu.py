import os

# --- Configuration ---

# The list of population sizes (N) you want to test
N_VALUES = [
    100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
    100000, 200000, 500000, 1000000, 2000000, 5000000
]

# A list of dictionaries, where each dictionary defines a command to be run.
# The placeholders {N} and {LOG_FILE} will be replaced for each job.
TASKS = [
    {
        "algorithm": "gillespie",
        "command": "~/.conda/envs/gemf/bin/python benchmark.py --algorithm gillespie --N {N} --T_MAX 5 --runs 1000 --device cuda --log_file {LOG_FILE}"
    },
    {
        "algorithm": "tau_node",
        "command": "~/.conda/envs/gemf/bin/python benchmark.py --algorithm tau_node --N {N} --T_MAX 5 --runs 1000 --theta 5 --device cuda --log_file {LOG_FILE}"
    },
    {
        "algorithm": "tau_system",
        "command": "~/.conda/envs/gemf/bin/python benchmark.py --algorithm tau_system --N {N} --T_MAX 5 --runs 1000 --theta 5 --device cuda --log_file {LOG_FILE}"
    },
    {
        "algorithm": "gemfpy",
        "command": "~/.conda/envs/gemf/bin/python benchmark.py --algorithm gemfpy --N {N} --T_MAX 5 --runs 1000 --device cuda --log_file {LOG_FILE}"
    }
]

# Directory to save the generated SLURM scripts
OUTPUT_DIR = "slurm_jobs"

# --- SLURM Template ---
# This is the base template for your SLURM script.
# Placeholders like {JOB_NAME}, {TIME}, {MEM}, and {COMMANDS} will be filled in.
SLURM_TEMPLATE = """#!/bin/bash

# Submit this script with: sbatch {FILENAME}

#SBATCH --time={TIME}          # job time limit
#SBATCH --nodes=1              # number of nodes
#SBATCH --ntasks-per-node=1    # number of tasks per node
#SBATCH --cpus-per-task=16     # number of CPU cores per task
#SBATCH --gres=gpu:a100:1   # gpu devices per node
#SBATCH --partition=gpu   # partition
#SBATCH --mem={MEM}            # memory
#SBATCH -J "{JOB_NAME}"        # job name
#SBATCH --account=shakeri-lab  # allocation name
#SBATCH --export=NONE          # do not export environment variables

echo "======================================================"
echo "Job started on $(hostname) at $(date)"
echo "Starting Slurm Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "======================================================"
echo ""

module purge
module load miniforge
module load cuda
module load cudnn

conda activate gemf

nvidia-smi

echo ""
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

cd /home/yxn7cj/GEMFPy2

{COMMANDS}

echo "----------------------------------------------------"
echo "Job finished at $(date)"
echo "----------------------------------------------------"
"""

def get_resource_allocation(n_value):
    """
    Determines the time and memory allocation based on the population size N.
    Adjust these values based on your expected resource needs.
    """
    return "0:30:00", "16G"

def generate_scripts():
    """
    Main function to generate all the SLURM scripts.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # To create a master script to run all jobs
    master_submit_script_path = os.path.join(OUTPUT_DIR, "submit_all_gpu.sh")
    with open(master_submit_script_path, "w") as master_file:
        master_file.write("#!/bin/bash\n")
        master_file.write("# This script submits all generated SLURM jobs\n\n")

        for task in TASKS:
            for n in N_VALUES:
                algorithm = task["algorithm"]

                # 1. Define names and paths
                job_name = f"{algorithm}_{n}_gpu"
                
                # Construct log file name, handling special case for discrete_time
                if algorithm == "discrete_time":
                    log_file_name = f"{algorithm}_{n}_50_gpu.json"
                elif algorithm in ["tau_node", "tau_system"]:
                    log_file_name = f"{algorithm}_{n}_5_gpu.json"
                else:
                    log_file_name = f"{algorithm}_{n}_gpu.json"

                slurm_filename = f"{job_name}.slurm"
                slurm_filepath = os.path.join(OUTPUT_DIR, slurm_filename)

                # 2. Get resource allocation
                time_limit, mem_limit = get_resource_allocation(n)

                # 3. Format the command
                command = task["command"].format(N=n, LOG_FILE=log_file_name)

                # 4. Populate the SLURM template
                script_content = SLURM_TEMPLATE.format(
                    FILENAME=slurm_filename,
                    TIME=time_limit,
                    MEM=mem_limit,
                    JOB_NAME=job_name,
                    COMMANDS=command
                )

                # 5. Write the SLURM script to a file
                with open(slurm_filepath, "w") as f:
                    f.write(script_content)

                print(f"Generated: {slurm_filepath}")
                
                # 6. Add the sbatch command to the master script
                master_file.write(f"sbatch {slurm_filename}\n")

    # Make the master script executable
    os.chmod(master_submit_script_path, 0o755)
    print("\n----------------------------------------------------")
    print("All SLURM scripts have been generated.")
    print(f"A master submission script has been created at: {master_submit_script_path}")
    print(f"Navigate to '{OUTPUT_DIR}' and run './submit_all_gpu.sh' to submit all jobs.")
    print("----------------------------------------------------")


if __name__ == "__main__":
    generate_scripts()
