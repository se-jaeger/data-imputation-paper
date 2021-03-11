
import subprocess
from pathlib import Path

# CHANGE ME!
experiment_name = "1"


# Default Values
num_repetitions = 5
base_path = "/results"

MISSING_FRACTIONS = [0.01, 0.05, 0.1, 0.3, 0.5]
MISSING_TYPES = ["MCAR", "MNAR", "MAR"]
IMPUTER = ["mode", "knn", "forest", "dl", "gain", "vae"]
STRATEGIES = ["single_single", "single_all"]

binary_task_ids = Path("../data/raw/binary.txt").read_text().split(",")
multi_task_ids = Path("../data/raw/multi.txt").read_text().split(",")
regression_task_ids = Path("../data/raw/regression.txt").read_text().split(",")

task_ids = [*binary_task_ids, *multi_task_ids, *regression_task_ids]
task_ids = [binary_task_ids[0], multi_task_ids[0], regression_task_ids[0]]
task_ids = [binary_task_ids[0], binary_task_ids[-1], multi_task_ids[0], multi_task_ids[-1], regression_task_ids[0], regression_task_ids[-1]]

###############

types_as_argument_string = r'\,'.join(MISSING_TYPES)
strategies_as_argument_string = r'\,'.join(STRATEGIES)
fractions_as_argument_string = r'\,'.join([str(x) for x in MISSING_FRACTIONS])

cmd = "helm install --generate-name"
template = "../cluster/helm/data-imputation"
name_arg = f"--set experiment_name={experiment_name}"
types_arg = f"--set missing_types='{types_as_argument_string}'"
strategies_arg = f"--set strategies='{strategies_as_argument_string}'"
fractions_arg = f"--set missing_fractions='{fractions_as_argument_string}'"
num_repetitions_arg = f"--set num_repetitions='{num_repetitions}'"
base_path_arg = f"--set base_path='{base_path}'"

for task_id in task_ids:
    for imputer in IMPUTER:
        command = f"{cmd} --set task_id={task_id} --set imputer={imputer} {name_arg} {types_arg} {strategies_arg} \
            {fractions_arg} {num_repetitions_arg} {base_path_arg} {template}"
        output = subprocess.run(command, shell=True, capture_output=True)
        print(f"Started id: {task_id:<5} - imputer: {imputer:<6} - Kubernets Job name: {output.stdout.decode('utf-8').splitlines()[0][6:]}")
