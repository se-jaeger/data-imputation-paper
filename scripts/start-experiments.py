
import subprocess
from pathlib import Path

experiment_name = "test"

# Default Values
MISSING_FRACTIONS = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
MISSING_TYPES = ["MCAR", "MNAR", "MAR"]
IMPUTER = ["mode", "knn", "forest", "dl", "gain"]  # TODO: add VAE

binary_path = Path("../data/raw/binary.txt")
multi_path = Path("../data/raw/multi.txt")
regression_path = Path("../data/raw/regression.txt")

binary_task_ids = binary_path.read_text().split(",")
multi_task_ids = multi_path.read_text().split(",")
regression_task_ids = regression_path.read_text().split(",")
task_ids = [*binary_task_ids, *multi_task_ids, *regression_task_ids]

###############

temp_type = r'\,'.join(MISSING_TYPES)
temp_fractions = r'\,'.join([str(x) for x in MISSING_FRACTIONS])

cmd = "helm install --generate-name"
template = "../cluster/helm/data-imputation"
name_arg = f"--set arguments.experiment_name={experiment_name}"
types_arg = f"--set arguments.missing_types='{temp_type}'"
fractions_arg = f"--set arguments.missing_fractions='{temp_fractions}'"

for task_id in task_ids[:3]:
    for imputer in IMPUTER:
        command = f"{cmd} --set arguments.task_id={task_id} --set arguments.imputer={imputer} {name_arg} {types_arg} {fractions_arg} {template}"
        output = subprocess.run(command, shell=True, capture_output=True)
        print(f"Started id: {task_id:<5} - imputer: {imputer:<6} - Kubernets Job name: {output.stdout.decode('utf-8').splitlines()[0][6:]}")
