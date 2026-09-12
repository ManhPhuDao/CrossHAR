"""CrossHAR runner arranged as Kaggle/Jupyter cells.

Open this file in VS Code or copy each ``# %%`` block into a Kaggle notebook.
The expected Kaggle input contains one or more dataset folders with
``data_20_120.npy`` and ``label_20_120.npy`` files.
"""

# %% [markdown]
# CrossHAR on Kaggle
#
# Run the cells from top to bottom. Before running them, add the dataset as a
# Kaggle input and update INPUT_ROOT in the configuration cell.

# %% Cell 1: configuration
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/kaggle/working/CrossHAR")
INPUT_ROOT = Path("/kaggle/input/crosshar-dataset")
REPOSITORY_URL = "https://github.com/kingdomrush2/CrossHAR.git"

SOURCE_DATASET = "uci"
TARGET_DATASET = "hhar"
VERSION = "20_120"
LABEL_RATE = 0.1
AUGMENT_METHOD = "channel_aug"
GPU = "0"
RUN_CROSS_DATASET = True


# %% Cell 2: install dependencies and locate the repository
if not ROOT.exists():
    subprocess.run(["git", "clone", REPOSITORY_URL, str(ROOT)], check=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "einops==0.7.0", "torchinfo==1.8.0", "ujson==5.9.0"],
    check=True,
)

print(f"Repository: {ROOT}")
print(f"Python: {sys.executable}")


# %% Cell 3: copy Kaggle datasets into CrossHAR's expected layout
def find_dataset_source(input_root: Path, dataset_name: str, version: str) -> Path:
    filenames = (f"data_{version}.npy", f"label_{version}.npy")
    candidates = (input_root / dataset_name, input_root)

    for candidate in candidates:
        if all((candidate / filename).exists() for filename in filenames):
            return candidate

    raise FileNotFoundError(
        f"Khong tim thay {filenames} trong {input_root}. "
        "Cau truc dung: <input_root>/<dataset>/<file>.npy"
    )


def copy_dataset(input_root: Path, dataset_name: str, version: str) -> None:
    source_dir = find_dataset_source(input_root, dataset_name, version)
    target_dir = ROOT / "dataset" / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in (f"data_{version}.npy", f"label_{version}.npy"):
        shutil.copy2(source_dir / filename, target_dir / filename)
    print(f"Copied {dataset_name} from {source_dir}")


(ROOT / "saved").mkdir(exist_ok=True)
(ROOT / "embed").mkdir(exist_ok=True)
copy_dataset(INPUT_ROOT, SOURCE_DATASET, VERSION)
if RUN_CROSS_DATASET and TARGET_DATASET:
    copy_dataset(INPUT_ROOT, TARGET_DATASET, VERSION)


# %% Cell 4: inspect the data and GPU
import numpy as np
import torch


for dataset_name in [SOURCE_DATASET] + ([TARGET_DATASET] if RUN_CROSS_DATASET else []):
    data_path = ROOT / "dataset" / dataset_name / f"data_{VERSION}.npy"
    label_path = ROOT / "dataset" / dataset_name / f"label_{VERSION}.npy"
    data = np.load(data_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    print(f"{dataset_name}: data={data.shape}, labels={labels.shape}")

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# %% Cell 5: self-supervised pretraining
def run_script(script_name: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / script_name), *args]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


common_args = [
    "-d", SOURCE_DATASET,
    "-dv", VERSION,
    "-g", GPU,
    "-s", "model",
    "-am", AUGMENT_METHOD,
]
run_script("pretrain.py", *common_args)


# %% Cell 6: generate source-dataset embeddings
run_script("embedding.py", *common_args)


# %% Cell 7: fine-tune the activity classifier
run_script("classifier.py", *common_args, "-lr", str(LABEL_RATE))


# %% Cell 8: evaluate on an unseen target dataset
if RUN_CROSS_DATASET:
    cross_args = [
        "-d", SOURCE_DATASET,
        "-td", TARGET_DATASET,
        "-dv", VERSION,
        "-g", GPU,
        "-s", "model",
    ]
    run_script("cross_dataset_test.py", *cross_args)


# %% Cell 9: list outputs
for output_dir in (ROOT / "saved", ROOT / "embed"):
    print(f"\n{output_dir}")
    for output_file in sorted(output_dir.rglob("*")):
        if output_file.is_file():
            print(output_file.relative_to(ROOT))