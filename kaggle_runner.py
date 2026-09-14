from pathlib import Path
import shutil
import subprocess
import sys



ROOT = Path("/kaggle/working/CrossHAR")

# Tự động tìm đường dẫn chứa CrossDataset trong /kaggle/input (không lo sai hoa/thường)
input_candidates = list(Path("/kaggle/input").rglob("CrossDataset")) + list(Path("/kaggle/input").rglob("crossdataset"))
if input_candidates:
    INPUT_ROOT = input_candidates[0]
else:
    # Nếu không tìm thấy slug cụ thể, quét toàn bộ /kaggle/input
    INPUT_ROOT = Path("/kaggle/input")

REPOSITORY_URL = "https://github.com/ManhPhuDao/CrossHAR.git"

SOURCE_DATASET = "uci"
TARGET_DATASET = "hhar"
VERSION = "20_120"
LABEL_RATE = 0.1
AUGMENT_METHOD = "channel_aug"
GPU = "0"
RUN_CROSS_DATASET = True

print(f"-> Thư mục dữ liệu đầu vào xác định tại: {INPUT_ROOT}")
print(f"-> Đã thiết lập cấu hình: Source={SOURCE_DATASET}, Target={TARGET_DATASET}")


# %% [Cell 2]: Clone GitHub Repository (LUON XOA SACH BAN CU TRUOC KHI CLONE) và Cài đặt Thư viện
# QUAN TRONG: xoa thu muc cu truoc, de dam bao repo luon la ban goc sach,
# khong bi anh huong boi cac lan patch loi truoc do (vi /kaggle/working
# duoc giu nguyen giua cac lan chay cell trong cung 1 session).
if ROOT.exists():
    print(f"Đang xóa bản clone cũ tại {ROOT} để đảm bảo lấy bản gốc sạch...")
    shutil.rmtree(ROOT)

print(f"Đang clone repository từ {REPOSITORY_URL}...")
subprocess.run(["git", "clone", REPOSITORY_URL, str(ROOT)], check=True)

print("Đang cài đặt các thư viện phụ thuộc...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "einops==0.7.0", "torchinfo==1.8.0", "ujson==5.9.0", "transforms3d", "scikit-learn"],
    check=True,
)

print(f"Thư mục dự án: {ROOT}")
print(f"Môi trường Python: {sys.executable}")

# %% [Cell 2b]: Patch hàm permutation trong augmentations.py - PATCH THEO TUNG DONG (an toan ve thut le)
import re

aug_file = ROOT / "augmentations.py"

if aug_file.exists():
    with open(aug_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    patched = False
    for line in lines:
        # Chi khop CHINH XAC dong chua ca 2 cum tu nay, tranh khop nham
        if "np.random.permutation(splits)" in line and "np.concatenate" in line:
            # Tu phat hien indent CHI TU CHINH DONG NAY (khong lan sang dong khac)
            indent_match = re.match(r"^([ \t]*)", line)
            indent = indent_match.group(1) if indent_match else ""
            new_lines.append(f"{indent}np.random.shuffle(splits)\n")
            new_lines.append(f"{indent}warp = np.concatenate(splits).ravel()\n")
            patched = True
        else:
            new_lines.append(line)

    if patched:
        with open(aug_file, "w") as f:
            f.writelines(new_lines)
        print("-> Đã patch thành công augmentations.py (patch theo dòng, an toàn về thụt lề)!")
    else:
        print("-> Không tìm thấy dòng cần patch. Có thể phiên bản repo đã khác - "
              "hãy in nội dung augmentations.py ra kiểm tra thủ công dòng chứa 'permutation'.")
else:
    print(f"-> Không tìm thấy file {aug_file}, kiểm tra lại đường dẫn ROOT.")

# Kiem tra nhanh cu phap file sau khi patch, phat hien loi thut le som truoc khi chay pretrain.py
import py_compile
try:
    py_compile.compile(str(aug_file), doraise=True)
    print("-> Cú pháp augmentations.py hợp lệ sau khi patch.")
except py_compile.PyCompileError as e:
    print("-> LỖI CÚ PHÁP sau khi patch, xem chi tiết bên dưới và kiểm tra file thủ công:")
    print(e)
    raise

# %% [Cell 2c]: Patch train.py - LIVE PROGRESS + ETA TOÀN BỘ

train_file = ROOT / "train.py"

with open(train_file, "r") as f:
    content = f.read()

anchor = "                global_step += 1\n"

live_progress_code = """                global_step += 1

                # ===== LIVE PROGRESS + ETA TOAN BO PRETRAIN =====
                if i == 0:
                    self._epoch_start_time_hienthi = time.perf_counter()

                    if e == n_epoch_now:
                        self._total_train_start_time = time.perf_counter()

                so_batch_moi_epoch = len(data_loader_train)

                if (i + 1) % 20 == 0 or (i + 1) == so_batch_moi_epoch:

                    # Thoi gian epoch hien tai
                    thoi_gian_epoch = (
                        time.perf_counter()
                        - self._epoch_start_time_hienthi
                    )

                    thoi_gian_tb_batch = thoi_gian_epoch / (i + 1)

                    batch_con_lai = (
                        so_batch_moi_epoch - (i + 1)
                    )

                    eta_epoch = (
                        thoi_gian_tb_batch * batch_con_lai
                    )

                    # Tien do toan bo
                    tong_epoch = self.cfg.n_epochs

                    epoch_da_xong = e - n_epoch_now

                    tien_do_epoch = (
                        (i + 1) / so_batch_moi_epoch
                    )

                    so_epoch_da_hoan_thanh = (
                        epoch_da_xong + tien_do_epoch
                    )

                    tong_thoi_gian_da_chay = (
                        time.perf_counter()
                        - self._total_train_start_time
                    )

                    # Thoi gian trung binh / epoch
                    if so_epoch_da_hoan_thanh > 0:
                        thoi_gian_tb_epoch = (
                            tong_thoi_gian_da_chay
                            / so_epoch_da_hoan_thanh
                        )
                    else:
                        thoi_gian_tb_epoch = 0

                    # Epoch con lai
                    epoch_con_lai = max(
                        0,
                        tong_epoch - so_epoch_da_hoan_thanh
                    )

                    # ETA toan bo
                    eta_toan_bo = (
                        thoi_gian_tb_epoch
                        * epoch_con_lai
                    )

                    # Format thoi gian
                    def _format_time(seconds):
                        seconds = int(max(0, seconds))
                        h = seconds // 3600
                        m = (seconds % 3600) // 60
                        s = seconds % 60
                        return f"{h:02d}:{m:02d}:{s:02d}"

                    # Progress %
                    progress = (
                        so_epoch_da_hoan_thanh
                        / tong_epoch
                        * 100
                    )

                    # Thoi diem du kien xong
                    import datetime

                    finish_time = (
                        datetime.datetime.now()
                        + datetime.timedelta(
                            seconds=eta_toan_bo
                        )
                    )

                    print(
                        "\\n"
                        + "=" * 75
                        + "\\n"
                        + "PRETRAINING PROGRESS\\n"
                        + "=" * 75
                        + "\\n"
                        + f"Epoch        : {e + 1}/{tong_epoch}\\n"
                        + f"Batch        : {i + 1}/{so_batch_moi_epoch}\\n"
                        + f"Progress     : {progress:.2f}%\\n"
                        + f"Loss         : {loss.item():.4f}\\n"
                        + "-" * 75
                        + "\\n"
                        + f"Epoch time   : {_format_time(thoi_gian_epoch)}\\n"
                        + f"Total elapsed: {_format_time(tong_thoi_gian_da_chay)}\\n"
                        + f"Avg/epoch    : {_format_time(thoi_gian_tb_epoch)}\\n"
                        + f"ETA epoch    : {_format_time(eta_epoch)}\\n"
                        + f"ETA toàn bộ   : {_format_time(eta_toan_bo)}\\n"
                        + f"Còn lại      : {epoch_con_lai:.1f} epochs\\n"
                        + f"Dự kiến xong : {finish_time.strftime('%H:%M:%S')}\\n"
                        + "=" * 75,
                        flush=True,
                    )
"""

if anchor in content and "PRETRAINING PROGRESS" not in content:
    content = content.replace(anchor, live_progress_code, 1)

    with open(train_file, "w") as f:
        f.write(content)

    print("-> Đã patch train.py thành công!")
    print("-> Đã thêm ETA toàn bộ pretrain.")

elif "PRETRAINING PROGRESS" in content:
    print("-> train.py đã có ETA, không patch lại.")

else:
    print("-> Không tìm thấy vị trí patch trong train.py.")


# Kiểm tra cú pháp
import py_compile

py_compile.compile(
    str(train_file),
    doraise=True
)

print("-> train.py hợp lệ.")



# %% [Cell 3]: Tự động quét và chuẩn bị cấu hình dữ liệu vào CrossHAR
def find_dataset_source(input_root: Path, dataset_name: str, version: str) -> Path:
    filenames = (f"data_{version}.npy", f"label_{version}.npy")

    # Quét tất cả các thư mục con trong /kaggle/input để tìm folder khớp với dataset_name
    for path in input_root.rglob(dataset_name):
        if path.is_dir() and all((path / f).exists() for f in filenames):
            return path

    raise FileNotFoundError(
        f"Không tìm thấy các file {filenames} thuộc dataset [{dataset_name}] trong {input_root}."
    )

def copy_dataset(input_root: Path, dataset_name: str, version: str) -> None:
    source_dir = find_dataset_source(input_root, dataset_name, version)
    target_dir = ROOT / "dataset" / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in (f"data_{version}.npy", f"label_{version}.npy"):
        shutil.copy2(source_dir / filename, target_dir / filename)
    print(f"Đã sao chép thành công [{dataset_name}] từ {source_dir}")


# Tạo các thư mục lưu trữ mô hình và embedding
(ROOT / "saved").mkdir(exist_ok=True)
(ROOT / "embed").mkdir(exist_ok=True)

# Sao chép tập nguồn
copy_dataset(INPUT_ROOT, SOURCE_DATASET, VERSION)

# Sao chép tập mục tiêu (nếu kích hoạt chạy Cross-dataset)
if RUN_CROSS_DATASET and TARGET_DATASET:
    copy_dataset(INPUT_ROOT, TARGET_DATASET, VERSION)


# %% [Cell 4]: Kiểm tra kích thước dữ liệu và thông tin GPU
import numpy as np
import torch

print("--- KIỂM TRA DỮ LIỆU ---")
datasets_to_check = [SOURCE_DATASET] + ([TARGET_DATASET] if RUN_CROSS_DATASET else [])
for ds_name in datasets_to_check:
    data_path = ROOT / "dataset" / ds_name / f"data_{VERSION}.npy"
    label_path = ROOT / "dataset" / ds_name / f"label_{VERSION}.npy"

    data = np.load(data_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    print(f"Dataset [{ds_name}]: Shape data={data.shape}, Shape labels={labels.shape}")

print("\n--- KIỂM TRA MÔI TRƯỜNG PHẦN CỨNG ---")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))


# %% [Cell 5]: Bước 1 - Pre-training (Học tự giám sát)
import time
from datetime import timedelta

# Luu lai thoi gian chay cua tung buoc de tong ket cuoi cung
step_durations = {}  # {ten_buoc: so_giay}


def format_duration(seconds: float) -> str:
    """Doi so giay sang dinh dang de doc: vi du 1:23:45"""
    return str(timedelta(seconds=round(seconds)))


def run_script(script_name: str, *args: str, step_name: str | None = None) -> float:
    """Chay 1 script con va DO THOI GIAN thuc thi.

    Tra ve so giay da chay, dong thoi luu vao step_durations neu co step_name.
    """
    command = [sys.executable, "-u", str(ROOT / script_name), *args]
    label = step_name or script_name
    print(f"\n>>> Running: {' '.join(command)}")

    start_time = time.perf_counter()
    start_wall = time.strftime("%H:%M:%S")
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start_time
    end_wall = time.strftime("%H:%M:%S")

    print(f">>> [{label}] bat dau luc {start_wall}, ket thuc luc {end_wall}, "
          f"thoi gian chay: {format_duration(elapsed)} ({elapsed:.1f} giay)")

    if step_name:
        step_durations[step_name] = elapsed
    return elapsed


# Tham số dùng chung cho quá trình huấn luyện tập nguồn
common_args = [
    "-d", SOURCE_DATASET,
    "-dv", VERSION,
    "-g", GPU,
    "-s", "model",
    "-am", AUGMENT_METHOD,
]

print("=== START STEP 1: PRE-TRAINING ===")
run_script("pretrain.py", *common_args, step_name="pretrain")


# %% [Cell 6]: Bước 2 - Trích xuất Đặc trưng (Embedding Generation)
print("=== START STEP 2: GENERATING EMBEDDINGS ===")
run_script("embedding.py", *common_args, step_name="embedding")


# %% [Cell 7]: Bước 3 - Tinh chỉnh Bộ phân loại (Classifier Fine-tuning)
print("=== START STEP 3: FINE-TUNING CLASSIFIER ===")
# Tinh chỉnh chỉ với tỷ lệ nhãn hạn chế (LABEL_RATE)
run_script("classifier.py", *common_args, "-lr", str(LABEL_RATE), step_name="classifier_finetune")


# %% [Cell 8]: Bước 4 - Đánh giá Chuyển giao Đa tập dữ liệu (Cross-Dataset Testing)
if RUN_CROSS_DATASET:
    print("=== START STEP 4: CROSS-DATASET EVALUATION ===")
    cross_args = [
        "-d", SOURCE_DATASET,
        "-td", TARGET_DATASET,
        "-dv", VERSION,
        "-g", GPU,
        "-s", "model",
    ]
    run_script("cross_dataset_test.py", *cross_args, step_name="cross_dataset_test")


# %% [Cell 9]: Tổng kết và liệt kê các tệp kết quả thu được
print("\n=== DANH SÁCH FILE KẾT QUẢ TẠO RA ===")
for output_dir in (ROOT / "saved", ROOT / "embed"):
    print(f"\nThư mục: {output_dir}")
    for output_file in sorted(output_dir.rglob("*")):
        if output_file.is_file():
            print(f" - {output_file.relative_to(ROOT)}")

print("\n=== TỔNG KẾT THỜI GIAN CHẠY TỪNG BƯỚC ===")
total_seconds = sum(step_durations.values())
for step_name, seconds in step_durations.items():
    percent = (seconds / total_seconds * 100) if total_seconds else 0
    print(f" - {step_name:<20}: {format_duration(seconds)} ({seconds:.1f} giây, {percent:.1f}% tổng thời gian)")
print(f"\nTỔNG THỜI GIAN TRAIN (tất cả các bước): {format_duration(total_seconds)} ({total_seconds:.1f} giây)")

# Luu lai file thoi gian de tham khao sau nay (vi du dua vao luan van)
timing_log_path = ROOT / "saved" / "timing_log.txt"
with open(timing_log_path, "w") as f:
    f.write(f"Nguồn: {SOURCE_DATASET}, Đích: {TARGET_DATASET}, Version: {VERSION}\n")
    f.write(f"Thời gian chạy ghi lúc: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    for step_name, seconds in step_durations.items():
        f.write(f"{step_name}: {format_duration(seconds)} ({seconds:.1f} giây)\n")
    f.write(f"\nTổng: {format_duration(total_seconds)} ({total_seconds:.1f} giây)\n")
print(f"\n-> Đã lưu log thời gian vào: {timing_log_path}")