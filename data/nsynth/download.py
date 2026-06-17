import requests
import tarfile
import time
import shutil
from tqdm import tqdm
from paths import NSYNTH_DIR

NSYNTH_TEST_URL  = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
NSYNTH_VALID_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz"
NSYNTH_TRAIN_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"


def download_file(url, target_path):
    print(f"Downloading {url} → {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB

    with open(target_path, "wb") as f, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=target_path.name
    ) as pbar:
        for data in response.iter_content(block_size):
            if not data:
                continue
            f.write(data)
            pbar.update(len(data))

    print("Download complete.")


def extract_tar_gz(archive_path, extract_to):
    print(f"Extracting {archive_path} → {extract_to}")
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()

        # Progress bar + safe extraction (handles upcoming tarfile filter behavior)
        try:
            for _ in tqdm(members, desc="Extracting", unit="file"):
                pass
            tar.extractall(path=extract_to, filter="data")
        except TypeError:
            for member in tqdm(members, desc="Extracting", unit="file"):
                tar.extract(member, path=extract_to)

    print("Extraction complete.")
    archive_path.unlink(missing_ok=True)


def normalize_no_wrapper(nsynth_dir):
    """
    Assumes extracted split folders live directly in nsynth_dir:
      nsynth_dir/nsynth-test, nsynth_dir/nsynth-valid, nsynth_dir/nsynth-train
    Renames/moves them to:
      nsynth_dir/test, nsynth_dir/validation, nsynth_dir/training
    """
    rename_map = {
        "nsynth-test": "test",
        "nsynth-valid": "validation",
        "nsynth-train": "training",
    }

    for src_name, dst_name in rename_map.items():
        src = nsynth_dir / src_name
        dst = nsynth_dir / dst_name

        if not src.exists():
            continue

        if dst.exists():
            raise FileExistsError(f"Target already exists: {dst}")

        print(f"Moving {src} → {dst}")
        shutil.move(str(src), str(dst))


def ensure_split(url, archive_name, final_dirname):
    archive_path = NSYNTH_DIR / archive_name
    final_dir = NSYNTH_DIR / final_dirname

    if final_dir.exists():
        print(f"{final_dirname} already exists at {final_dir}, skipping.")
        return

    download_file(url, archive_path)
    extract_tar_gz(archive_path, NSYNTH_DIR)
    normalize_no_wrapper(NSYNTH_DIR)

    if not final_dir.exists():
        contents = sorted([p.name for p in NSYNTH_DIR.iterdir()])
        raise RuntimeError(
            f"Expected {final_dir} to exist after extraction/rename, but it doesn't.\n"
            f"Contents of {NSYNTH_DIR}: {contents}"
        )


if __name__ == "__main__":
    start = time.time()

    NSYNTH_DIR.mkdir(parents=True, exist_ok=True)

    ensure_split(NSYNTH_TEST_URL,  "nsynth-test.jsonwav.tar.gz",  "test")
    ensure_split(NSYNTH_VALID_URL, "nsynth-valid.jsonwav.tar.gz", "validation")
    ensure_split(NSYNTH_TRAIN_URL, "nsynth-train.jsonwav.tar.gz", "training")

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)")