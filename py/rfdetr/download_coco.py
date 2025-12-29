"""Download COCO 2017 dataset."""

import os
import urllib.request
import zipfile
from pathlib import Path

COCO_URLS = {
    # "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    print(f"  To: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(
                f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"  Done extracting to: {dest_dir}")


def main():
    # datasets dir is at repo root, not inside py/rfdetr
    base_dir = Path(__file__).parent.parent.parent / "datasets" / "coco"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("COCO 2017 Dataset Downloader")
    print("=" * 50)
    print(f"Destination: {base_dir}")
    print()

    for name, url in COCO_URLS.items():
        print(f"[{name}]")
        zip_name = url.split("/")[-1]
        zip_path = base_dir / zip_name

        download_file(url, zip_path)

        # Check if already extracted
        if name == "annotations":
            extracted_dir = base_dir / "annotations"
        else:
            extracted_dir = base_dir / name

        if extracted_dir.exists() and any(extracted_dir.iterdir()):
            print(f"  Already extracted: {extracted_dir}")
        else:
            extract_zip(zip_path, base_dir)

        print()

    print("=" * 50)
    print("COCO 2017 dataset download complete!")
    print(f"Location: {base_dir}")


if __name__ == "__main__":
    main()
