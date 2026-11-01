"""
BeeGuard HiveSense - Dataset Preparation & Ingestion Pipeline
Sources real apicultural imagery from Zenodo VarroaDataset (Record 4085044)
and establishes the production 7-class BeeGuard HiveSense directory structure:
  1. Varroa Mites
  2. American Foulbrood
  3. European Foulbrood
  4. Chalkbrood
  5. Sacbrood
  6. Nosema
  7. Healthy

Strict compliance:
- Ground truth from gt.csv (Zenodo 4085044)
- Zero synthetic image generation (No fabricated diagnostic samples)
- Strict split separation to guarantee untouched held-out evaluation
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    from .config import (
        PROJECT_ROOT,
        DATASET_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        CLASS_NAMES,
        RANDOM_SEED,
    )
except ImportError:
    from config import (
        PROJECT_ROOT,
        DATASET_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        CLASS_NAMES,
        RANDOM_SEED,
    )

ZENODO_RECORD_ID = "4085044"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files"
GT_CSV_URL = f"{ZENODO_BASE_URL}/gt.csv/content"
VAL_ZIP_URL = f"{ZENODO_BASE_URL}/val.zip/content"
TEST_ZIP_URL = f"{ZENODO_BASE_URL}/test.zip/content"
TRAIN_ZIP_URL = f"{ZENODO_BASE_URL}/train.zip/content"


class DatasetPreparer:
    def __init__(self, target_dir: Path = DATASET_DIR):
        self.target_dir = Path(target_dir)
        self.splits = {
            "train": self.target_dir / "train",
            "validation": self.target_dir / "validation",
            "test": self.target_dir / "test",
        }
        self.gt_csv_path = PROJECT_ROOT / "gt.csv"

    def initialize_structure(self):
        """Creates the full 7-class directory hierarchy across all splits."""
        print("Initializing 7-class BeeGuard directory hierarchy...")
        for split_name, split_path in self.splits.items():
            for c in CLASS_NAMES:
                class_dir = split_path / c
                class_dir.mkdir(parents=True, exist_ok=True)
                
        # Write metadata manifest for classes awaiting clinical field capture
        manifest_classes = [
            ("American Foulbrood", "Paenibacillus larvae", "Sunken, punctured dark cappings; dark adherent scales; ropiness test positive."),
            ("European Foulbrood", "Melissococcus plutonius", "Displaced, twisted, yellowish-brown unsealed larvae; sour odor."),
            ("Chalkbrood", "Ascosphaera apis", "Chalk-like white/mottled hard mummified larvae in cells or hive entrance."),
            ("Sacbrood", "Sacbrood virus (SBV)", "Gondola-shaped fluid-filled larval sac; failed pupation."),
            ("Nosema", "Nosema apis / Nosema ceranae", "Microsporidian gut infection; hive entrance dysentery spotting; crawling bees."),
        ]
        for c, pathogen, symptoms in manifest_classes:
            manifest_text = (
                f"Class: {c}\n"
                f"Etiological Agent: {pathogen}\n"
                f"Visual Indicators: {symptoms}\n"
                f"Status: Initialized for BeeGuard 7-class ontology.\n"
                f"Protocol: In accordance with medical and diagnostic guidelines,\n"
                f"synthetic images are strictly prohibited. Field data must be sourced from\n"
                f"molecularly verified clinical apicultural inspections.\n"
            )
            for split_name, split_path in self.splits.items():
                m_file = split_path / c / "_CLASS_MANIFEST.txt"
                if not m_file.exists():
                    m_file.write_text(manifest_text, encoding="utf-8")
        print("  [OK] 7-class directory hierarchy initialized.")

    def download_gt_csv(self) -> Path:
        """Downloads the official Zenodo ground truth annotation file if missing."""
        if not self.gt_csv_path.exists() or self.gt_csv_path.stat().st_size == 0:
            print(f"Downloading ground truth annotations from Zenodo: {GT_CSV_URL}")
            cmd = ["curl.exe", "-s", "-S", "-o", str(self.gt_csv_path), GT_CSV_URL]
            subprocess.run(cmd, check=True)
            print(f"  [OK] gt.csv downloaded successfully ({self.gt_csv_path.stat().st_size} bytes).")
        else:
            print(f"  [OK] gt.csv already present ({self.gt_csv_path.stat().st_size} bytes).")
        return self.gt_csv_path

    def parse_annotations(self) -> Dict[str, Tuple[str, str]]:
        """
        Parses gt.csv lines into mapping:
        filename -> (canonical_split, disease_class)
        where disease_class is 'Varroa Mites' if count > 0 else 'Healthy'.
        """
        self.download_gt_csv()
        mapping = {}
        with open(self.gt_csv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                rel_path = parts[0]  # e.g., val/videos/.../bee_id_...png
                varroa_count = int(parts[1])
                disease_label = "Varroa Mites" if varroa_count > 0 else "Healthy"
                
                # Split prefix in gt.csv
                split_prefix = rel_path.split("/")[0]  # 'train', 'val', 'test'
                filename = Path(rel_path).name
                mapping[filename] = (split_prefix, disease_label)
        print(f"  [OK] Loaded {len(mapping)} annotations from gt.csv.")
        return mapping

    def extract_and_organize_zip(self, zip_path: Path, annotations: Dict[str, Tuple[str, str]], fallback_stratified: bool = False):
        """
        Extracts images from a Zenodo zip file directly into the appropriate
        BeeGuardDataset split/class folder based on ground truth annotations.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip archive not found: {zip_path}")

        print(f"\nProcessing archive: {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.1f} MB)...")
        temp_extract = PROJECT_ROOT / "temp_extract"
        temp_extract.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                file_list = [f for f in zf.namelist() if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                print(f"  Extracting {len(file_list)} image files...")
                zf.extractall(temp_extract)

            extracted_images = list(temp_extract.rglob("*.png")) + list(temp_extract.rglob("*.jpg"))
            print(f"  Found {len(extracted_images)} extracted images. Sorting into dataset splits...")

            # If fallback_stratified is requested (e.g. user only downloaded val.zip which has 1,876 images),
            # partition deterministically into train (70%), validation (15%), test (15%)
            if fallback_stratified:
                print("  Applying deterministic stratified train/val/test partitioning...")
                rng = np.random.RandomState(RANDOM_SEED)
                varroa_imgs = []
                healthy_imgs = []
                for p in extracted_images:
                    meta = annotations.get(p.name)
                    if meta and meta[1] == "Varroa Mites":
                        varroa_imgs.append(p)
                    else:
                        healthy_imgs.append(p)

                rng.shuffle(varroa_imgs)
                rng.shuffle(healthy_imgs)

                def split_list(lst):
                    n = len(lst)
                    n_train = int(n * 0.70)
                    n_val = int(n * 0.15)
                    return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

                v_tr, v_val, v_te = split_list(varroa_imgs)
                h_tr, h_val, h_te = split_list(healthy_imgs)

                assignments = [
                    ("train", "Varroa Mites", v_tr),
                    ("validation", "Varroa Mites", v_val),
                    ("test", "Varroa Mites", v_te),
                    ("train", "Healthy", h_tr),
                    ("validation", "Healthy", h_val),
                    ("test", "Healthy", h_te),
                ]

                moved_counts = {}
                for split_name, class_name, files in assignments:
                    dest_dir = self.splits[split_name] / class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for f in files:
                        dest = dest_dir / f.name
                        shutil.copy2(f, dest)
                    moved_counts[f"{split_name}/{class_name}"] = len(files)
                    
                for k, v in moved_counts.items():
                    print(f"    - {k:<25}: {v} images")

            else:
                # Canonical mapping based on gt.csv split_prefix
                split_map = {"train": "train", "val": "validation", "test": "test"}
                copied = 0
                for img_p in extracted_images:
                    meta = annotations.get(img_p.name)
                    if meta:
                        canon_split, disease_label = meta
                        target_split = split_map.get(canon_split, "train")
                        dest = self.splits[target_split] / disease_label / img_p.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_p, dest)
                        copied += 1
                print(f"  [OK] Sorted and copied {copied} images to canonical splits.")

        finally:
            if temp_extract.exists():
                shutil.rmtree(temp_extract, ignore_errors=True)

    def prepare_dataset(self, zip_filename: str = "val.zip", use_stratified_split: bool = True):
        """High-level orchestration for dataset preparation."""
        self.initialize_structure()
        annotations = self.parse_annotations()
        
        zip_path = PROJECT_ROOT / zip_filename
        if zip_path.exists():
            self.extract_and_organize_zip(zip_path, annotations, fallback_stratified=use_stratified_split)
        else:
            print(f"[!] Archive {zip_filename} not found in {PROJECT_ROOT}.")
            print("    Please ensure the Zenodo archive is downloaded before extraction.")


if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.initialize_structure()
    if (PROJECT_ROOT / "val.zip").exists():
        preparer.prepare_dataset("val.zip", use_stratified_split=True)
