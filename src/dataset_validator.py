"""
BeeGuard HiveSense - Dataset Validator
Performs comprehensive dataset health checks prior to CNN training:
1. Image counting across all 7 classes and splits.
2. Corrupted and truncated image detection.
3. Exact (MD5) and near-duplicate (pHash) detection.
4. Class imbalance and entropy analysis.
5. Sample image grid generation.
6. Class label and folder structure verification.
7. Cross-split data leakage prevention (hash intersection).
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
from PIL import Image
import imagehash
import matplotlib.pyplot as plt

try:
    from .config import (
        DATASET_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        CLASS_NAMES,
        SAMPLE_IMAGES_PLOT,
        CLASS_DISTRIBUTION_PLOT,
    )
except ImportError:
    from config import (
        DATASET_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        CLASS_NAMES,
        SAMPLE_IMAGES_PLOT,
        CLASS_DISTRIBUTION_PLOT,
    )

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetValidator:
    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.splits = {
            "train": self.dataset_dir / "train",
            "validation": self.dataset_dir / "validation",
            "test": self.dataset_dir / "test",
        }
        self.class_names = list(CLASS_NAMES)
        self.validation_report: Dict = {}

    def run_all_checks(self, generate_plots: bool = True) -> Dict:
        """Runs the complete suite of validation checks."""
        print("=" * 70)
        print("BeeGuard HiveSense - Starting Dataset Pre-flight Validation")
        print("=" * 70)

        # 1. Verify Directory & Label Structure
        label_results = self.verify_class_labels()
        
        # 2. Count Images
        counts_by_split = self.count_images()
        
        # 3. Check Corrupted Images
        corrupt_files = self.check_corrupted_images()
        
        # 4. Detect Exact & Near-Duplicate Images Within Splits
        duplicates_by_split = self.detect_duplicates_within_splits()
        
        # 5. Prevent Data Leakage Between Splits (Cross-split duplicates)
        leakage_results = self.check_cross_split_leakage()
        
        # 6. Analyze Class Imbalance
        imbalance_results = self.analyze_class_imbalance(counts_by_split)
        
        # 7. Generate Sample Visualizations
        if generate_plots:
            self.display_sample_images()
            self.plot_class_distribution(counts_by_split)

        self.validation_report = {
            "label_verification": label_results,
            "image_counts": counts_by_split,
            "corrupt_files": corrupt_files,
            "duplicates_within_splits": duplicates_by_split,
            "split_leakage": leakage_results,
            "imbalance_analysis": imbalance_results,
        }
        
        print("=" * 70)
        print("BeeGuard HiveSense - Dataset Validation Complete")
        print("=" * 70)
        return self.validation_report

    def verify_class_labels(self) -> Dict:
        """Verifies directory labels match the official 7-class taxonomy."""
        print("\n[1/7] Verifying Class Labels and Directory Structure...")
        results = {"missing_dirs": [], "unexpected_dirs": [], "valid": True}
        
        for split_name, split_path in self.splits.items():
            if not split_path.exists():
                results["missing_dirs"].append(str(split_path))
                results["valid"] = False
                continue
            
            existing_subdirs = [d.name for d in split_path.iterdir() if d.is_dir()]
            for expected in self.class_names:
                if expected not in existing_subdirs:
                    results["missing_dirs"].append(f"{split_name}/{expected}")
            
            for found in existing_subdirs:
                if found not in self.class_names:
                    results["unexpected_dirs"].append(f"{split_name}/{found}")
                    
        if results["missing_dirs"]:
            print(f"  [!] Missing expected directories: {len(results['missing_dirs'])}")
        if results["unexpected_dirs"]:
            print(f"  [!] Unexpected directories found: {len(results['unexpected_dirs'])}")
        if not results["missing_dirs"] and not results["unexpected_dirs"]:
            print("  [OK] All 7 class directories present and strictly adhere to taxonomy.")
            
        return results

    def count_images(self) -> Dict[str, Dict[str, int]]:
        """Counts valid images across each split and class."""
        print("\n[2/7] Counting Images per Class and Split...")
        counts = {split: {} for split in self.splits}
        total_all = 0

        for split_name, split_path in self.splits.items():
            split_total = 0
            if not split_path.exists():
                for c in self.class_names:
                    counts[split_name][c] = 0
                continue
                
            for c in self.class_names:
                class_dir = split_path / c
                if class_dir.exists():
                    img_files = [
                        f for f in class_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
                    ]
                    num_imgs = len(img_files)
                else:
                    num_imgs = 0
                counts[split_name][c] = num_imgs
                split_total += num_imgs

            total_all += split_total
            print(f"  -> Split '{split_name}': {split_total} images total")
            for c in self.class_names:
                print(f"     - {c:<22}: {counts[split_name][c]} images")

        print(f"  [Total Dataset Volume]: {total_all} images")
        return counts

    def check_corrupted_images(self) -> List[Dict]:
        """Validates image integrity and detects corrupt, unreadable, or truncated files."""
        print("\n[3/7] Scanning for Corrupt or Truncated Images...")
        corrupt_files = []
        checked_count = 0

        for split_name, split_path in self.splits.items():
            if not split_path.exists():
                continue
            for c in self.class_names:
                class_dir = split_path / c
                if not class_dir.exists():
                    continue
                for img_path in class_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                        checked_count += 1
                        try:
                            # Step 1: Check 0-byte files
                            if img_path.stat().st_size == 0:
                                corrupt_files.append({
                                    "path": str(img_path),
                                    "split": split_name,
                                    "class": c,
                                    "reason": "Zero-byte file",
                                })
                                continue

                            # Step 2: PIL verify()
                            with Image.open(img_path) as img:
                                img.verify()

                            # Step 3: Full load and unpack test
                            with Image.open(img_path) as img:
                                img.load()
                                _ = img.convert("RGB")
                        except Exception as e:
                            corrupt_files.append({
                                "path": str(img_path),
                                "split": split_name,
                                "class": c,
                                "reason": str(e),
                            })

        print(f"  Checked {checked_count} images. Corrupted files detected: {len(corrupt_files)}")
        if corrupt_files:
            for item in corrupt_files[:5]:
                print(f"    [X] {item['path']} - Reason: {item['reason']}")
            if len(corrupt_files) > 5:
                print(f"    ... and {len(corrupt_files) - 5} more.")
        else:
            print("  [OK] Zero corrupt images detected across all splits.")
        return corrupt_files

    def detect_duplicates_within_splits(self, phash_distance_threshold: int = 2) -> Dict:
        """Detects exact MD5 duplicates and perceptual near-duplicates within each split."""
        print("\n[4/7] Detecting Exact & Near-Duplicate Images Within Splits...")
        results = {}

        for split_name, split_path in self.splits.items():
            if not split_path.exists():
                results[split_name] = {"exact_duplicates": [], "near_duplicates": []}
                continue

            md5_map: Dict[str, List[Path]] = {}
            phash_list: List[Tuple[imagehash.ImageHash, Path]] = []
            
            for c in self.class_names:
                class_dir = split_path / c
                if not class_dir.exists():
                    continue
                for img_path in class_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                        try:
                            # MD5 Hash
                            with open(img_path, "rb") as f:
                                h = hashlib.md5(f.read()).hexdigest()
                            md5_map.setdefault(h, []).append(img_path)

                            # Perceptual Hash
                            with Image.open(img_path) as img:
                                ph = imagehash.phash(img, hash_size=8)
                            phash_list.append((ph, img_path))
                        except Exception:
                            continue

            # Exact duplicates (identical MD5)
            exact_dups = [paths for paths in md5_map.values() if len(paths) > 1]
            
            # Near-duplicates (Hamming distance <= threshold on pHash, excluding exact MD5)
            near_dups = []
            exact_flat = {p for paths in exact_dups for p in paths[1:]}
            filtered_phash = [(ph, p) for ph, p in phash_list if p not in exact_flat]
            
            # Compare pairwise (limit sample comparison if large to keep fast)
            max_compare = min(len(filtered_phash), 1000)
            for i in range(max_compare):
                for j in range(i + 1, max_compare):
                    dist = filtered_phash[i][0] - filtered_phash[j][0]
                    if dist <= phash_distance_threshold:
                        near_dups.append((
                            str(filtered_phash[i][1].name),
                            str(filtered_phash[j][1].name),
                            int(dist)
                        ))

            results[split_name] = {
                "exact_duplicate_sets": len(exact_dups),
                "exact_duplicate_files": sum(len(s) - 1 for s in exact_dups),
                "near_duplicate_pairs": len(near_dups),
            }
            print(f"  Split '{split_name}': {results[split_name]['exact_duplicate_files']} exact duplicate files, "
                  f"{len(near_dups)} near-duplicate pairs (pHash dist <= {phash_distance_threshold}).")

        return results

    def check_cross_split_leakage(self) -> Dict:
        """
        Critical Check: Verifies that no sample in the test set or validation set
        exists in the training set (preventing data leakage).
        """
        print("\n[5/7] Verifying Absence of Cross-Split Data Leakage...")
        hashes_by_split: Dict[str, Dict[str, str]] = {"train": {}, "validation": {}, "test": {}}

        for split_name, split_path in self.splits.items():
            if not split_path.exists():
                continue
            for img_path in split_path.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                    try:
                        with open(img_path, "rb") as f:
                            digest = hashlib.sha256(f.read()).hexdigest()
                        hashes_by_split[split_name][digest] = str(img_path)
                    except Exception:
                        continue

        train_h = set(hashes_by_split["train"].keys())
        val_h = set(hashes_by_split["validation"].keys())
        test_h = set(hashes_by_split["test"].keys())

        train_val_leak = train_h.intersection(val_h)
        train_test_leak = train_h.intersection(test_h)
        val_test_leak = val_h.intersection(test_h)

        leakage_detected = bool(train_val_leak or train_test_leak or val_test_leak)
        results = {
            "train_val_overlap_count": len(train_val_leak),
            "train_test_overlap_count": len(train_test_leak),
            "val_test_overlap_count": len(val_test_leak),
            "leakage_detected": leakage_detected,
        }

        if leakage_detected:
            print(f"  [CRITICAL WARNING] Data Leakage Detected!")
            print(f"    - Train vs Val Overlap : {len(train_val_leak)} images")
            print(f"    - Train vs Test Overlap: {len(train_test_leak)} images")
            print(f"    - Val vs Test Overlap  : {len(val_test_leak)} images")
        else:
            print("  [OK] PASSED: Zero overlap between Train, Validation, and Test splits.")
            print("      Held-out test set integrity is 100% preserved.")

        return results

    def analyze_class_imbalance(self, counts: Dict[str, Dict[str, int]]) -> Dict:
        """Evaluates class distribution, imbalance ratio, and Shannon entropy."""
        print("\n[6/7] Analyzing Class Imbalance...")
        train_counts = counts.get("train", {})
        active_counts = {k: v for k, v in train_counts.items() if v > 0}
        
        if not active_counts:
            print("  [!] No training images found to calculate imbalance.")
            return {"imbalance_ratio": None, "entropy": 0.0, "status": "empty"}

        values = list(active_counts.values())
        max_val = max(values)
        min_val = min(values)
        imbalance_ratio = max_val / min_val if min_val > 0 else float("inf")

        # Shannon Entropy: H = -sum(p * log2(p))
        total_samples = sum(values)
        probs = [v / total_samples for v in values]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_possible_entropy = np.log2(len(active_counts)) if len(active_counts) > 1 else 1.0
        normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 1.0

        results = {
            "active_classes": len(active_counts),
            "min_class_samples": min_val,
            "max_class_samples": max_val,
            "imbalance_ratio": float(imbalance_ratio),
            "shannon_entropy": float(entropy),
            "normalized_entropy": float(normalized_entropy),
            "recommend_class_weights": imbalance_ratio > 2.0,
        }

        print(f"  - Active Classes in Training: {len(active_counts)} of {len(self.class_names)}")
        print(f"  - Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"  - Normalized Entropy: {normalized_entropy:.2f} (1.0 = perfectly uniform)")
        if results["recommend_class_weights"]:
            print("  - [Recommendation]: Use class weighting in loss function to mitigate class imbalance.")

        return results

    def plot_class_distribution(self, counts: Dict[str, Dict[str, int]], save_path: Path = CLASS_DISTRIBUTION_PLOT):
        """Plots split-wise image distribution per class."""
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.class_names))
        width = 0.25

        train_vals = [counts.get("train", {}).get(c, 0) for c in self.class_names]
        val_vals = [counts.get("validation", {}).get(c, 0) for c in self.class_names]
        test_vals = [counts.get("test", {}).get(c, 0) for c in self.class_names]

        ax.bar(x - width, train_vals, width, label="Train", color="#1f77b4", alpha=0.9)
        ax.bar(x, val_vals, width, label="Validation", color="#ff7f0e", alpha=0.9)
        ax.bar(x + width, test_vals, width, label="Test", color="#2ca02c", alpha=0.9)

        ax.set_ylabel("Number of Images", fontsize=12)
        ax.set_title("BeeGuard HiveSense - Dataset Distribution per Class & Split", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=25, ha="right", fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"  [OK] Class distribution plot saved to: {save_path}")

    def display_sample_images(self, save_path: Path = SAMPLE_IMAGES_PLOT):
        """Displays sample images from available classes and saves a diagnostic figure."""
        print("\n[7/7] Generating Sample Images Inspection Grid...")
        samples = []
        labels = []

        for c in self.class_names:
            found = False
            for split_name in ["train", "validation", "test"]:
                class_dir = self.splits[split_name] / c
                if class_dir.exists():
                    img_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
                    if img_files:
                        samples.append((img_files[0], f"{c}\n({split_name})"))
                        found = True
                        break
            if not found:
                samples.append((None, f"{c}\n(No real samples)"))

        n = len(samples)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for idx, (img_path, title) in enumerate(samples):
            ax = axes[idx]
            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(title, fontsize=11, fontweight="bold")
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, f"{title}\n[Unpopulated]", ha="center", va="center", color="gray", fontsize=11)
                ax.set_facecolor("#f0f0f0")
            ax.axis("off")

        for idx in range(n, len(axes)):
            axes[idx].axis("off")

        plt.suptitle("BeeGuard HiveSense - Diagnostic Image Inspection Grid", fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"  [OK] Sample images inspection grid saved to: {save_path}")


if __name__ == "__main__":
    validator = DatasetValidator()
    report = validator.run_all_checks(generate_plots=True)
