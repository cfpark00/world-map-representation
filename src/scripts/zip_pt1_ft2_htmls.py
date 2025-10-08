#!/usr/bin/env python
"""Collect and zip all pt1_ft2-X PCA timeline HTML files."""

import sys
import shutil
import tempfile
from pathlib import Path
import zipfile

def main():
    # Define parameters
    x_range = range(1, 22)  # 1 to 21
    y_suffixes = ["", "_raw", "_na"]

    # Base path for data directory
    base_path = Path("data/experiments")

    # Collect all required file paths
    required_files = []
    for x in x_range:
        for y_suffix in y_suffixes:
            # Construct the path
            html_path = base_path / f"pt1_ft2-{x}" / "analysis_higher" / "distance_firstcity_last_and_trans_l5" / f"pca_timeline{y_suffix}" / "pca_3d_timeline.html"
            required_files.append(html_path)

    # Check all files exist before proceeding
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        print(f"ERROR: Missing {len(missing_files)} files:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        sys.exit(1)

    print(f"Found all {len(required_files)} required HTML files")

    # Create temporary directory for collecting files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy all files with organized structure
        for x in x_range:
            for y_suffix in y_suffixes:
                # Source path
                html_path = base_path / f"pt1_ft2-{x}" / "analysis_higher" / "distance_firstcity_last_and_trans_l5" / f"pca_timeline{y_suffix}" / "pca_3d_timeline.html"

                # Destination path with clear naming
                y_name = "default" if y_suffix == "" else y_suffix[1:]  # Remove leading underscore
                dest_filename = f"pt1_ft2-{x}_pca_timeline_{y_name}.html"
                dest_path = temp_path / dest_filename

                # Copy file
                shutil.copy2(html_path, dest_path)
                print(f"Copied: pt1_ft2-{x} pca_timeline{y_suffix}")

        # Create zip file in home directory
        home_wm1 = Path.home() / "WM_1"
        home_wm1.mkdir(parents=True, exist_ok=True)
        zip_path = home_wm1 / "all_pt1_ft2_pca_htmls.zip"

        # Create zip archive
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_path.glob("*.html"):
                zipf.write(file, file.name)

        print(f"\nSuccessfully created zip file: {zip_path}")
        print(f"Total files zipped: {len(list(temp_path.glob('*.html')))}")

if __name__ == "__main__":
    main()