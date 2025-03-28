"""
@file   collect.py
@brief  Collect stats from CGP circuit optimizations and save them all to a CSV file.
@author Jan Zdeněk (xzdene01)
@date   27/3/2025

@project Aproximace násobiček pomocí CGP
@course  BIN - Biologií inspirované počítače
@faculty Faculty of Information Technology, Brno University of Technology
"""

import os
import json
import argparse
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot stats from CGP circuit optimizations.")
    parser.add_argument("-s", "--source_dir", type=str, required=True, help="The directory with the logs to collect stats from.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="The directory to save the stats to.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all metadata files in log directory
    metadata_files = []
    for root, _, files in os.walk(args.source_dir):
        for file in files:
            if file == "metadata.json":
                metadata_files.append(os.path.join(root, file))
    print("Found", len(metadata_files), "metadata files.")

    # Read all metadata files
    df = pd.DataFrame()
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        new_row = {
            "log_file": metadata_file,
            "src_file": os.path.basename(metadata["file"]),
            "criterion": metadata["criterion"],
            "population": metadata["population"],
            "generations": metadata["epochs"],
            "mut_raate": metadata["mutation_rate"],
            "tau": metadata["tau"],
            "pretrain": metadata["pretrain"],
            "finetune": metadata["finetune"],
            "best_area": metadata["best_area"],
            "best_error": metadata["best_error"],
            "run_log": os.path.join(os.path.dirname(metadata_file), "log.csv"),
        }

        new_row = pd.DataFrame([new_row])
        df = pd.concat([df, new_row], ignore_index=True)

    # Save the dataframe to a CSV file
    df.to_csv(os.path.join(args.output_dir, "logs.csv"), index=False)
    print("Saved metadata to CSV file.")


if __name__ == "__main__":
    main()