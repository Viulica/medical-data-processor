#!/usr/bin/env python3
"""CLI tool to run CPT prediction with the UNI model."""

import sys
import os
import argparse

# Ensure backend is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import predict_cpt_background, job_status, ProcessingJob


def main():
    parser = argparse.ArgumentParser(description="Run CPT code prediction")
    parser.add_argument("csv_file", help="Path to CSV/XLSX file with 'Procedure Description' column")
    parser.add_argument("--client", default="uni", choices=["uni", "sio-stl", "gap-fin", "apo-utp"], help="Client model (default: uni)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)

    job_id = "cli-manual"
    job_status[job_id] = ProcessingJob(job_id)

    print(f"Running CPT prediction on: {args.csv_file}")
    print(f"Client model: {args.client}")
    print()

    predict_cpt_background(job_id, args.csv_file, client=args.client)

    job = job_status[job_id]
    if job.status == "completed":
        print(f"Done! {job.message}")
        print(f"CSV:  {job.result_file}")
        print(f"XLSX: {job.result_file_xlsx}")
    else:
        print(f"Failed: {job.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
