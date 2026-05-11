from pathlib import Path
import subprocess
import sys

VOLUNTEERS = ["V002", "V003", "V005", "V006", "V008"]

def run(cmd):
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for vid in VOLUNTEERS:
        print("\n" + "=" * 60)
        print(f"Processing {vid}")
        print("=" * 60)

        run([
            sys.executable,
            "src/debug_volunteer_error_profile.py",
            "--anon_id", vid
        ])

        run([
            sys.executable,
            "src/filter_bad_segments_all.py",
            "--anon_id", vid
        ])

    print("\nDone. Selected volunteers processed.")

if __name__ == "__main__":
    main()