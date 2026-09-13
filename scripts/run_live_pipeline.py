from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_stage(name, script):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    script_path = SCRIPTS / script

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT)
    )

    if result.returncode != 0:
        print()
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        return False

    return True


def main():
    print("=" * 70)
    print("FORECASTMITR - COMPLETE LIVE PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # STAGE 1
    # Download latest usable GEFS cycle
    # ---------------------------------------------------------
    if not run_stage(
        "STAGE 1/2 - DOWNLOADING LATEST GEFS",
        "download_latest_gefs.py"
    ):
        sys.exit(1)

    # ---------------------------------------------------------
    # STAGE 2
    # Run multi-lead ForecastMitr inference
    # ---------------------------------------------------------
    if not run_stage(
        "STAGE 2/2 - RUNNING MULTI-LEAD FORECASTMITR INFERENCE",
        "run_live_forecastmitr_timeline.py"
    ):
        sys.exit(1)

    # ---------------------------------------------------------
    # Verify output
    # ---------------------------------------------------------
    output_file = ROOT / "data" / "forecastmitr_live_timeline.csv"

    print()
    print("=" * 70)

    if output_file.exists():
        print("FORECASTMITR LIVE PIPELINE COMPLETE")
        print("=" * 70)
        print()
        print(f"Timeline generated:")
        print(output_file)
        print()
    else:
        print("WARNING: Pipeline completed but timeline CSV was not found.")
        print(output_file)
        print()


if __name__ == "__main__":
    main()