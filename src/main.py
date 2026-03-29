"""
Main Pipeline Entry Point
Orchestrates the full evaluation pipeline:
1. Validate golden dataset
2. Load RAG system
3. Run evaluation on all 25 questions
4. Score results using semantic similarity
5. Detect regressions against baseline
6. Save results to SQLite
7. Generate HTML report
8. Exit with code 1 if regressions detected (fails CI pipeline)
"""

import sys

from src.config import (
    DATABASE_PATH,
    GOLDEN_DATASET_PATH,
    REPORTS_DIR,
    FAIL_THRESHOLD,
    WARN_THRESHOLD,
)
from src.database import get_last_run, save_run
from src.evaluator import run_evaluation
from src.regression_detector import detect_regressions
from src.reporter import generate_report
from src.scorer import score_eval_run
from scripts.validate_dataset import validate_dataset


def main() -> None:
    """
    Run the full LLM regression detection pipeline.

    Exit codes:
        0 — Pipeline passed (PASS or WARN status)
        1 — Pipeline failed (FAIL status — regressions detected)
        2 — Pipeline error (dataset invalid, vector store missing, etc.)
    """
    print("=" * 60)
    print("LLM Regression Detection System")
    print("=" * 60)
    print(f"Dataset     : {GOLDEN_DATASET_PATH}")
    print(f"Reports dir : {REPORTS_DIR}")
    print(f"Database    : {DATABASE_PATH}")
    print(f"Thresholds  : WARN={WARN_THRESHOLD:.0%}  FAIL={FAIL_THRESHOLD:.0%}")
    print("=" * 60)

    # Step 1 - Validate golden dataset
    print("\n[1/6] Validating golden dataset...")
    is_valid, errors = validate_dataset(GOLDEN_DATASET_PATH)

    if not is_valid:
        print(f"\nDataset validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        print("\nFix the dataset and re-run.")
        sys.exit(2)

    print("Dataset valid.")

    # Step 2- Load baseline from database
    print("\n[2/6] Loading baseline from database...")
    baseline_run = get_last_run()

    if baseline_run is None:
        print("No baseline found. This will be the first run.")
    else:
        print(f"Baseline loaded: run '{baseline_run.run_id}' "
              f"(avg score: {baseline_run.average_score:.4f})")

    # Step 3 - Run evaluation
    print("\n[3/6] Running evaluation pipeline...")
    try:
        eval_run = run_evaluation(dataset_path=GOLDEN_DATASET_PATH)
    except RuntimeError as e:
        print(f"\nEvaluation failed: {e}")
        sys.exit(2)

    # Step 4 - Score results
    print("\n[4/6] Scoring results...")
    eval_run = score_eval_run(eval_run)

    # Step 5 - Detect regressions
    print("\n[5/6] Detecting regressions...")
    regression_report = detect_regressions(eval_run, baseline_run)

    # Step 6 - Save results and generate report
    print("\n[6/6] Saving results and generating report...")
    save_run(eval_run)
    report_path = generate_report(eval_run, regression_report)

    # Final summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"  Overall Status  : {regression_report.overall_status}")
    print(f"  Average Score   : {eval_run.average_score:.4f}")
    print(f"  PASS            : {regression_report.pass_count}")
    print(f"  WARN            : {regression_report.warn_count}")
    print(f"  FAIL            : {regression_report.fail_count}")
    print(f"  Report          : {report_path}")
    print("=" * 60)

    # Exit with code 1 if regressions detected - fails CI pipeline
    if regression_report.overall_status == "FAIL":
        print("\nRegressions detected. Pipeline FAILED.")
        sys.exit(1)

    print("\nPipeline passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()