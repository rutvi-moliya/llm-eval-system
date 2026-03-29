"""
Regression Detector:
Compares current evaluation scores against the previous baseline.
Flags regressions as PASS, WARN, or FAIL based on score drop thresholds.

Thresholds (configurable via .env):
    WARN: score dropped more than 3%
    FAIL: score dropped more than 8%
"""

from dataclasses import dataclass, field
from typing import Optional

from src.config import FAIL_THRESHOLD, WARN_THRESHOLD
from src.evaluator import EvalResult, EvalRun


# Data Classes
@dataclass
class RegressionResult:
    """Regression detection result for a single question."""
    question_id: str
    question: str
    current_score: float
    baseline_score: float
    score_delta: float        # Negative means regression
    status: str               # PASS, WARN, FAIL, NEW
    message: str


@dataclass
class RegressionReport:
    """Full regression detection report for an eval run."""
    run_id: str
    baseline_run_id: Optional[str]
    overall_status: str               # PASS, WARN, FAIL, NO_BASELINE
    current_average: float
    baseline_average: float
    average_delta: float
    total_questions: int
    pass_count: int
    warn_count: int
    fail_count: int
    new_count: int                    # Questions with no baseline
    regression_results: list[RegressionResult] = field(default_factory=list)


# Core Detection Logic
def detect_regressions(
    current_run: EvalRun,
    baseline_run: Optional[EvalRun],
) -> RegressionReport:
    """
    Compare current eval run against baseline and detect regressions.

    If no baseline exists (first ever run), all questions are marked NEW
    and overall status is NO_BASELINE. The current run becomes the baseline.

    Args:
        current_run: The freshly completed and scored EvalRun.
        baseline_run: The previous EvalRun to compare against.
                      None if this is the first run.

    Returns:
        RegressionReport with per-question and overall status.
    """
    print("\nRunning regression detection...")

    # No baseline — first run ever
    if baseline_run is None:
        print("  No baseline found. This run will become the baseline.")
        return _build_no_baseline_report(current_run)

    # Build lookup of baseline scores by question ID
    baseline_scores: dict[str, float] = {
        result.question_id: result.score
        for result in baseline_run.results
    }

    regression_results: list[RegressionResult] = []
    pass_count = warn_count = fail_count = new_count = 0

    for result in current_run.results:
        qid = result.question_id
        current_score = result.score
        baseline_score = baseline_scores.get(qid)

        if baseline_score is None:
            # New question added since last baseline
            reg_result = RegressionResult(
                question_id=qid,
                question=result.question,
                current_score=current_score,
                baseline_score=0.0,
                score_delta=0.0,
                status="NEW",
                message="New question — no baseline to compare against.",
            )
            new_count += 1

        else:
            score_delta = current_score - baseline_score
            drop_pct = abs(score_delta) if score_delta < 0 else 0.0

            if drop_pct >= FAIL_THRESHOLD:
                status = "FAIL"
                message = (
                    f"Score dropped {drop_pct:.1%} "
                    f"(from {baseline_score:.4f} to {current_score:.4f}). "
                    f"Exceeds FAIL threshold of {FAIL_THRESHOLD:.1%}."
                )
                fail_count += 1

            elif drop_pct >= WARN_THRESHOLD:
                status = "WARN"
                message = (
                    f"Score dropped {drop_pct:.1%} "
                    f"(from {baseline_score:.4f} to {current_score:.4f}). "
                    f"Exceeds WARN threshold of {WARN_THRESHOLD:.1%}."
                )
                warn_count += 1

            else:
                status = "PASS"
                if score_delta >= 0:
                    message = f"Score stable or improved ({score_delta:+.4f})."
                else:
                    message = (
                        f"Score dropped {drop_pct:.1%} — within acceptable range."
                    )
                pass_count += 1

            reg_result = RegressionResult(
                question_id=qid,
                question=result.question,
                current_score=current_score,
                baseline_score=baseline_score,
                score_delta=score_delta,
                status=status,
                message=message,
            )

        regression_results.append(reg_result)

    # Overall status
    if fail_count > 0:
        overall_status = "FAIL"
    elif warn_count > 0:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    # Average delta
    baseline_average = baseline_run.average_score
    current_average = current_run.average_score
    average_delta = current_average - baseline_average

    report = RegressionReport(
        run_id=current_run.run_id,
        baseline_run_id=baseline_run.run_id,
        overall_status=overall_status,
        current_average=current_average,
        baseline_average=baseline_average,
        average_delta=average_delta,
        total_questions=len(current_run.results),
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        new_count=new_count,
        regression_results=regression_results,
    )

    _print_summary(report)
    return report


# No Baseline Report
def _build_no_baseline_report(current_run: EvalRun) -> RegressionReport:
    """Build a report when no baseline exists."""
    regression_results = [
        RegressionResult(
            question_id=result.question_id,
            question=result.question,
            current_score=result.score,
            baseline_score=0.0,
            score_delta=0.0,
            status="NEW",
            message="First run — establishing baseline.",
        )
        for result in current_run.results
    ]

    return RegressionReport(
        run_id=current_run.run_id,
        baseline_run_id=None,
        overall_status="NO_BASELINE",
        current_average=current_run.average_score,
        baseline_average=0.0,
        average_delta=0.0,
        total_questions=len(current_run.results),
        pass_count=0,
        warn_count=0,
        fail_count=0,
        new_count=len(current_run.results),
        regression_results=regression_results,
    )

# Summary Printer

def _print_summary(report: RegressionReport) -> None:
    """Print a human-readable regression summary to the console."""
    print("\n" + "=" * 60)
    print("Regression Detection Summary")
    print("=" * 60)
    print(f"  Overall Status  : {report.overall_status}")
    print(f"  Current Average : {report.current_average:.4f}")
    print(f"  Baseline Average: {report.baseline_average:.4f}")
    print(f"  Average Delta   : {report.average_delta:+.4f}")
    print(f"  PASS  : {report.pass_count}")
    print(f"  WARN  : {report.warn_count}")
    print(f"  FAIL  : {report.fail_count}")
    print(f"  NEW   : {report.new_count}")

    if report.fail_count > 0:
        print("\nREGRESSIONS DETECTED:")
        for r in report.regression_results:
            if r.status == "FAIL":
                print(f"{r.question_id}: {r.message}")

    if report.warn_count > 0:
        print("\nWARNINGS:")
        for r in report.regression_results:
            if r.status == "WARN":
                print(f"{r.question_id}: {r.message}")

    print("=" * 60)