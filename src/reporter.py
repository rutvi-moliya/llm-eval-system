"""HTML Report Generator:
Generates a self-contained HTML evaluation report using Jinja2 templates.
Reports are saved to the reports/ directory with timestamp in filename.
"""

import os
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import REPORTS_DIR
from src.database import get_run_history
from src.evaluator import EvalRun
from src.regression_detector import RegressionReport


# Report Generator

def generate_report(
    eval_run: EvalRun,
    regression_report: RegressionReport,
    template_dir: str = "templates",
) -> str:
    """
    Generate an HTML evaluation report from eval and regression results.

    Args:
        eval_run: Completed and scored EvalRun.
        regression_report: RegressionReport from regression detector.
        template_dir: Directory containing Jinja2 templates.

    Returns:
        Path to the generated HTML report file.
    """
    # Ensure reports directory exists
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    # Load Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )

    template = env.get_template("report.html")

    # Get run history for trend chart
    run_history = get_run_history(limit=5)

    # Build per-question data combining eval and regression results
    regression_lookup = {
        r.question_id: r for r in regression_report.regression_results
    }

    question_rows = []
    for result in eval_run.results:
        reg = regression_lookup.get(result.question_id)
        question_rows.append({
            "id": result.question_id,
            "question": result.question,
            "expected": result.expected_answer,
            "actual": result.actual_answer,
            "score": result.score,
            "status": result.status,
            "difficulty": result.difficulty,
            "category": result.category,
            "score_delta": reg.score_delta if reg else 0.0,
            "regression_status": reg.status if reg else "NEW",
            "regression_message": reg.message if reg else "",
            "error": result.error,
        })

    # Render template
    html_content = template.render(
        run_id=eval_run.run_id,
        timestamp=eval_run.timestamp,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # Summary stats
        total_questions=eval_run.total_questions,
        completed=eval_run.completed_questions,
        failed_questions=eval_run.failed_questions,
        average_score=eval_run.average_score,

        # Regression summary
        overall_status=regression_report.overall_status,
        baseline_average=regression_report.baseline_average,
        average_delta=regression_report.average_delta,
        pass_count=regression_report.pass_count,
        warn_count=regression_report.warn_count,
        fail_count=regression_report.fail_count,
        new_count=regression_report.new_count,

        # Per-question data
        question_rows=question_rows,

        # Trend data for chart
        run_history=run_history,
    )

    # Save report
    report_filename = f"eval_report_{eval_run.run_id}.html"
    report_path = os.path.join(REPORTS_DIR, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nReport generated: {report_path}")
    return report_path