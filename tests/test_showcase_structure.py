from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


PROJECTS = {
    "surgical_skill_statistics": "bash scripts/run_summary.sh",
    "toe_image_quality_assessment": "bash scripts/run_all.sh",
    "clinical_ml_benchmarks": "bash scripts/run_summary.sh",
    "ecg_signal_mining": "bash scripts/run_question.sh q1",
}


def test_readme_quick_start_index_points_to_each_project():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Quick Start Index" in readme
    for project, command in PROJECTS.items():
        assert project in readme
        assert command in readme


def test_subproject_readmes_have_quick_run_and_result_snapshot():
    for project in PROJECTS:
        readme = (ROOT / project / "README.md").read_text(encoding="utf-8")
        assert "## Quick Run" in readme
        assert "## Result Snapshot" in readme


def test_shell_entrypoints_are_syntax_valid():
    scripts = sorted(ROOT.glob("*/scripts/*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_notebook_uses_function_named_dataset_placeholders():
    notebook = (ROOT / "clinical_ml_benchmarks" / "analysis.ipynb").read_text(
        encoding="utf-8"
    )
    assert "cw3_a" not in notebook
    assert "cw3_b" not in notebook
    assert "surgical_motion_data" in notebook
    assert "covid_ct_features" in notebook
