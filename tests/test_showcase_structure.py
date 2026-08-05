from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


PROJECTS = {
    "SurgeryStats": "bash scripts/run_summary.sh",
    "ImageQuality": "bash scripts/run_all.sh",
    "ClinicalBenchmarks": "bash scripts/run_summary.sh",
    "CardiacSignals": "bash scripts/run_question.sh q1",
}


def test_readme_quick_start_index_points_to_each_project():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README_en.md").read_text(encoding="utf-8")
    assert "## 快速上手索引" in readme
    assert "## 简历亮点" in readme
    assert "## 复现边界" in readme
    assert "## Resume Highlights" in english
    assert "## Reproducibility Boundaries" in english
    for project, command in PROJECTS.items():
        assert project in readme
        assert command in readme
    assert (ROOT / "README_en.md").is_file()


def test_showcase_preview_asset_exists_and_is_valid_svg():
    image = ROOT / "docs" / "images" / "clinical-analytics-preview.svg"
    assert image.is_file()
    ET.parse(image)


def test_project_folder_names_are_pascal_case():
    for project in PROJECTS:
        assert "_" not in project
        assert project[0].isupper()
        assert (ROOT / project).is_dir()


def test_subproject_readmes_have_quick_run_and_result_snapshot():
    for project in PROJECTS:
        readme = (ROOT / project / "README.md").read_text(encoding="utf-8")
        assert "## 快速运行" in readme
        assert "## 结果快照" in readme
        assert (ROOT / project / "README_en.md").is_file()


def test_shell_entrypoints_are_syntax_valid():
    scripts = sorted(ROOT.glob("*/scripts/*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_notebook_uses_function_named_dataset_placeholders():
    notebook = (ROOT / "ClinicalBenchmarks" / "analysis.ipynb").read_text(
        encoding="utf-8"
    )
    assert "cw3_a" not in notebook
    assert "cw3_b" not in notebook
    assert "surgical_motion_data" in notebook
    assert "covid_ct_features" in notebook
