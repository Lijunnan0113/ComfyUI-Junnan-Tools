from pathlib import Path

import pytest

from ComfyUI_Junnan_Tools.nodes.minimax_h3_skills import (
    _parse_frontmatter,
    _REGISTERED_ROOTS,
    load_h3_skill,
    discover_skills,
    read_skill_resource,
)


def test_parse_frontmatter_prefers_multiline_description():
    text = "\n".join(
        [
            "---",
            "name: example-skill",
            "description: |",
            "  First line.",
            "  Second line.",
            "---",
            "# Skill",
        ]
    )

    assert _parse_frontmatter(text) == {
        "name": "example-skill",
        "description": "First line. Second line.",
    }


def test_discover_skills_prefers_chinese_metadata_and_lists_resources(tmp_path: Path):
    skill_folder = tmp_path / "demo"
    skill_folder.mkdir()
    (skill_folder / "SKILL.md").write_text(
        "---\nname: demo\ndescription: English\n---\n", encoding="utf-8"
    )
    (skill_folder / "SKILL.cn.md").write_text(
        "---\nname: demo\ndescription: 中文说明\n---\n", encoding="utf-8"
    )
    (skill_folder / "references").mkdir()
    (skill_folder / "references" / "guide.txt").write_text(
        "guide", encoding="utf-8"
    )

    registry = discover_skills(tmp_path)

    assert registry["demo"]["description"] == "中文说明"
    assert (
        read_skill_resource(registry["demo"]["folder"], "references/guide.txt")
        == "guide"
    )


def test_read_skill_resource_rejects_path_escape(tmp_path: Path):
    skill_folder = tmp_path / "demo"
    skill_folder.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="超出了技能目录"):
        read_skill_resource(skill_folder, "../outside.txt")


def test_load_h3_skill_returns_document_and_resource_catalog(tmp_path: Path):
    skill_folder = tmp_path / "demo"
    skill_folder.mkdir()
    (skill_folder / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo\n---\nBody", encoding="utf-8"
    )
    (skill_folder / "guide.txt").write_text("guide", encoding="utf-8")
    _REGISTERED_ROOTS.clear()
    _REGISTERED_ROOTS.append(tmp_path)

    result = load_h3_skill("demo", "生成一个测试提示词", language="en")

    assert "Body" in result
    assert "guide.txt" in result