import json
import os
from pathlib import Path
from typing import Any

from comfy_api.latest import io


DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parent / "skills"
TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".csv",
}
MAX_TEXT_CHARS = 80_000
_REGISTERED_ROOTS: list[Path] = []
_TOOL_HOOKS = ["load_h3_skill", "read_h3_skill_resource"]


def _register_root(root: Path) -> None:
    if root not in _REGISTERED_ROOTS:
        _REGISTERED_ROOTS.append(root)


def _search_roots() -> list[Path]:
    return _REGISTERED_ROOTS or [DEFAULT_SKILLS_ROOT]


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    try:
        end_index = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration:
        return {}

    header = lines[1:end_index]
    metadata: dict[str, str] = {}
    index = 0

    while index < len(header):
        line = header[index]
        if line.startswith("name:"):
            metadata["name"] = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("description:"):
            value = line.split(":", 1)[1].strip()
            if value in {"|", ">"}:
                description, index = _read_block_scalar(header, index + 1)
                metadata["description"] = description
                continue
            metadata["description"] = value.strip("'\"")
        index += 1

    return metadata


def _read_block_scalar(header: list[str], start: int) -> tuple[str, int]:
    collected: list[str] = []
    index = start

    while index < len(header):
        line = header[index]
        if line.strip() and not line.startswith((" ", "\t")):
            break
        collected.append(line.strip())
        index += 1

    return " ".join(part for part in collected if part), index


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"MiniMax H3 skills 目录不存在：{root}")
    return resolved


def discover_skills(root: Path) -> dict[str, dict[str, Any]]:
    resolved = _resolve_root(root)
    registry: dict[str, dict[str, Any]] = {}

    for folder in sorted(resolved.iterdir()):
        if not folder.is_dir():
            continue

        en_file = folder / "SKILL.md"
        cn_file = folder / "SKILL.cn.md"
        if not en_file.exists() and not cn_file.exists():
            continue

        en_meta = _parse_frontmatter(_read_text(en_file)) if en_file.exists() else {}
        cn_meta = _parse_frontmatter(_read_text(cn_file)) if cn_file.exists() else {}
        skill_name = en_meta.get("name") or cn_meta.get("name") or folder.name

        if skill_name in registry:
            previous = registry[skill_name]["folder"].name
            raise ValueError(
                f"技能名称重复：{skill_name!r} 同时来自 {previous} 和 {folder.name}。"
            )

        registry[skill_name] = {
            "name": skill_name,
            "description": (
                cn_meta.get("description")
                or en_meta.get("description")
                or f"MiniMax H3 skill: {skill_name}"
            ),
            "folder": folder,
            "en_file": en_file if en_file.exists() else None,
            "cn_file": cn_file if cn_file.exists() else None,
        }

    return registry


def _get_skill(skill_name: str) -> dict[str, Any]:
    available: set[str] = set()
    for root in _search_roots():
        try:
            registry = discover_skills(root)
        except (OSError, ValueError):
            continue
        if skill_name in registry:
            return registry[skill_name]
        available.update(registry)

    raise ValueError(
        f"找不到技能 {skill_name!r}。可用技能：{', '.join(sorted(available)) or '无'}"
    )


def _list_text_resources(skill_folder: Path) -> list[str]:
    resources = []
    for path in sorted(skill_folder.rglob("*")):
        if (
            path.is_file()
            and path.name not in {"SKILL.md", "SKILL.cn.md"}
            and path.suffix.lower() in TEXT_EXTENSIONS
        ):
            resources.append(path.relative_to(skill_folder).as_posix())
    return resources


def read_skill_resource(skill_folder: Path, relative_path: str) -> str:
    target = (skill_folder.resolve() / str(relative_path).strip()).resolve()
    try:
        target.relative_to(skill_folder.resolve())
    except ValueError as error:
        raise ValueError("拒绝读取：资源路径超出了技能目录。") from error

    if not target.is_file():
        raise FileNotFoundError(f"资源不存在：{relative_path}")
    if target.suffix.lower() not in TEXT_EXTENSIONS:
        raise ValueError(f"资源 {relative_path} 不是受支持的文本文件。")
    return _truncate(_read_text(target), "资源内容")


def _truncate(content: str, label: str) -> str:
    if len(content) <= MAX_TEXT_CHARS:
        return content
    return content[:MAX_TEXT_CHARS] + f"\n\n[{label}因长度限制被截断]"


def load_h3_skill(skill_name: str, task: str, language: str = "auto") -> str:
    skill = _get_skill(skill_name)
    language = str(language or "auto").lower()
    if language in {"en", "english"}:
        selected_file = skill["en_file"] or skill["cn_file"]
    else:
        selected_file = skill["cn_file"] or skill["en_file"]

    if selected_file is None:
        return f"技能 {skill_name} 没有可读取的 SKILL 文件。"

    resources = _list_text_resources(skill["folder"])
    resources_text = (
        "\n".join(f"- {item}" for item in resources) if resources else "无额外文本资源"
    )
    return (
        "已加载 MiniMax H3 Skill。\n\n"
        f"技能名称：\n{skill_name}\n\n"
        f"技能文件：\n{selected_file.name}\n\n"
        f"用户当前任务：\n<user_task>\n{task}\n</user_task>\n\n"
        f"技能正文：\n<skill_document>\n{_truncate(_read_text(selected_file), 'SKILL 内容')}"
        "\n</skill_document>\n\n"
        f"该技能目录中的可读取文本资源：\n<available_resources>\n{resources_text}"
        "\n</available_resources>"
    )


def read_h3_skill_resource(skill_name: str, relative_path: str) -> str:
    skill = _get_skill(skill_name)
    try:
        content = read_skill_resource(skill["folder"], relative_path)
    except (FileNotFoundError, ValueError) as error:
        return str(error)

    return (
        "已读取技能资源。\n\n"
        f"技能：\n{skill_name}\n\n"
        f"资源：\n{relative_path}\n\n"
        f"<skill_resource>\n{content}\n</skill_resource>"
    )


class MiniMaxH3SkillsTool(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="JunnanMiniMaxH3Skills",
            display_name="MiniMax H3 Skills",
            category="Junnan Tools/LLM",
            description="Discover MiniMax H3 skills and expose them as LLM tool definitions.",
            inputs=[
                io.String.Input(
                    "skills_root",
                    display_name="Skills Directory",
                    default=str(DEFAULT_SKILLS_ROOT),
                ),
                io.Boolean.Input("is_enable", display_name="Enable", default=True),
            ],
            outputs=[io.String.Output("tool")],
        )

    @classmethod
    def execute(cls, skills_root: str, is_enable: bool = True) -> io.NodeOutput:
        if not is_enable:
            return io.NodeOutput(None)

        root = _resolve_root(Path(os.path.expandvars(skills_root)))
        registry = discover_skills(root)
        if not registry:
            raise RuntimeError(f"目录中没有发现 SKILL.md：{skills_root}")

        _register_root(root)

        skill_names = sorted(registry)
        catalog = "\n".join(
            f"- {name}: {registry[name]['description']}" for name in skill_names
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "load_h3_skill",
                    "description": (
                        "加载最适合当前任务的 MiniMax H3 Skill。"
                        "当用户请求与下列技能之一明确匹配时调用。"
                        "普通问答不要调用。调用后必须遵循技能正文，"
                        "并按需读取其 references。\n\n可用技能：\n" + catalog
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "enum": skill_names,
                                "description": "需要加载的技能名称",
                            },
                            "task": {
                                "type": "string",
                                "description": "用户当前的完整任务。",
                            },
                            "language": {
                                "type": "string",
                                "enum": ["auto", "zh", "en"],
                                "description": "技能文档语言。中文用户优先 zh。",
                                "default": "auto",
                            },
                        },
                        "required": ["skill_name", "task"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_h3_skill_resource",
                    "description": "读取已加载技能目录中的文本资源。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "enum": skill_names,
                            },
                            "relative_path": {
                                "type": "string",
                                "description": "相对于技能目录的资源路径。",
                            },
                        },
                        "required": ["skill_name", "relative_path"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        return io.NodeOutput(json.dumps(tools, ensure_ascii=False))