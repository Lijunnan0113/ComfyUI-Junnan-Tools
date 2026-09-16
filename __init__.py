from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes.bbox import BBoxToMask
from .nodes.minimax_h3_skills import MiniMaxH3SkillsTool


class JunnanToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BBoxToMask,
            MiniMaxH3SkillsTool,
        ]


async def comfy_entrypoint() -> JunnanToolsExtension:
    return JunnanToolsExtension()