from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes.bbox import BBoxToMask


class JunnanToolsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BBoxToMask,
        ]


async def comfy_entrypoint() -> JunnanToolsExtension:
    return JunnanToolsExtension()