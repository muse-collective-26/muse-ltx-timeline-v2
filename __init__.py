from .infinite_sampler_v7 import NODE_CLASS_MAPPINGS as _N7, NODE_DISPLAY_NAME_MAPPINGS as _D7
from .muse_director_v1 import NODE_CLASS_MAPPINGS as _NM1, NODE_DISPLAY_NAME_MAPPINGS as _DM1
from .muse_director_v2 import NODE_CLASS_MAPPINGS as _NM2, NODE_DISPLAY_NAME_MAPPINGS as _DM2
from .muse_director_v2_5 import NODE_CLASS_MAPPINGS as _NM2_5, NODE_DISPLAY_NAME_MAPPINGS as _DM2_5
from .muse_director_v3 import NODE_CLASS_MAPPINGS as _NM3, NODE_DISPLAY_NAME_MAPPINGS as _DM3
from .muse_director_v4 import NODE_CLASS_MAPPINGS as _NM4, NODE_DISPLAY_NAME_MAPPINGS as _DM4
from .muse_director_v5 import NODE_CLASS_MAPPINGS as _NM5, NODE_DISPLAY_NAME_MAPPINGS as _DM5
from .muse_guide import NODE_CLASS_MAPPINGS as _NMG, NODE_DISPLAY_NAME_MAPPINGS as _DMG
from .muse_person_segmenter import NODE_CLASS_MAPPINGS as _NPS, NODE_DISPLAY_NAME_MAPPINGS as _DPS
from .muse_seed_scout import NODE_CLASS_MAPPINGS as _NSS, NODE_DISPLAY_NAME_MAPPINGS as _DSS
from .muse_face_lock import NODE_CLASS_MAPPINGS as _NFL, NODE_DISPLAY_NAME_MAPPINGS as _DFL
from .muse_ambient_audio import NODE_CLASS_MAPPINGS as _NAA, NODE_DISPLAY_NAME_MAPPINGS as _DAA
from .muse_prompt_splitter import NODE_CLASS_MAPPINGS as _NPSp, NODE_DISPLAY_NAME_MAPPINGS as _DPSp

NODE_CLASS_MAPPINGS = {**_N7, **_NM1, **_NM2, **_NM2_5, **_NM3, **_NM4, **_NM5, **_NMG, **_NPS, **_NSS, **_NFL, **_NAA, **_NPSp}
NODE_DISPLAY_NAME_MAPPINGS = {**_D7, **_DM1, **_DM2, **_DM2_5, **_DM3, **_DM4, **_DM5, **_DMG, **_DPS, **_DSS, **_DFL, **_DAA, **_DPSp}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
