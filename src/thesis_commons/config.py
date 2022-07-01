from __future__ import annotations

from typing import TYPE_CHECKING


IS_PROD = False
DEBUG_USE_QUICK_MODE = True
DEBUG_QUICK_EVO_MODE = True

DEBUG_QUICK_TRAIN = False
DEBUG_USE_MOCK = False
DEBUG_SKIP_DYNAMICS = True if not IS_PROD else False
DEBUG_SKIP_VIZ = True if not IS_PROD else False
FIX_BINARY_OFFSET = 1
DEBUG_DISTRIBUTION = True
    
