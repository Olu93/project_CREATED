from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader

from thesis_readers import OutcomeMockReader
from thesis_readers import OutcomeBPIC12ReaderShort


IS_PROD = False
DEBUG_USE_QUICK_MODE = True
DEBUG_USE_MOCK = False
DEBUG_SKIP_DYNAMICS = True if not IS_PROD else False
DEBUG_SKIP_VIZ = True if not IS_PROD else False


    
Reader:AbstractProcessLogReader = OutcomeMockReader if DEBUG_USE_MOCK else OutcomeBPIC12ReaderShort