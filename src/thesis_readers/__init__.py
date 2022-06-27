from .readers.AbstractProcessLogReader import *
from .readers.AbstractProcessLogReader import (AbstractProcessLogReader,
                                               DatasetModes, TaskModes)
from .readers.BPIC12LogReader import BPIC12LogReader
from .readers.DomesticDeclarationsLogReader import \
    DomesticDeclarationsLogReader
from .readers.HospitalLogReader import HospitalLogReader
from .readers.MockReader import MockReader
from .readers.OutcomeReader import *
from .readers.PermitLogReader import PermitLogReader
from .readers.RaboTicketsLogReader import RabobankTicketsLogReader
from .readers.RequestForPaymentLogReader import RequestForPaymentLogReader
from .readers.SepsisLogReader import SepsisLogReader
from .readers.VolvoIncidentsReader import VolvoIncidentsReader

from thesis_commons.config import DEBUG_USE_MOCK
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_readers import OutcomeMockReader
from thesis_readers import OutcomeBPIC12ReaderShort
Reader:AbstractProcessLogReader = OutcomeMockReader if DEBUG_USE_MOCK else OutcomeBPIC12ReaderShort