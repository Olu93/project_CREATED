from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from thesis_commons.model_commons import GeneratorMixin

from numbers import Number
from typing import Any, Callable, Dict, List, Mapping, Sequence, TypedDict

import pandas as pd
from numpy.typing import NDArray
from thesis_commons.functions import decode_sequences, decode_sequences_str, remove_padding

from thesis_commons.representations import Cases, EvaluatedCases
from thesis_viability.viability.viability_function import MeasureMask


# TODO: Move evolutionary statistics here
# TODO: Represent other statistics here: ViabilityMeasure, EvoluationaryStrategy, counterfactual Wrappers
class UpdateSet(TypedDict):
    model: GeneratorMixin
    results: Sequence[EvaluatedCases]
    measure_mask: MeasureMask


class StatsMixin(ABC):
    def __init__(self, level="NA", **kwargs):
        self.level: str = level
        self.name: str = self.level
        self._store: Dict[int, StatsMixin] = kwargs.pop('_store', {})
        self._additional: Dict[int, StatsMixin] = kwargs.pop('_additional', {})
        self._stats: List[StatsMixin] = kwargs.pop('_stats', [])
        self._identity: Union[str, int] = kwargs.pop('_identity', {self.level: 1})
        self.is_digested: bool = False

    def append(self, datapoint: StatsMixin) -> StatsMixin:
        self._store[len(self._store) + 1] = datapoint
        return self

    def attach(self, key: str, val: Union[Number, Dict, str]) -> StatsMixin:
        accessor = f"{self.level}.{key}"
        if (type(val) == dict) and accessor in self._additional:
            self._additional[accessor] = {**self._additional[accessor], **val}
            return self
        self._additional[accessor] = val
        return self

    def set_identity(self, identity: Union[str, int] = 1) -> StatsMixin:
        self._identity = {self.level: identity}
        return self

    def _digest(self) -> StatsMixin:
        self._stats = [item.set_identity(idx)._digest() for idx, item in self._store.items()]
        self._is_digested = True
        return self

    def gather(self) -> List[Dict[str, Union[str, Number, Dict]]]:
        result_list = []
        self = self._digest()
        for value in self._stats:
            result_list.extend([{**self._identity, **self._additional, **d} for d in value.gather()])
        return result_list

    @property
    def data(self) -> pd.DataFrame:
        # https://stackoverflow.com/a/66684215
        return pd.json_normalize(self.gather())

    @property
    def num_digested(self):
        return sum(v.is_digested for v in self._store.values())

    def __repr__(self):
        return f"@{self.name}[Size:{len(self)} - Digested: {self.num_digested}]"

    @classmethod
    def from_stats(cls, **kwargs) -> StatsMixin:
        return cls(**kwargs)

    def __getitem__(self, key):
        return self.gather()[key]

    def __len__(self):
        return len(self._store)


class RowData(StatsMixin):
    def __init__(self, **kwargs) -> None:
        super().__init__(name="row")
        self._store = kwargs.pop('_store', {})
        self._digested_data = None
        self._combined_data = None

    # num_generation, num_population, num_survivors, fitness_values
    def attach(self, stat_name: str, val: Union[Number, Dict], transform_fn: Callable = None) -> RowData:
        self._store = {**self._store, **{stat_name: val if not transform_fn else transform_fn(val)}}
        return self

    def __repr__(self):
        dict_copy = dict(self._store)
        return f"@{self.level}[{repr(dict_copy)}]"

    def _digest(self) -> RowData:
        self._stats = [{**self._store}]
        self.is_digested = True
        return self

    def gather(self) -> List[Dict[str, Union[str, Number]]]:
        return [{**self._identity, **item} for item in self._stats]


class IterationData(StatsMixin):
    _store: Dict[int, RowData]

    def __init__(self):
        super().__init__(level="iteration")


class InstanceData(StatsMixin):
    _store: Dict[int, IterationData]

    def __init__(self) -> None:
        super().__init__(level="instance")


class RunData(StatsMixin):
    _store: Dict[int, InstanceData]

    def __init__(self) -> None:
        super().__init__(level="model")


class ResultStatistics(RunData):
    _store: Dict[int, InstanceData]

    def __init__(self, idx2vocab: Dict[int, str], pad_id: int = 0) -> None:
        super(RunData, self).__init__(level="global")
        self._data: Mapping[str, UpdateSet] = {}
        self.DEPRECATED_digested_data = None
        self.idx2vocab = idx2vocab
        self.pad_id = pad_id


    # num_generation, num_population, num_survivors, fitness_values
    def update(self, results:Sequence[EvaluatedCases]):
        # store = {}
        for instance_idx, instance_result in enumerate(results):
            all_results = []
            cf_events = []
            fa_events = []
            instance_data = InstanceData()
            iteration_data = IterationData()
            
            for case_idx, case_dict in enumerate(instance_result.to_dict_stream()):
                case_result = self._transform(case_dict)
                cf_events.append(case_result.pop('cf_events'))
                fa_events.append(case_result.pop('fa_events'))
                all_results.append(case_result)
                
            cf_events_no_padding = remove_padding(cf_events, self.pad_id)
            fa_events_no_padding = remove_padding(fa_events, self.pad_id)
            cf_events_decoded = decode_sequences(cf_events_no_padding, self.idx2vocab)
            fa_events_decoded = decode_sequences(fa_events_no_padding, self.idx2vocab)
                            
            for item, cf, fa in zip(all_results, cf_events_decoded, fa_events_decoded):
                row_data = RowData(_store=item).attach("case", {"cf":cf, "fa":fa})
                iteration_data.append(row_data)
            # store[len(self._store)] = iteration_data 
            self.append(instance_data.append(iteration_data))
        return self


    def _transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_name": result.get("creator"),
            # "instance_num": result.get("instance_num"),
            "rank": result.get("rank"),
            "likelihood": result.get("likelihood"),
            "outcome": result.get("outcome"),
            "viability": result.get("viability"),
            "sparcity": result.get("sparcity"),
            "similarity": result.get("similarity"),
            "dllh": result.get("dllh"),
            "ollh": result.get("ollh"),
            "cf_events": result.get("cf_events"),
            "fa_events": result.get("fa_events"),
            "source_outcome": result.get("fa_outcomes"),
            "target_outcome": 1 - result.get("fa_outcomes"),
        }

    def _add_global_vals(self, result: Dict[str, Any], mask_settings: Dict[str, bool]) -> Dict[str, NDArray]:

        return {**result, **mask_settings}


class ExperimentStatistics(StatsMixin):
    _store: Dict[int, RunData]

    def __init__(self) -> None:
        super().__init__(level="experiment")

