from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from thesis_commons.model_commons import GeneratorWrapper

from numbers import Number
from typing import Any, Callable, Dict, List, Mapping, Sequence, TypedDict

import pandas as pd
import numpy as np
from thesis_commons.functions import decode_sequences, decode_sequences_str, remove_padding

from thesis_commons.representations import BetterDict, Cases, EvaluatedCases
from thesis_viability.viability.viability_function import MeasureMask


# TODO: Move evolutionary statistics here
# TODO: Represent other statistics here: ViabilityMeasure, EvoluationaryStrategy, counterfactual Wrappers
class UpdateSet(TypedDict):
    model: GeneratorWrapper
    results: Sequence[EvaluatedCases]
    measure_mask: MeasureMask


class StatsMixin(ABC):
    def __init__(self, level="NA", **kwargs):
        self.pad_id = kwargs.pop("pad_id", 0)
        self.level: str = level
        self.name: str = self.level
        self._store: Dict[int, StatsMixin] = kwargs.pop('_store', {})
        self._additional: BetterDict = kwargs.pop('_additional', BetterDict())
        self._stats: List[StatsMixin] = kwargs.pop('_stats', [])
        self._identity: Union[str, int] = kwargs.pop('_identity', {self.level: 1})
        self.is_digested: bool = False

    def append(self, datapoint: StatsMixin) -> StatsMixin:
        self._store[len(self._store) + 1] = datapoint
        return self

    def attach(self, key: str, val: Union[Number, Dict, str], transform_fn: Callable = None, with_level: bool = False) -> StatsMixin:
        val = val if transform_fn is None else transform_fn(val)
        if key is not None:
            self._additional[f"{self.level}.{key}"] = val
            return self

        if not isinstance(val, dict):
            raise Exception(f"Val has to be a dictionary if key is not supplied \nKey is {key} \nVal is{val}")
        d = {self.level: val} if with_level else val
        self._additional.merge(d)
        return self

    def set_identity(self, identity: Union[str, int] = 1) -> StatsMixin:
        self._identity = {self.level: {'no': identity}}
        return self

    def _digest(self) -> StatsMixin:
        self._stats = [item.set_identity(idx)._digest() for idx, item in self._store.items()]
        self._is_digested = True
        return self

    def gather(self) -> List[Dict[str, Union[str, Number, Dict]]]:
        result_list = []
        self = self._digest()
        for value in self._stats:
            result_list.extend([BetterDict().merge(self._additional).merge(self._identity).merge(d) for d in value.gather()])
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


class StatRow(StatsMixin):
    def __init__(self, data: Dict = None, **kwargs) -> None:
        super().__init__(level="row")
        self._additional = data or BetterDict()
        self._digested_data = None
        self._combined_data = None

    def __repr__(self):
        return f"@{self.level}[{repr(self._additional)}]"

    def _digest(self) -> StatRow:
        self._stats = [self._additional]
        self.is_digested = True
        return self

    def gather(self) -> List[Dict[str, Union[str, Number]]]:
        return [BetterDict().merge(self._identity).merge(item) for item in self._stats]


class StatIteration(StatsMixin):
    _store: Dict[int, StatRow]

    def __init__(self):
        super().__init__(level="iteration")


class StatCases(StatIteration):
    _store: Dict[int, StatRow]

    def attach(self, val: EvaluatedCases, **kwargs):
        if not isinstance(val, EvaluatedCases):
            return super().attach(val, **kwargs)
        all_results = []
        cf_events = []
        fa_events = []
        for _, case_dict in enumerate(val.to_dict_stream()):
            case_result = self._transform(case_dict)
            cf_events.append(case_result.pop('cf_events'))
            fa_events.append(case_result.pop('fa_events'))
            all_results.append(case_result)

        cf_events_no_padding = decode_sequences(cf_events)
        fa_events_no_padding = decode_sequences(fa_events)
        # cf_events_decoded = decode_sequences(cf_events_no_padding, self.idx2vocab)
        # fa_events_decoded = decode_sequences(fa_events_no_padding, self.idx2vocab)

        for item, cf, fa in zip(all_results, cf_events_no_padding, fa_events_no_padding):
            row_data = StatRow(data=BetterDict().merge(item).merge({"cf": cf, "fa": fa}))
            self.append(row_data)
        return self

    def _transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_name": result.get("creator"),
            # "instance_num": result.get("instance_num"),
            "rank": result.get("rank"),
            "likelihood": result.get("likelihood"),
            "viability": result.get("viability"),
            "sparcity": result.get("sparcity"),
            "similarity": result.get("similarity"),
            "feasibility": result.get("dllh"),
            "delta": result.get("ollh"),
            "cf_events": result.get("cf_events"),
            "fa_events": result.get("fa_events"),
            "cf_num_zeros": result.get("cf_num_zeros"),
            "result_outcome": result.get("outcome"),
            "source_outcome": result.get("fa_outcome"),
            "target_outcome": 1 - result.get("fa_outcome"),
        }


class StatInstance(StatsMixin):
    _store: Dict[int, StatIteration]

    def __init__(self) -> None:
        super().__init__(level="instance")


class StatRun(StatsMixin):
    _store: Dict[int, StatInstance]

    def __init__(self) -> None:
        super().__init__(level="run")


class ExperimentStatistics(StatsMixin):
    _store: Dict[int, StatRun]

    def __init__(self, **kwargs) -> None:
        super().__init__(level="experiment")

    @property
    def data(self) -> pd.DataFrame:
        data = pd.json_normalize(self.gather())
        return data


def attach_descriptive_stats(curr_stats: StatIteration, counterfactuals: EvaluatedCases, factuals:EvaluatedCases = None):
    cf_sum_zeroes = (counterfactuals.events == 0).sum(-1)
    cf_sum_events = (counterfactuals.events != 0).sum(-1)
    cf_max_len = counterfactuals.events.shape[-1]
    curr_stats.attach("mean_num_zeros", (np.divide(cf_sum_zeroes, cf_max_len).mean(-1)))
    curr_stats.attach("mean_num_events", (np.divide(cf_sum_events, cf_max_len).mean(-1)))
    curr_stats.attach("mean_sparcity", counterfactuals.viabilities.sparcity.mean())
    curr_stats.attach("mean_similarity", counterfactuals.viabilities.similarity.mean())
    curr_stats.attach("mean_feasibility", counterfactuals.viabilities.dllh.mean())
    curr_stats.attach("mean_delta", counterfactuals.viabilities.ollh.mean())
    curr_stats.attach("mean_viability", counterfactuals.avg_viability[0])
    curr_stats.attach("min_sparcity", counterfactuals.viabilities.sparcity.min())
    curr_stats.attach("min_similarity", counterfactuals.viabilities.similarity.min())
    curr_stats.attach("min_feasibility", counterfactuals.viabilities.dllh.min())
    curr_stats.attach("min_delta", counterfactuals.viabilities.ollh.min())
    curr_stats.attach("min_viability", counterfactuals.min_viability[0])
    curr_stats.attach("max_sparcity", counterfactuals.viabilities.sparcity.max())
    curr_stats.attach("max_similarity", counterfactuals.viabilities.similarity.max())
    curr_stats.attach("max_feasibility", counterfactuals.viabilities.dllh.max())
    curr_stats.attach("max_delta", counterfactuals.viabilities.ollh.max())
    curr_stats.attach("max_viability", counterfactuals.max_viability[0])
    curr_stats.attach("median_sparcity", np.median(counterfactuals.viabilities.sparcity))
    curr_stats.attach("median_similarity", np.median(counterfactuals.viabilities.similarity))
    curr_stats.attach("median_feasibility", np.median(counterfactuals.viabilities.dllh))
    curr_stats.attach("median_delta", np.median(counterfactuals.viabilities.ollh))
    curr_stats.attach("median_viability", counterfactuals.median_viability[0])
    if factuals:
        divisor_zeros = (factuals.events == 0).sum(-1)
        divisor_events = (factuals.events != 0).sum(-1)
        curr_stats.attach("mean_pred_outcome", counterfactuals.outcomes.mean())
        curr_stats.attach("target_outcome", (1-factuals.outcomes)[:,0].mean())
        curr_stats.attach("rel_mean_num_zeros",((np.divide(cf_sum_zeroes, divisor_zeros)).mean(-1)))
        curr_stats.attach("rel_mean_num_events",((np.divide(cf_sum_events, divisor_events)).mean(-1)))
    return curr_stats