from typing import Any, Dict, Mapping, Sequence, TypedDict
import pandas as pd
from numpy.typing import NDArray
from thesis_commons.model_commons import GeneratorMixin
from thesis_commons.representations import Cases, EvaluatedCases

# TODO: Move evolutionary statistics here
# TODO: Collect other wrapper statistics here
class UpdateSet(TypedDict):
    model: GeneratorMixin
    results: Sequence[EvaluatedCases]


class ResultStatistics():
    def __init__(self) -> None:
        self._data: Mapping[str, UpdateSet] = {}
        self._digested_data = None

    # num_generation, num_population, num_survivors, fitness_values
    def update(self, model: GeneratorMixin, data: Cases):
        self._data[model.name] = {"model": model, "results": model.generate(data)}
        return self

    def _digest(self):
        all_generator_results = [res for k, v in self._data.items() for res in v["results"]]
        all_digested_results = [self._transform(dict_result) for v in all_generator_results for dict_result in v.to_dict_stream()]
        self._digested_data = pd.DataFrame(all_digested_results)
        return self

    @property
    def data(self) -> pd.DataFrame:
        self._digest()
        return self._digested_data

    def _transform(self, result: Dict[str, Any]) -> Dict[str, NDArray]:

        return {
            "model_name": result.get("creator"),
            "instance_num": result.get("instance_num"),
            "likelihood": result.get("likelihood"),
            "outcome": result.get("outcome"),
            "viability": result.get("viability"),
        }

    def __repr__(self):
        return repr(self.data.groupby(["model_name", "instance_num"]).agg({'viability': ['mean', 'min', 'max', 'median'], 'likelihood': ['mean', 'min', 'max', 'median']}))