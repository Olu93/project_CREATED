from thesis_commons.random import random
from typing import Tuple
import numpy as np
from thesis_commons.modes import DatasetModes, FeatureModes
from thesis_commons.representations import Cases, MutationRate
# from numpy.typing import np.ndarray
from thesis_readers.readers.AbstractProcessLogReader import \
    AbstractProcessLogReader


def test_reader(
        reader: AbstractProcessLogReader,
        recompute_log: bool = True,
        with_viz_bpmn: bool = True,
        with_viz_procmap: bool = True,
        with_viz_dfg: bool = True,
        save_preprocessed: bool = True,
        save_viz = False,
):
    if recompute_log:
        reader = reader.init_log(save_preprocessed)
    reader = reader.init_meta()
    ds_counter = reader.get_dataset()
    example = next(iter(ds_counter.batch(10)))
    print(example[0][0].shape)
    print(example[0][1].shape)
    if with_viz_bpmn:
        print("Inititiating BPMN visualization")
        reader.viz_bpmn("white", save_viz)
    if with_viz_procmap:
        print("Inititiating Process Map visualization")
        reader.viz_process_map("white", save_viz)
    if with_viz_dfg:
        print("Inititiating DFG visualization")
        reader.viz_dfg("white", save_viz)
    
    return reader.get_data_statistics()

def get_all_data(
    reader: AbstractProcessLogReader,
    ft_mode: FeatureModes = FeatureModes.FULL,
    tr_num: int = None,
    tr_filter_lbl: int = None,
    cf_num: int = None,
    cf_filter_lbl: int = None,
    fa_num: int = None,
    fa_filter_lbl: int = None,
) -> Tuple[Cases, Cases, Cases]:
    (tr_events, tr_features), tr_labels = reader._generate_dataset(data_mode=DatasetModes.TRAIN, ft_mode=ft_mode)
    (cf_events, cf_features), cf_labels = reader._generate_dataset(data_mode=DatasetModes.VAL, ft_mode=ft_mode)
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=ft_mode)

    tr_cases = apply_filters_on_data(tr_events, tr_features, tr_labels, tr_num, tr_filter_lbl)
    cf_cases = apply_filters_on_data(cf_events, cf_features, cf_labels, cf_num, cf_filter_lbl)
    fa_cases = apply_filters_on_data(fa_events, fa_features, fa_labels, fa_num, fa_filter_lbl)
    
    if not all((len(tr_cases), len(cf_cases), len(fa_cases))):
        raise Exception(f"One of the dataset is empty. The sizes are tr_cases:{len(tr_cases)}, cf_cases:{len(cf_cases)}, fa_cases:{len(fa_cases)}")

    return tr_cases, cf_cases, fa_cases


def get_even_data(
    reader: AbstractProcessLogReader,
    ft_mode: FeatureModes = FeatureModes.FULL,
    ds_mode: DatasetModes = DatasetModes.TEST,
    fa_num: int = None,
) -> Tuple[Cases, Cases, Cases]:
    (fa_events, fa_features), fa_labels = reader._generate_dataset(data_mode=DatasetModes.TEST, ft_mode=ft_mode)

    fa_cases_1 = apply_filters_on_data(fa_events, fa_features, fa_labels, fa_num, 1)
    fa_cases_2 = apply_filters_on_data(fa_events, fa_features, fa_labels, fa_num, 0)
    
    if not all((len(fa_cases_1),len(fa_cases_2))):
        raise Exception(f"One of the dataset is empty. The sizes are fa_cases_1:{len(fa_cases_1)}, fa_cases_2:{len(fa_cases_2)}")

    return fa_cases_1 + fa_cases_2

def apply_filters_on_data(events: np.ndarray, features: np.ndarray, labels: np.ndarray, num: int, filter_lbl: int) -> Cases:
    if filter_lbl is not None:
        selected = (labels == filter_lbl).flatten()
        events, features, labels = events[selected], features[selected], labels[selected]
    if num is not None:
        events, features, labels = events[:num], features[:num], labels[:num]
    fa_cases = Cases(events, features, labels)
    return fa_cases

def create_random_mrate():
    remainder = 1
    scores = random.uniform(0, 1, 5)
    return MutationRate(*list(scores))