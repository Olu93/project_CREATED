from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader

import importlib
import io
import json
import pathlib
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from thesis_commons.modes import DatasetModes, FeatureModes
from thesis_commons.representations import Cases

TFLossSpec = Union[str, str, Dict[str, Any]]


def create_path(pthname: str, pth: pathlib.Path):
    print(f"{pthname}: {pth.absolute()}")
    if not pth.exists():
        print(f"ATTENTION: Create a new path {pth.absolute()}")
    pth.mkdir(parents=True, exist_ok=True)
    return pth


def shift_seq_forward(seq):
    seq = np.roll(seq, 1, -1)
    seq[:, 0] = 0
    return seq


def shift_seq_backward(seq):
    seq = np.roll(seq, -1, -1)
    seq[:, -1] = 0
    return seq


def reverse_sequence(data_container):
    original_data = np.array(data_container)
    flipped_data = np.flip(data_container, axis=-2)
    results = np.zeros_like(original_data)
    results[np.nonzero(original_data.sum(-1) != 0)] = flipped_data[(flipped_data.sum(-1) != 0) == True]
    return results


def reverse_sequence_2(data_container):
    original_data = np.array(data_container)
    results = np.zeros_like(original_data)
    if len(data_container.shape) == 2:
        flipped_data = np.flip(data_container, axis=-1)
        results[np.nonzero(flipped_data != 0)] = original_data[(original_data != 0)]
    if len(data_container.shape) == 3:
        flipped_data = np.flip(data_container, axis=-2)
        results[np.nonzero(flipped_data.sum(-1) != 0)] = original_data[(original_data.sum(-1) != 0)]

    return results


def split_params(input):
    mus, logsigmas = input[:, :, 0], input[:, :, 1]
    return mus, logsigmas


# @tf.function
def sample(inputs):
    mean, logvar = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean))
    # TODO: Maybe remove the 0.5 and include proper log handling -- EVERYWHERE
    return mean + tf.exp(0.5 * logvar) * epsilon


def stack_data(a):
    a_evs, a_fts = zip(*a)
    a_evs_stacked, a_fts_stacked = tf.concat(list(a_evs), axis=0), tf.concat(list(a_fts), axis=0)
    return a_evs_stacked.numpy(), a_fts_stacked.numpy()


# def reverse_sequence(data_container):
#     original_data = tf.TensorArray(data_container)
#     flipped_data = np.flip(data_container, axis=-2)
#     results = np.zeros_like(original_data)
#     results[np.nonzero(original_data.sum(-1) != 0)] = flipped_data[(flipped_data.sum(-1) != 0) == True]
#     return results


def extract_loss(fn: tf.keras.losses.Loss) -> TFLossSpec:
    result = {}
    result['module_name'] = fn.__module__
    result['class_name'] = fn.__class__.__name__
    result['config'] = fn.get_config()
    return result


def instantiate_loss(cls_details: TFLossSpec) -> object:
    module = importlib.import_module(cls_details.get('module_name'))
    class_description = getattr(module, cls_details.get('class_name'))
    cfg = cls_details.get('config')
    # fns = cfg.pop('fns')
    instance = class_description().from_config(cfg)
    return instance


def save_loss(path: pathlib.Path, fn: tf.keras.losses.Loss):
    cls_details = extract_loss(fn)
    try:
        new_path = path / "loss.json"
        json.dump(cls_details, io.open(new_path, 'w'))
        return new_path.absolute()
    except Exception as e:
        print(e)
    return None


def save_metrics(path: pathlib.Path, fns: Sequence[tf.keras.losses.Loss]):
    cls_details = [extract_loss(fn) for fn in fns]
    paths = []
    try:
        for i in enumerate(cls_details):
            new_path = path / f"metrics_{i}.json"
            json.dump(cls_details, io.open(new_path, 'w'))
            paths.append(path.absolute(new_path))
        return paths
    except Exception as e:
        print(e)
    return None


def load_loss(path: pathlib.Path):
    try:
        cls_details = json.load(io.open(path, 'r'))
        instance = instantiate_loss(cls_details)
        return instance
    except Exception as e:
        print(e)
    return None


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

    return tr_cases, cf_cases, fa_cases


def apply_filters_on_data(events: NDArray, features: NDArray, labels: NDArray, num: int, filter_lbl: int) -> Cases:
    if filter_lbl is not None:
        selected = (labels == filter_lbl).flatten()
        events, features, labels = events[selected], features[selected], labels[selected]
    if num is not None:
        events, features, labels = events[:num], features[:num], labels[:num]
    fa_cases = Cases(events, features, labels)
    return fa_cases
