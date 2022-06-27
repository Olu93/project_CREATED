import os

import tensorflow as tf

from thesis_commons.config import DEBUG_USE_MOCK
from thesis_commons.constants import PATH_MODELS_PREDICTORS
from thesis_commons.distributions import DataDistribution, DistributionConfig
from thesis_readers.helper.helper import get_all_data
from thesis_commons.modes import FeatureModes, TaskModes
from thesis_predictors.models.lstms.lstm import OutcomeLSTM
from thesis_readers import Reader
from thesis_readers.readers.AbstractProcessLogReader import AbstractProcessLogReader
from thesis_viability.datallh.datallh_measure import DatalikelihoodMeasure
from thesis_viability.outcomellh.outcomllh_measure import ImprovementMeasure as OutcomelikelihoodMeasure
from thesis_viability.similarity.similarity_measure import SimilarityMeasure
from thesis_viability.sparcity.sparcity_measure import SparcityMeasure
import thesis_viability.helper.base_distances as distances

DEBUG_SPARCITY = 0
DEBUG_SIMILARITY = 0
DEBUG_DLLH = 1
DEBUG_OLLH = 1

if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME_PREDEFINED
    ft_mode = FeatureModes.FULL

    epochs = 50
    reader: AbstractProcessLogReader = Reader.load()
    vocab_len = reader.vocab_len
    max_len = reader.max_len
    # TODO: Implement cleaner version. Could use from_config instead of init_metrics as both are static methods
    custom_objects_predictor = {obj.name: obj for obj in OutcomeLSTM.init_metrics()}

    # generative_reader = GenerativeDataset(reader)
    tr_cases, cf_cases, fa_cases = get_all_data(reader, ft_mode=ft_mode, fa_num=10, fa_filter_lbl=None, cf_num=15)
    print("\n")
    if DEBUG_SPARCITY:
        print("Run Sparcity")
        sparcity_computer: SparcityMeasure = SparcityMeasure().set_vocab_len(vocab_len).set_max_len(max_len).init()
        sparcity_values = sparcity_computer.compute_valuation(fa_cases, cf_cases).normalize()
        print(sparcity_values)

    if DEBUG_SIMILARITY:
        print("Run Similarity")
        similarity_computer: SimilarityMeasure = SimilarityMeasure().set_vocab_len(vocab_len).set_max_len(max_len).init()
        similarity_values = similarity_computer.compute_valuation(fa_cases, cf_cases).normalize()
        print(similarity_values)

    if DEBUG_DLLH:
        print("Run Data Likelihood")
        data_distribution = DataDistribution(tr_cases, vocab_len, max_len, reader._idx_distribution, DistributionConfig.registry()[0])
        dllh_computer: DatalikelihoodMeasure = DatalikelihoodMeasure().set_data_distribution(data_distribution).init()
        dllh_values = dllh_computer.compute_valuation(fa_cases, cf_cases).normalize()
        sampled_cases = dllh_computer.sample(5)
        print(dllh_values)
        print(sampled_cases)

    if DEBUG_OLLH:
        print("Run Outcome Likelihood")
        all_models = os.listdir(PATH_MODELS_PREDICTORS)
        pmodel = tf.keras.models.load_model(PATH_MODELS_PREDICTORS / all_models[-1], custom_objects=custom_objects_predictor)
        dist = distances.LikelihoodDifference()
        improvement_computer: OutcomelikelihoodMeasure = OutcomelikelihoodMeasure().set_evaluator(dist).set_predictor(pmodel).set_vocab_len(vocab_len).set_max_len(max_len).init()
        improvement_values = improvement_computer.compute_valuation(fa_cases, cf_cases).normalize()
        print(improvement_values)
        
    print("DONE")
