# %%

import jsonlines
import pandas as pd

from thesis_readers import *
from thesis_readers.helper.helper import test_reader

stats_collector = []

# reader = AbstractProcessLogReader(
#     log_path=constants.DATA_FOLDER / 'dataset_bpic2020_tu_travel/RequestForPayment.xes',
#     csv_path=constants.DATA_FOLDER_PREPROCESSED / 'RequestForPayment.csv',
#     mode=TaskModes.SIMPLE,
# )
# reader = reader.init_log(save=0)
# reader = reader.init_data()
# point = next(reader._generate_examples(DatasetModes.TRAIN))
# mode_combos = list(it.product(ShapeModes, ShapeModes))
# for combo in mode_combos:
#     test_dataset(reader, DatasetModes.TRAIN, *combo)

# print(reader.get_data_statistics())
# %% -------------------------------------------------------------------------------
reader = BPIC12LogReader()
stats_collector.append(test_reader(reader, True, save_viz=True))
reader.data.head()
# %% -------------------------------------------------------------------------------
reader = DomesticDeclarationsLogReader()
stats_collector.append(test_reader(reader, True, save_viz=True))
reader.data.head()
# %% -------------------------------------------------------------------------------
reader = PermitLogReader()
stats_collector.append(test_reader(reader, True, save_viz=True))
reader.data.head()
# %% -------------------------------------------------------------------------------
reader = RabobankTicketsLogReader()
stats_collector.append(test_reader(reader, True, save_viz=True))
reader.data.head()
# %% -------------------------------------------------------------------------------
reader = RequestForPaymentLogReader()
stats_collector.append(test_reader(reader, True, save_viz=True))
reader.data.head()
# %% -------------------------------------------------------------------------------
reader = VolvoIncidentsReader()
stats_collector.append(test_reader(reader, True, save_viz=True))
reader.data.head()
# %% -------------------------------------------------------------------------------
reader = HospitalLogReader()
stats_collector.append(test_reader(reader, True, with_viz_procmap=False, save_viz=True))
reader.data.head()
# # %% -------------------------------------------------------------------------------
# reader = SepsisLogReader()
# stats_collector.append(test_reader(reader, True))
# reader.data.head()

# %% ------------------------------
jsonlines.open('statst.jsonl', mode='w').write_all(stats_collector)
pd.json_normalize([dict(jl, column_stats=None) for jl in stats_collector]).to_csv('statst.csv')