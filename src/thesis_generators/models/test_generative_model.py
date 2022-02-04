from thesis_readers import DomesticDeclarationsLogReader as Reader
from thesis_readers.helper.modes import TaskModes, FeatureModes

if __name__ == "__main__":
    task_mode = TaskModes.OUTCOME
    reader = Reader(mode=task_mode).init_data()
    idx = 1
    training_data = reader.get_dataset(ft_mode=FeatureModes.EVENT_ONLY).batch(15)
    
