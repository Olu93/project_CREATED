
import importlib_resources

MODEL_FOLDER = importlib_resources.files(__package__).parent.parent.parent / "models"
EVAL_RESULTS_FOLDER = importlib_resources.files(__package__).parent.parent.parent / "results"

print("================= Folder =====================")
print(f"Models: {MODEL_FOLDER}")
print(f"Evaluation Results: {EVAL_RESULTS_FOLDER}")
print("==============================================")

SEQUENCE_LENGTH = "seq_len"
NUMBER_OF_INSTANCES = 'num_instances'