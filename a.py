
from datasets import load_dataset

# Load the high-quality subset
data = load_dataset("HuggingFaceTB/finemath", "finemath-4plus")
print(data[0])