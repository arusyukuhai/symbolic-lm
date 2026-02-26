from datasets import load_dataset

# Load a specific split
train = load_dataset("ronantakizawa/github-top-code", split="train")
test = load_dataset("ronantakizawa/github-top-code", split="test")

print(train[0]["content"])
print(test[0]["content"])
print(len(train))
print(len(test))