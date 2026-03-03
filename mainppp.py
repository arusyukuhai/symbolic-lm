from datasets import load_dataset
import tqdm
import re
import string
import random
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


model = sklearn.linear_model.LinearRegression()

print(string.printable)

# Load a specific split
train = load_dataset("ronantakizawa/github-top-code", split="train")
test = load_dataset("ronantakizawa/github-top-code", split="test")

random.seed(0)

xs = []
ys = []
#used_chars = set()
for code in tqdm.tqdm(range(100000)):
    stri = list(train[code]["content"])
    s = random.randint(1, len(stri))
    s2 = (len(stri) - s) / len(stri)
    for j in random.choices(list(range(len(stri))), k=s):
        stri[j] = string.printable[random.randint(0, len(string.printable)-1)]
    x = np.array([stri.count(c) / len(stri) for c in string.printable])
    y = s2
    xs.append(x)
    ys.append(y)

model.fit(xs, ys)
g = model.predict(xs)
print(np.corrcoef(ys, g)[0, 1])
print(spearmanr(ys, g).statistic)

plt.scatter(ys, g, s=0.25)
plt.show()

print(model.coef_)

np.savez("coef.npz", coef=model.coef_, intercept=model.intercept_)

funcs_ = [
    lambda x, z: {
        
    }
]