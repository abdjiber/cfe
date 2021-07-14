"""
Example of application of the CFE algorithm.
"""
import numpy as np
from cfe import CFE

soybean = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data", delimiter=",", dtype="O")
n_features = soybean.shape[1]
features = [f"A{i}" for i in range(1, n_features + 1)]
true_labels = soybean[:, -1] # Last column corresponds to objects classes.
soybean = np.delete(soybean, n_features - 1, axis=1)
cfe = CFE(n_clusters=4, m=1.1, verbose=False)
cfe.fit(soybean, features)
print("Scores")
print("Partition coefficient: ", cfe.pe)
print("Partition entropy: ", cfe.pc)
print(cfe.predict(soybean[:10]))
