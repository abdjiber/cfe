"""
Example of application of the CFE algorithm.
"""
import pandas as pd
from cfe import CFE
soybean_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data")
soybean_df.columns = [f"A{i}" for i in range(1, soybean_df.shape[1] + 1)]
true_labels = soybean_df.A36.values # Last column corresponds to objects classes.
soybean_df = soybean_df.drop("A36", axis=1)
X = soybean_df.values
features = list(soybean_df)
cfe = CFE(n_clusters=4, m=1.1, verbose=False)
cfe.fit(X, features)
ari = cfe.ari(true_labels)
print("Scores")
print("Partition coefficient: ", cfe.pe)
print("Partition entropy: ", cfe.pc)
print("ARI: ", ari)
