# About
This repository contains the Python implementation of my research paper algorithm **categorical fuzzy entropy c-means** (CFE).

# Examples
`import pandas as pd`
`from cfe import CFE`
`soybean_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data")`
`soybean_df.columns = [f"A{i}" for i in range(1, soybean_df.shape[1] + 1)]`
`true_labels = soybean_df.A36.values # Last column corresponds to the objects classes.`
`soybean_df = soybean_df.drop("A36", axis=1)`
`X = soybean_df.values`
`features = list(soybean_df)`
`cfe = CFE(n_clusters=4, m=1.1, verbose=False)`
`cfe.fit(X, features)`
`ari = cfe.ari(true_labels)`
`print("Scores")`
`print("Partition coefficient: ", cfe.pe)`
`print("Partition entropy: ", cfe.pc)`
`print("ARI: ", ari)`
`

# Citations
If you use this work, please cite the following papers.
> A. J. Djiberou Mahamadou, V. Antoine, E. M. Nguifo and S. Moreno, "Categorical fuzzy entropy c-means" 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE),  Glasgow, UK.

> Abdoul Jalil Djiberou Mahamadou, Violaine Antoine, Engelbert Mephu Nguifo, Sylvain Moreno, “Apport de l'entropie pour les c-moyennes floues sur des données catégorielles”, EGC 2021, vol. RNTI-E-37.
