"""
author: dajmue
date: last updated: Jan, 2023
"""

print("Importing dimensionality reduction modules ...")
from SCAE_tf2 import *
from FCCAE_tf2 import *
from PCA_tf2 import *
from UMAP_tf2 import *
print("Done. ")


print("\nThe following functions are now available:\n \
	  \nStacked contractive autoencoder: train_scae() \
	  \nFully connected contractive autoencoder: train_fccae() \
	  \nPCA: perform_pca(), load_pca_model()\
	  \nUMAP: perform_umap(),load_umap_model()"
	  )
print("\nMore informations regarding the functions can be found on our GitHub: \
	   \nhttps://github.com/RUB-Bioinf/DimensionalityReduction"
	   )