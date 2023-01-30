"""
author: dajmue
date: last updated: Jan, 2023
"""

import numpy as np 
import os, random

#------------------------ Seed ------------------------ 
seed_value = 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)



def perform_umap(arrx2_float_data,
				str_saving_path: str = ".",
				int_comp: int = 16, 
				int_neigh: int = 15, 
				float_min_dist: float = 0.1,
				str_metric: str = "correlation", 
				bool_standardize: bool = True):


	import numpy as np
	from time import time as t
	import pickle as pk
	import umap
	from sklearn.preprocessing import StandardScaler


	#Reshape input data if it's not shape (X*Y, channel)
	if len(arrx2_float_data.shape)>2:
		X_2d = arrx2_float_data.reshape(-1,arrx2_float_data.shape[-1])
	else:
		X_2d = arrx2_float_data

	if bool_standardize:
		print("\n1. Standardize each spectrum separately with StandardScaler:")
		print("   Data before: std", X_2d.std(axis=1), "; mean" ,X_2d.mean(axis=1))
		#Standardize each spectrum seperately - not each channel
		X_2dT = X_2d.T
		sc = StandardScaler()
		X_scaled = sc.fit_transform(X_2dT).T
		print("   Done standardizing!")
		print("   Data after: std", X_scaled.std(axis=1), "; mean", X_scaled.mean(axis=1))
	else:
		X_scaled = X_2d

	x,y = X_2d.shape


	print("\n2. Performing UMAP with settings: \nn_components ={}, \nn_neighbors = {}, \nmin_dist = {}, \nmetric = {} ...\n".format(int_comp, int_neigh, float_min_dist, str_metric))
	model = umap.UMAP(n_neighbors=int_neigh, n_components=int_comp, min_dist=float_min_dist, metric=str_metric)
	start = t()
	X_comp = model.fit_transform(X_scaled)
	end = t()
	print("UMAP fit-transform took:", end - start,"s")



	#Save umap model & scaler
	print("\n3. Saving UMAP model and scaler if data was standardized...")
	str_save = "umap_neigh" + str(int_neigh) + "_dist" + str(float_min_dist) + "_metric" + str(str_metric) + "_comp" + str(int_comp)
	try:
		pk.dump(model, open(str_saving_path + str_save +'_model.pkl','wb'))
	except:
		import joblib
		joblib.dump(model, str_saving_path + str_save +'_joblib.sav')

	if bool_standardize:
		pk.dump(sc, open(str_saving_path + str_save + '_scaler.pkl','wb'))


	print("\n4. Saving encoded data ...")
	np.save(str_saving_path + str_save + "_train.npy", X_comp)
	
	return X_comp




def load_umap_model(arrx2_float_data,
				str_saving_path: str = ".",
				str_data_name: str = "test", 
				str_path_umap_model = "umap_neigh15_dist0.1_metriccorrelation_comp2_model.pkl",
				bool_standardize: bool = True,
				str_path_scaler_model = "umap_neigh15_dist0.1_metriccorrelation_comp2_scaler.pkl"):


	#Reshape input data if it's not shape (X*Y, channel)
	import pickle as pk
	import numpy as np
	import umap

	#Reshape input data if it's not shape (X*Y, channel)
	if len(arrx2_float_data.shape)>2:
		X_2d = arrx2_float_data.reshape(-1,arrx2_float_data.shape[-1])
	else:
		X_2d = arrx2_float_data

	#Load pca model
	print("\n1. Load UMAP model ...")
	umap_model = pk.load(open(str_path_umap_model, 'rb'))


	if bool_standardize:
		print("\n2. Load saved standard scaler and scale new data ...")
		sc = pk.load(open(str_path_scaler_model,'rb'))
		X_2dT = X_2d.T
		X_scaled = sc.fit_transform(X_2dT).T
	else:
		X_scaled = X_2d


	print("\n3. Apply dimensional reduction to new data ..")
	X_comp = umap_model.transform(X_scaled)

	print("\n4. Saving encoded data ...")
	np.save(str_saving_path + "/" + str_data_name +"_umap.npy", X_comp)

	return X_comp





if __name__ =="__main__":

	import numpy as np
	arr = np.random.randint(1,high=5, size=(30, 30,3))
	x,y,z = arr.shape

	str_saving_path = "/prodi/hpcmem/dajmue/results/Paper_SAE_CompSegNet/8_Docker_Github/"
	data = perform_umap(arr,str_saving_path,int_comp=2)
	data = load_umap_model(arr)


