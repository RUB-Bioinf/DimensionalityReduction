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



def perform_pca(arrx2_float_data,
				int_comp: int = 16,
				str_saving_path: str ="",
				bool_standardize: bool = True):

	"""
	Perform principle component analysis on spectral
	data of shape (X*Y,spectra). Data can be standarized
	such that they have a mean of 0 and standard deviation 
	of 1. 
	"""
	import os, sys
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler
	from time import time as t
	import pickle as pk


	#Check files and path arguments, create saving dictionaries
	if not os.path.isdir(str_saving_path):
		sys.exit("The path", str_saving_path, "does not exist!")

	#Reshape input data if it's not shape (X*Y, channel)
	if len(arrx2_float_data.shape)>2:
		arrx2_float_data = arrx2_float_data.reshape(-1,arrx2_float_data.shape[-1])

	if bool_standardize:
		print("\n1. Standardize each spectrum separately with StandardScaler:")
		print("   Data before: std", arrx2_float_data.std(axis=1), "; mean" ,arrx2_float_data.mean(axis=1))
		#Standardize each spectrum seperately - not each channel
		X_2dT = arrx2_float_data.T
		sc = StandardScaler()
		X_scaled = sc.fit_transform(X_2dT).T
		print("   Done standardizing!")
		print("   Data after: std", X_scaled.std(axis=1), "; mean", X_scaled.mean(axis=1))
	else:
		X_scaled = arrx2_float_data


	#PCA
	x,y = arrx2_float_data.shape
	print("\n2. Performing PCA fit ...")

	start = t()
	model = PCA(n_components = int_comp).fit(X_scaled)
	end = t()
	print("   PCA fit took:", end - start,"s")
	
	#Save pca fit model & scaler
	print("   Saving pca model and scaler to .pkl ...")
	pk.dump(model, open(str_saving_path + '/pca.pkl','wb'))
	if bool_standardize:
		pk.dump(sc, open(str_saving_path + '/scaler_pca.pkl','wb'))

	print("\n3. Apply dimensional reduction to data ..")
	s = t()
	X_pc = model.transform(X_scaled)
	e = t()
	print("   Transform took: ", e - s, "s")


	print("\n4. Saving encoded data ...")
	np.save(str_saving_path + "/PCA_encoded_"+str(int_comp)+"_comp.npy",X_pc)


	#Calculating explained variance
	print("\n5. Calculating individual and cumulative explained variance")
	exp_var_pca = model.explained_variance_ratio_
	cum_sum_eigenvalues = exp_var_pca.cumsum()
	print("   Individual explained variance: ",exp_var_pca,"\n   Cumulative explained variance: ",cum_sum_eigenvalues)

	#Create plot with explained variance
	b = plt.bar(range(0,len(exp_var_pca)),exp_var_pca, alpha=0.7, align ='center', 
			label='Individual explained variance')
	plt.step(range(0,len(cum_sum_eigenvalues)),cum_sum_eigenvalues,color="orange", where='mid', label='Cumulative explained variance')
	plt.ylabel('Explained variance ratio')
	plt.xlabel('Principal component index')

	for r in b:
		width = r.get_width()
		height = r.get_height()
		xi,yi = r.get_xy()
		perc = '{:.2f}%'.format(100 * r.get_height())
		plt.text(xi+width/2,  yi+height*1.01, str(perc),ha='center',size=8)

	plt.legend(loc='center right')
	plt.tight_layout()
	plt.xticks(np.arange(0, len(cum_sum_eigenvalues), 1.0))
	plt.savefig(str_saving_path + 'PCA_explained_variance.png')
	plt.close()


	#Calculate the most important features on the PCs
	print("\n6. Calculate the most important feature on the PCs with names")
	# number of components
	n_pcs= model.components_.shape[0]
	# get the index of the most important feature on each component i.e. largest absolute value (using list comprehensine here)
	most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
	initial_feature_names = list(np.arange(y))
	# get the names
	most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
	# using list comprehensine here
	dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
	df = pd.DataFrame(dic.items())
	print(df)

	return X_pc #return encoded data




def load_pca_model(arrx2_float_data,
					str_data_name: str = "test",
					int_comp: int = 16,
					str_saving_path: str = "",
					str_path_pca_model = None,
					bool_standardize: bool = True,
					str_path_scaler_model = None):

	"""
	Perform principle component analysis on test or
	validation data of shape (X*Y,spectra). The saved
	pca model will be loaded and the standard scaler if the
	data was standarized in the training process. 
	"""

	import os, sys
	import pickle as pk
	import numpy as np
	from sklearn.decomposition import PCA

	#Check files and path arguments, create saving dictionaries
	if not os.path.isdir(str_saving_path):
		sys.exit("The path", str_saving_path, "does not exist!")

	#Reshape input data if it's not shape (X*Y, channel)
	if len(arrx2_float_data.shape) >2:
		arrx2_float_data = arrx2_float_data.reshape(-1, arrx2_float_data.shape[-1])

	#Load pca model
	print("\n1. Load PCA model ..")
	if not str_path_pca_model:
		str_path_pca_model = str_saving_path+"/pca.pkl"
	pca_model = pk.load(open(str_path_pca_model, 'rb'))


	if bool_standardize:
		print("\n2. Load saved standard scaler and scale new data ...")
		if not str_path_scaler_model:
			str_path_scaler_model = str_saving_path + "/scaler_pca.pkl"
		sc = pk.load(open(str_path_scaler_model,'rb'))
		#Standardize each spectrum seperately - not each channel!
		#std_scaler = StandardScaler()
		X_2dT = arrx2_float_data.T
		X_scaled = sc.fit_transform(X_2dT).T

	print("\n3. Apply dimensional reduction to new data ..")
	X_pc = pca_model.transform(X_scaled)

	print("\n4. Saving encoded data ...")
	np.save(str_saving_path + "/PCA_encoded_"+ str_data_name +"_" + str(int_comp) +"_comp.npy", X_pc)

	return X_pc





if __name__ =="__main__":

	import numpy as np
	arr = np.random.randint(1,high=5, size=(2, 2, 3))
	print(arr.shape,"\n",arr[:,:,0],"\n",arr[:,:,1],"\n",arr[:,:,2])
	x,y,z = arr.shape

	import os
	cwd = os.getcwd()

	data = perform_pca(arr,2,cwd,bool_standardize=True)
	data = load_pca_model(arr,"test",2,str_saving_path=cwd, bool_standardize=True)

