"""
author: dajmue
date: last updated: Jan, 2023
"""

import sys, os, random, datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
import matplotlib.pyplot as plt


#----------------------- Eager execution -------------
#tf.config.run_functions_eagerly(False)
tf.compat.v1.disable_eager_execution()


#------------------------ Seed ------------------------ 
seed_value= 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)




class AE(Model):
	"""
	Stacked autoencoder (SAE) which trains each hidden layer seperately
	and uses the code layer as the input for the next autoencoder.
	Initialized with glorot_uniform and trained with a contractive
	loss.
	nb_name: = None: builds SAE with names as continious numbers e.g.: enc_0, enc_1
		 = str : builds SAE with names as given string (Warning: given string
					 has to differ for training of each SAE, e.g. AE(str(test_name)+str[i]), ....)
	layers: List of dict of hidden layers: [ {'n_nodes': 256 }, { 'n_nodes': 128 }]
	input_shape : training_data.shape[1]

	"""

	def __init__(self,nb_name, layers=None, input_shape=(427),**hyperparameters):
		super(Model, self).__init__(**hyperparameters)
		super(AE, self).__init__() # Necessary in tf >=2.3
		# Configure base (super) class
		#Model.__init__(self, hyperparameters, **hyperparameters)
		self.initializer = tf.compat.v1.keras.initializers.glorot_uniform(seed=0)
		self._layers = layers
		self._input_shape = input_shape
		self._target_layer = None
		self._nb_name = nb_name
		inputs = Input(input_shape, name='input_0')
		encoder = self.encoder(inputs, layers=layers)
		outputs = self.decoder(encoder, layers=layers)
		self._model = Model(inputs, outputs)
		self._enc = Model(inputs,encoder)


	def encoder(self,x,**metaparameters):
		layers = metaparameters['layers']
		for _ in range(len(layers)): 
			if not self._nb_name: 
				_name = '_'+str(_)
			else:
				_name = str(self._nb_name)
			#_name = 'enc' + _name
			n_nodes = layers[_]['n_nodes']
			x = Dense(n_nodes, name=('enc'+str(_name)),kernel_initializer=self.initializer)(x)
			x = BatchNormalization(name=('bn'+str(_name)))(x)
			x = Activation(activation='sigmoid')(x)
		self._target_layer ='enc'+str(_name)
		return x

	def decoder(self,x,**metaparameters):
		layers = metaparameters['layers']
		c = 0
		_name = str(self._nb_name)
		for _ in range(len(layers)-2, -1, -1):
			if not self._nb_name: 
				_name = '_'+str(_+1)
			#_name = 'dec' + str(_)
			n_nodes = layers[_]['n_nodes']
			x = Dense(n_nodes,name=('dec'+str(_name)), kernel_initializer=self.initializer)(x) #name=_name
			x = Activation(activation='sigmoid')(x)
			c = _
		if not self._nb_name:
			_name = "_"+str(c)
		outputs = Dense(self._input_shape,name=('dec'+str(_name)),activation='sigmoid')(x) #name="dec_last"
		return outputs


	def contractive_loss(self,y_pred, y_true):
		lam = 1e-3
		mse = K.mean(K.square(y_true - y_pred), axis=1)
		W = K.variable(value=(self._model.get_layer(self._target_layer)).get_weights()[0]) # N x N_hidden
		W = K.transpose(W)  # N_hidden x N
		h = self._model.get_layer(self._target_layer).output
		dh = h * (1 - h)  # N_batch x N_hidden
		contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1) # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
		return mse + contractive


def plot_history(history, str_saving_path, fname):
	# make loss plots for every submodel with matplotlib
	plt.semilogy(history.epoch, history.history['loss'])
	plt.semilogy(history.epoch, history.history['val_loss'], linestyle = "--")
	plt.title('SCAE Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(['Train', 'Vali'], loc = 'best')
	plt.savefig(str_saving_path+'/SCAE_Loss_code_{}.png'.format(str(fname), bbox_inches = 'tight'))
	plt.close()


def save_model(model,str_saving_path, str_fname = 'autoencoder_model'):

	model_json = model.to_json()
	json_fp = str_saving_path+"/"+str_fname +'.json'
	with open(json_fp, "w") as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5
	wt_fp = str_saving_path+"/"+str_fname +'.h5'
	model.save_weights(wt_fp)
	print("Saved %s to %s" % (str_fname, str_saving_path))
	return




def train_scae(arrx2_float_data_train = None, 
			  arrx2_float_data_val = None, 
			  arrx2_float_data_test = None,
			  int_epochs: int = 5,
			  early_stopping_epochs: int = 200,
			  batch_size: int = 50,
			  learning_rate: float = 0.003,
			  bool_l2_normalize_data: bool = False,
			  int_norm_axis: int = 1,
			  list_hidden_layers: dict = [ {'n_nodes': 256 }, { 'n_nodes': 128 }, { 'n_nodes': 64 },\
										   { 'n_nodes': 32 }, { 'n_nodes': 16 } ],
			  str_saving_path: str =".",
			  bool_show_summary: bool = False
			  ):
	''' 
	Main code to set parameters and build + train a stacked
	autoencoder. Shape of data has to be x*y,z e.g.:(4000,427)
	and can be L2 normalized if wanted. 

	Stacked Auotencoder:
	For each hidden layer a stacked autoencoder is build with just 
	one hidden layer, the weights will be saved and the training data
	will be predicted with the first trained SAE to serve as the 
	input for the next one. 
	SAE is trained with SGD and a contractive loss function (mse
	with an additive regularization term).

	Further comments included for debugging purpose.

	'''

	#Check files and path arguments, create saving dictionaries
	if not os.path.isdir(str_saving_path):
		sys.exit("The path", str_saving_path, "does not exist!")

	if not type(arrx2_float_data_train).__module__ == "numpy" and type(arrx2_float_data_val).__module__ == "numpy":
		sys.exit("Training and/or validation data has to be a numpy array")

	#Reshape input data if it's not shape (X*Y, channel)
	if len(arrx2_float_data_train.shape)>2:
		arrx2_float_data_train = arrx2_float_data_train.reshape(-1,arrx2_float_data_train.shape[-1])
	if len(arrx2_float_data_val.shape)>2:
		arrx2_float_data_val = arrx2_float_data_val.reshape(-1,arrx2_float_data_val.shape[-1])
	if arrx2_float_data_test is not None:
		if len(arrx2_float_data_test.shape)>2:
			arrx2_float_data_test = arrx2_float_data_test.reshape(-1,arrx2_float_data_test.shape[-1])



	if not os.path.isdir(str_saving_path+"/logs/"):
		os.mkdir(str_saving_path+"/logs/")
	if not os.path.isdir(str_saving_path+"/weights/"):
		os.mkdir(str_saving_path+"/weights/")


	print("Shape train: %s - Shape val: %s " % (str(arrx2_float_data_train.shape),\
												str(arrx2_float_data_val.shape)))

	# L2 normalize data if needed
	if bool_l2_normalize_data:
		import sklearn.preprocessing as skp
		arrx2_float_data_train =  skp.normalize(arrx2_float_data_train, norm='l2', axis=int_norm_axis, copy=True, return_norm=False)
		arrx2_float_data_val   =  skp.normalize(arrx2_float_data_val  , norm='l2', axis=int_norm_axis, copy=True, return_norm=False)


	#Iterate over every hidden layer, train model with one hidden layer each, save the weights in a dict to build the full autoencoder later
	dict_weights = {}
	dict_weights_ae = {}
	input_x = None

	for hl in range(0,len(list_hidden_layers)):
		if hl < 1:
			input_x = arrx2_float_data_train
			val_x = arrx2_float_data_val
		else:
			input_x = input_x
			val_x = val_x

		nb_name = "_"+str(hl)
		ae = AE(nb_name,list_hidden_layers[hl:hl+1],input_x.shape[1])
		autoencoder = ae._model
		encoder = ae._enc

		#print the network architecture if of every autoencoder and encoder
		if bool_show_summary:
			print("Encoder:", encoder.summary(), "\n")
			print("Decoder:", autoencoder.summary(),"\n")

		opt = tf.keras.optimizers.SGD(lr=learning_rate)
		autoencoder.compile(optimizer=opt, loss=ae.contractive_loss)

		#Create a unique id for each model and tensorboard
		log_dir = str_saving_path+"/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		checkpoint_path = str_saving_path+"/weights"+"/model_"+str(hl)+"_{epoch:04d}_{val_loss:.8f}.h5"

		#Callbacks to model the training
		my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=early_stopping_epochs),
					tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=70,verbose=1,mode="auto",min_delta=0.0000001, cooldown=0,min_lr=0),
					tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1, monitor='val_loss',mode='min',save_best_only=True),
					tf.keras.callbacks.TensorBoard(log_dir=log_dir)]


		#train the autoencoder with model.fit
		history = autoencoder.fit(input_x,input_x,
							epochs=int_epochs,
							batch_size=batch_size,
							shuffle=True,
							validation_data=(val_x, val_x),
							callbacks=my_callbacks,
							verbose=2)


		plot_history(history,str_saving_path, list_hidden_layers[hl:hl+1][0]["n_nodes"])

		#Save the weights for the encoder layers in a dict
		for l in encoder.layers:
			if 'enc' in l.name or 'bn' in l.name:
				_id = l.name
				dict_weights[_id] = l.get_weights()
				#print("Layer", l.name, " Shape",l.get_weights()[0].shape) 
				#print(l,l.get_weights()[0][0][0:5])

		#Save the weights for all layers in a dict
		for l in autoencoder.layers:
			if 'enc' in l.name or 'bn' in l.name or 'dec' in l.name:
				_id = l.name
				dict_weights_ae[_id] = l.get_weights()

		input_x = encoder.predict(input_x)
		val_x = encoder.predict(val_x)


	#Creating the autoencoder and decoder with all the layers 
	nb_name = None
	ae_new = AE(nb_name,list_hidden_layers,arrx2_float_data_train.shape[1])
	enc = ae_new._enc
	model = ae_new._model
	enc.summary()
	model.summary()


	# Set all the weights in the final encoder model
	for l in enc.layers:
		if l.name in dict_weights.keys():
			wb = dict_weights[l.name]
			l.set_weights(wb)

	# Set all the weights in the final autoencoder model
	for l in model.layers:
		if l.name in dict_weights_ae.keys():
			wb = dict_weights_ae[l.name]
			l.set_weights(wb)


	# Save model
	save_model(enc,str_saving_path,str_fname='stacked_encoder_model')
	save_model(model,str_saving_path,str_fname='stacked_autoencoder_model')
	enc.save(str_saving_path+"/model_stacked_encoder")
	model.save(str_saving_path+"/model_stacked_autoencoder")


	#Normalize and predict test data if given as an argument
	if arrx2_float_data_test is not None:
		if bool_l2_normalize_data:
			arrx2_float_data_test =  skp.normalize(arrx2_float_data_test, norm='l2', axis=int_norm_axis, copy=True, return_norm=False)

		encoded_imgs = enc.predict(arrx2_float_data_test)
		decoded_imgs = model.predict(arrx2_float_data_test)
		np.save(str_saving_path+"/stacked_encoder_results.npy", encoded_imgs)
		np.save(str_saving_path+"/stacked_decoder_results.npy", decoded_imgs)

	return enc, model



if __name__ == "__main__":
	arr = np.random.rand(200, 427)
	train_scae(arr,arr, list_hidden_layers = [ {'n_nodes': 256 }, { 'n_nodes': 128 },{ 'n_nodes': 64 }])
