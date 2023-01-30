"""
author: dajmue
date: last updated: Jan, 2023
"""

import sys, os, random, datetime

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Activation


# GPU settings - only take the first gpu
os.environ["VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), " Physical GPUs, ",len(logical_gpus)," Logical GPUs")
    except RuntimeError as e:
        print(e)



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

	def __init__(self, layers=None, input_shape=(427), **hyperparameters):
		super(Model, self).__init__(**hyperparameters)
		# Configure base (super) class
		#Model.__init__(self, hyperparameters, **hyperparameters)
		self.initializer = tf.compat.v1.keras.initializers.glorot_uniform(seed=0)
		self._layers = layers
		self._input_shape = input_shape
		self._target_layer = None
		inputs = Input(input_shape, name='input_0')
		encoder = self.encoder(inputs, layers=layers)
		outputs = self.decoder(encoder, layers=layers)
		self._model = Model(inputs, outputs)
		self._enc = Model(inputs,encoder)

	def encoder(self,x,**metaparameters):
		layers = metaparameters['layers']
		for _ in range(len(layers)): 
			_name =  'enc_' + str(_)
			n_nodes = layers[_]['n_nodes']
			x = Dense(n_nodes, name=_name,kernel_initializer=self.initializer)(x)
			x = BatchNormalization()(x)
			x = Activation(activation='sigmoid')(x)
		self._target_layer = _name
		return x

	def decoder(self,x,**metaparameters):
		layers = metaparameters['layers']
		for _ in range(len(layers)-2, -1, -1):
			n_nodes = layers[_]['n_nodes']
			x = Dense(n_nodes,kernel_initializer=self.initializer)(x)
			x = BatchNormalization()(x)
			x = Activation(activation='sigmoid')(x)
		outputs = Dense(self._input_shape,activation='sigmoid')(x)
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
	#Save weights and model again
	model_json = model.to_json()
	json_fp = str_saving_path+"/"+str_fname +'.json'
	with open(json_fp, "w") as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5
	wt_fp = str_saving_path+"/"+str_fname +'.h5'
	model.save_weights(wt_fp)
	print("Saved %s to %s" % (str_fname, str_saving_path))
	return




def train_fccae(arrx2_float_data_train = None, 
				arrx2_float_data_val = None, 
				arrx2_float_data_test = None,
				bool_model_avail = False,
				str_path_to_weights = ".",
				int_epochs: int = 5,
				int_epoch_start: int = 0,
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
	try:
		if len(arrx2_float_data_test.shape)>2:
			arrx2_float_data_test = arrx2_float_data_test.reshape(-1,arrx2_float_data_test.shape[-1])
	except:
		pass


	if not os.path.isdir(str_saving_path+"/logs/"):
		os.mkdir(str_saving_path+"/logs/")
	if not os.path.isdir(str_saving_path+"/weights/"):
		os.mkdir(str_saving_path+"/weights/")



	print("\nShape train: %s - Shape val: %s \n" % (str(arrx2_float_data_train.shape),\
												str(arrx2_float_data_val.shape)))

	# L2 normalize data if needed
	if bool_l2_normalize_data:
		import sklearn.preprocessing as skp
		arrx2_float_data_train =  skp.normalize(arrx2_float_data_train, norm='l2', axis=int_norm_axis, copy=True, return_norm=False)
		arrx2_float_data_val   =  skp.normalize(arrx2_float_data_val  , norm='l2', axis=int_norm_axis, copy=True, return_norm=False)



	ae = AE(list_hidden_layers,(arrx2_float_data_train.shape[1]))
	autoencoder = ae._model
	encoder = ae._enc

	if bool_show_summary:
		print(autoencoder.summary())
		print(encoder.summary())

	opt = tf.keras.optimizers.SGD(lr=learning_rate)
	autoencoder.compile(optimizer=opt, loss=ae.contractive_loss)


	if bool_model_avail:
		print("\nWeights are available. \
			\nLoad weights from %s and start training at epoch %s\n" % (str_path_to_weights, int_epoch_start))
		autoencoder.load_weights(str_path_to_weights) #works but only loads weights 


	log_dir = str_saving_path+"/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	checkpoint_path = str_saving_path+"/weights"+"/model_{epoch:04d}_{val_loss:.8f}.h5"


	my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=early_stopping_epochs),
					tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=70,verbose=1,mode="auto",min_delta=0.0000001, cooldown=0,min_lr=0),
					tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,verbose=1, monitor='val_loss',mode='min',save_best_only=True),
					tf.keras.callbacks.TensorBoard(log_dir=log_dir)]


	history = autoencoder.fit(arrx2_float_data_train, arrx2_float_data_train,
	 						epochs=int_epochs,
	 						batch_size=batch_size,
	 						initial_epoch = int_epoch_start,
	 						shuffle=True,
	 						validation_data=(arrx2_float_data_val, arrx2_float_data_val),
	 						callbacks=my_callbacks,
							verbose=2)


	# make matplotlib loss plots for every submodel
	plot_history(history,str_saving_path, "fccae")
	np.save((str_saving_path+'/history_fccae.npy'),history.history)


	# Save model
	save_model(encoder,str_saving_path,str_fname='encoder_model')
	save_model(autoencoder,str_saving_path,str_fname='autoencoder_model')
	encoder.save(str_saving_path+"/model_encoder")
	autoencoder.save(str_saving_path+"/model_autoencoder")


	#Normalize and predict test data if given as an argument
	if arrx2_float_data_test:
		if bool_l2_normalize_data:
			arrx2_float_data_test =  skp.normalize(arrx2_float_data_test, norm='l2', axis=int_norm_axis, copy=True, return_norm=False)

		encoded_imgs = encoder.predict(arrx2_float_data_test)
		decoded_imgs = autoencoder.predict(arrx2_float_data_test)
		np.save(str_saving_path+"/encoder_results.npy", encoded_imgs)
		np.save(str_saving_path+"/decoder_results.npy", decoded_imgs)

	return encoder, autoencoder




if __name__=="__main__":

	arr = np.random.rand(200, 427)
	train_fccae(arr,arr, list_hidden_layers = [ {'n_nodes': 256 }, { 'n_nodes': 128 },{ 'n_nodes': 64 }])

