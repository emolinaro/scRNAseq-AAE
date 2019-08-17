from keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, LeakyReLU, Softmax, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras import regularizers
from keras import backend as K
from keras.activations import softmax
from keras.losses import mse
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from umap import UMAP

from scanpy import read_h5ad
import matplotlib.pyplot as plt


def sampling(args):
	"""Reparameterization trick by sampling from an isotropic unit Gaussian.

		  Instead of sampling from Q(z|X), sample epsilon = N(0,I)
		  z = z_mean + sqrt(var) * epsilon

		:param  args:
			mean and log of variance of Q(z|X)
		:return:
			sampled latent vector
	"""

	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	# by default, random_normal has mean = 0 and std = 1.0
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon


def sampling_gumbel(logits_y, tau):
	"""Gumbel trick to sample from a discrete distribution.

	The original discrete distribution is replaced with a deterministic transformation of
	the stardard (parameterless) Gumbel distribution.

	:param logits_y:
		logits of the discrete distribution
	:type logit_y: tensorflow.python.framework.ops.Tensor

	:param tau:
		temperature parameter
	:type tau: float

	:return:
		tensor object with one-hot encoding of the discrete variable
	"""

	G = K.random_uniform(K.shape(logits_y), 0, 1)

	# logits of y + Gumbel noise
	y = logits_y - K.log(-K.log(G + 1e-20) + 1e-20)

	# apply softmax to approximate the argmax function
	tau_var = K.variable(tau, name="temperature")

	# Gumbel-Softmax estimator
	y = softmax(y / tau_var)

	return y


def sampling_cat(batch, dim):
	"""Draw samples from a multinomial distribution and return a one-hot encoded representation.


	:param batch:
		number of samples to bbe generated
	:type batch: int
	:param dim:
		dimension of the labels
	:type dim: int
	:return:
		matrix of one-hot encoded arrays (batch x dim)
	"""

	# TODO: function not used.

	counts = np.random.multinomial(10000, [1. / dim] * dim, size=batch)
	labels = np.argmax(counts, axis=-1)
	labels_code = to_categorical(labels).astype(int)

	if labels_code.shape[1] < dim:
		print("Not all categories were drawn. Resampling after adding the last category.")

		labels = np.append(labels, dim - 1)
		np.random.shuffle(labels)
		labels_code = to_categorical(labels).astype(int)

	return labels_code


def OneHot(num_classes=None, input_length=None, name='ohe_layer'):
	"""Define Keras layer to implement one-hot encoding of the imput labels.

	:param num_classes:
		number of classes
	:type num_classes: int
	:param input_length:
		number of data points
	:type input_length: int
	:param name:
		layer name
	:type name: str
	:return:
		tensor
	"""
	# TODO: function not used.

	# Check if inputs were supplied correctly
	if num_classes is None or input_length is None:
		raise TypeError("num_classes or input_length is not set")

	# Helper method (not inlined for clarity)
	def _one_hot(x, num_classes):
		return K.one_hot(K.cast(x, 'uint8'), num_classes=num_classes)

	# Final layer representation as a Lambda layer
	return Lambda(_one_hot, arguments={'num_classes': num_classes}, input_shape=(input_length,), name=name)


##########################################
############### BASE MODEL ###############
##########################################
class Base():
	"""Base Model Class.

		The class initialize parameters for the Adversarial and Variational Autoencoder Models.
		Overriding implemented for some of the methods in the child classes.

	Attributes
	----------
	original_dim: int
		number of cells in the dataset
	latent_dim: int
		dimension of the latent space Z
	layers_enc_dim: list
		array containing the dimension of encoder network dense layers
	layers_dec_dim: list
		array containing the dimension of decoder network dense layers
	layers_dis_dim: list
		array containing the dimension of discriminator network dense layers
	layers_dis_cat_dim: list
		array containing the dimension of categorical discriminator network dense layers
	alpha: float
		alpha parameter of LeackyReLU activation function
	do_rate: float
		dropout rate
	kernel_initializer: str
		kernel initializer of all dense layers
	bias_initializer: str
		bias initializer of all dense layers
	l2_weight: float
		weight parameter of l2 kernel regularization
	l1_weight: float
		weight parameter of l1 activity regularization
	batch_size: int
		batch size during training
	epochs: int
		number of epochs during training
	lr_dis: float
		learning rate discriminator optimizer
	lr_gen: float
		learning rate generator optimizer
	lr_ae: float
		learning rate autoencoder optimizer
	dr_dis: float
		decay rate discriminator optimizer
	dr_gen: float
		decay rate generator optimizer
	dr_ae: float
		decay rate autoencoder optimizer
	encoder: keras.engine.training.Model
		encoder deep neural network
	decoder: keras.engine.training.Model
		decoder deep neural network
	generator: keras.engine.training.Model
		generator deep neural network
	discriminator: keras.engine.training.Model
		discriminator deep neural network
	rec_loss: float
		reconstruction loss
	gen_loss: float
		generator loss
	dis_loss: float
		discriminator loss
	data: numpy.ndarray
		matrix containing gene expression
	gene_list: list
		list of gene names
	labels: list
		list of integers labelling cell subgroups

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	"""

	def __init__(self,
	             latent_dim=None,
	             layers_enc_dim=None,
	             layers_dec_dim=None,
	             layers_dis_dim=None,
	             alpha=0.1,
	             do_rate=0.1,
	             kernel_initializer='glorot_uniform',
	             bias_initializer='zeros',
	             l2_weight=0.01,
	             l1_weight=0.01,
	             batch_size=35,
	             epochs=50,
	             lr_dis=0.0001,
	             lr_gen=0.0001,
	             lr_ae=0.0002,
	             dr_dis=1e-6,
	             dr_gen=1e-6,
	             dr_ae=1e-6):

		self.latent_dim = latent_dim
		self.layers_enc_dim = layers_enc_dim
		self.layers_dec_dim = layers_dec_dim
		self.layers_dis_dim = layers_dis_dim
		self.alpha = alpha
		self.do_rate = do_rate
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.l2_weight = l2_weight
		self.l1_weight = l1_weight
		self.batch_size = batch_size
		self.epochs = epochs
		self.lr_dis = lr_dis
		self.lr_gen = lr_gen
		self.lr_ae = lr_ae
		self.dr_dis = dr_dis
		self.dr_gen = dr_gen
		self.dr_ae = dr_ae
		self.encoder = None
		self.decoder = None
		self.generator = None
		self.discriminator = None
		self.autoencoder = None
		self.rec_loss = None
		self.gen_loss = None
		self.dis_loss = None
		self.data = None
		self.gene_list = None
		self.labels = None

		# create a dictionary with all networks' parameters
		self._get_parameters()

	def _get_parameters(self):
		"""Create a dictionary with network parameter.

		"""

		# construct a dictionary with the list of network parameters

		self.dict = {"Parameter": [], "Value": [], "Description": []}

		self.dict["Parameter"] = np.hstack(['batch_size',
		                                    'epochs',
		                                    'alpha',
		                                    'do_rate',
		                                    'kernel_initializer',
		                                    'bias_initializer',
		                                    'l2_weight',
		                                    'l1_weight',
		                                    'latent_dim',
		                                    ['layer_' + str(k + 1) + '_enc_dim' for k in
		                                     range(len(self.layers_enc_dim))],
		                                    ['layer_' + str(k + 1) + '_dec_dim' for k in
		                                     range(len(self.layers_dec_dim))],
		                                    'lr_ae',
		                                    'dr_ae',
		                                    ['layer_' + str(k + 1) + '_dis_dim' for k in
		                                     range(len(self.layers_dis_dim))],
		                                    'lr_dis',
		                                    'dr_dis',
		                                    'lr_gen',
		                                    'dr_gen',
		                                    ])

		self.dict["Value"] = np.hstack([self.batch_size,
		                                self.epochs,
		                                self.alpha,
		                                self.do_rate,
		                                self.kernel_initializer,
		                                self.bias_initializer,
		                                self.l2_weight,
		                                self.l1_weight,
		                                self.latent_dim,
		                                self.layers_enc_dim,
		                                self.layers_dec_dim,
		                                self.lr_ae,
		                                self.dr_ae,
		                                self.layers_dis_dim,
		                                self.lr_dis,
		                                self.dr_dis,
		                                self.lr_gen,
		                                self.dr_gen
		                                ])

		self.dict["Description"] = np.hstack(["batch size",
		                                      "number of epochs",
		                                      "alpha coeff. in activation function",
		                                      "dropout rate",
		                                      "kernel initializer of all dense layers",
		                                      "bias initializer of all dense layers",
		                                      "weight of l2 kernel regularization",
		                                      "weight of l1 activity regularization",
		                                      "dimension of latent space Z",
		                                      ["dimension of encoder dense layer " + str(k + 1) for k in
		                                       range(len(self.layers_enc_dim))],
		                                      ["dimension of decoder dense layer " + str(k + 1) for k in
		                                       range(len(self.layers_dec_dim))],
		                                      "learning rate of autoencoder",
		                                      "decay rate of autoencoder",
		                                      ["dimension of discriminator dense layer " + str(k + 1) for k in
		                                       range(len(self.layers_dis_dim))],
		                                      "learning rate of discriminator",
		                                      "decay rate of discriminator",
		                                      "learning rate of generator",
		                                      "decay rate of generator"
		                                      ])

	def get_parameters(self):
		"""Print the list of network parameter.

		Returns
		-------
		Pandas dataframe object
		"""

		dataframe = pd.DataFrame(self.dict, index=self.dict["Parameter"]).drop(columns=['Parameter'])

		return dataframe

	def load_data(self, datapath):

		"""Read data file and initialize cell gene counts, gene name list and cell subgroups.

		Data file is a Scanpy AnnData object saved in h5ad format.
		This object contains cell subgoups obtained using Louvain algorithm.

		:param str datapath:
			path to data file (h5ad format)
		:return:
		:raises NameError: if Louvain clustering is not defined in the dataset
		"""

		adata = read_h5ad(datapath)

		self.data = adata.X

		self.original_dim = self.data.shape[1]

		self.gene_list = adata.var_names.values

		try:
			self.labels = adata.obs['louvain'].values.astype(int).tolist()
		except:
			print("Louvain clustering not defined in this dataset.")
			self.labels = None

		print("Dataset imported.")

	def rescale_data(self):

		"""Standardize gene expression counts by removing the mean and scaling to unit variance.

		:return:
		"""

		scaler = StandardScaler(with_mean=True, with_std=True)
		scaler.fit(self.data)
		self.data = scaler.transform(self.data)

		print("Gene expression data rescaled.")

	def get_summary(self):

		"""Print Adversarial Autoencoder model summary.

		:return:
		"""

		print("\nEncoder Network")
		print("===============")
		self.encoder.summary()

		print("\nDecoder Network")
		print("===============")
		self.decoder.summary()

		print("\nAutoencoder Network")
		print("===================")
		self.autoencoder.summary()

		print("\nGenerator Network")
		print("=================")
		# Freeze the discriminator weights during training of generator
		self.generator.summary()

		print("\nDiscriminator Network")
		print("=====================")
		self.discriminator.trainable = True  # note: model already compiled with trainable weights
		self.discriminator.summary()

	def export_graph(self, filepath):

		"""Save model graphs to PNG images.

		:param filepath:
			 path of output image files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		plot_model(self.encoder, to_file=filepath + "encoder.png", show_shapes=True)
		plot_model(self.decoder, to_file=filepath + "decoder.png", show_shapes=True)
		plot_model(self.autoencoder, to_file=filepath + "autoencoder.png", show_shapes=True)
		plot_model(self.generator, to_file=filepath + "generator.png", show_shapes=True)
		plot_model(self.discriminator, to_file=filepath + "discriminator.png", show_shapes=True)

		print("Model graphs saved.\n")

	def plot_umap(self, gene_selected, louvain=False):

		"""Plot the gene expression of selected genes in a 2-D latent space using the UMAP clustering algorithm.

		:param gene_selected:
			list of genes
		:type gene_selected: list
		:param louvain:
			if true, show label the 2-D latent space according Louvain clustering of cell data
		:type louvain: bool
		:return:
			figure
		"""

		z_mean = self.encoder.predict(self.data, batch_size=self.batch_size)[0]

		reducer = UMAP()
		z_mean = reducer.fit_transform(z_mean)

		for name in gene_selected:

			idx_name = np.where(self.gene_list == name)[0].tolist()[0]
			subset = self.data[:, idx_name]

			if louvain:

				plt.figure(figsize=(14, 5))

				plt.subplot(1, 2, 1)
				cmap = plt.get_cmap('viridis')  # RdBu
				plt.scatter(z_mean[:, 0], z_mean[:, 1],
				            c=subset,
				            cmap=cmap,
				            vmin=np.min(subset),
				            vmax=np.max(subset),
				            s=5)

				plt.colorbar()
				plt.title(name)
				plt.xlabel("z[0]")
				plt.ylabel("z[1]")

				plt.subplot(1, 2, 2)

				cmap2 = plt.get_cmap('tab20', np.max(self.labels) - np.min(self.labels) + 1)
				plt.scatter(z_mean[:, 0], z_mean[:, 1],
				            c=self.labels,
				            cmap=cmap2,
				            vmin=np.min(self.labels) - .5,
				            vmax=np.max(self.labels) + .5,
				            s=5)

				plt.colorbar()
				plt.title('Louvain Clustering')
				plt.xlabel("z[0]")
				plt.ylabel("z[1]")

				plt.tight_layout()
				plt.show()

			else:

				cmap = plt.get_cmap('viridis')  # RdBu
				plt.figure(figsize=(7, 5))
				plt.scatter(z_mean[:, 0], z_mean[:, 1],
				            c=subset,
				            cmap=cmap,
				            vmin=np.min(subset),
				            vmax=np.max(subset),
				            s=5)

				plt.colorbar()
				plt.title(name)
				plt.xlabel("z[0]")
				plt.ylabel("z[1]")

				plt.show()

	def export_model(self, filepath):

		"""Export the network models in h5 format.

		:param filepath:
			path of model files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		self.autoencoder.save(filepath + 'autoencoder.h5')
		self.generator.save(filepath + 'generator.h5')
		self.discriminator.save(filepath + 'discriminator.h5')
		self.encoder.save(filepath + 'encoder.h5')
		self.decoder.save(filepath + 'decoder.h5')

		print("All networks exported in h5 format.")


##########################################
############### VAE MODEL ################
##########################################
class VAE(Base):
	""" Unsupervised clustering with variational autoencoder model.

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	Raises
	------
	TypeError
		If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim.
	"""

	def __init__(self, **kwargs):
		super(VAE, self).__init__(**kwargs)

		if self.latent_dim is None or \
				self.layers_enc_dim is None or \
				self.layers_dec_dim is None:
			raise TypeError(
				"List of mandatory arguments: latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

		# create a dictionary with all networks' parameters
		self._get_parameters()

	def _get_parameters(self):
		"""Create a dictionary with network parameter.

		"""

		# construct a dictionary with the list of network parameters

		self.dict = {"Parameter": [], "Value": [], "Description": []}

		self.dict["Parameter"] = np.hstack(['batch_size',
		                                    'epochs',
		                                    'alpha',
		                                    'do_rate',
		                                    'kernel_initializer',
		                                    'bias_initializer',
		                                    'l2_weight',
		                                    'l1_weight',
		                                    'latent_dim',
		                                    ['layer_' + str(k + 1) + '_enc_dim' for k in
		                                     range(len(self.layers_enc_dim))],
		                                    ['layer_' + str(k + 1) + '_dec_dim' for k in
		                                     range(len(self.layers_dec_dim))],
		                                    'lr_ae',
		                                    'dr_ae'
		                                    ])

		self.dict["Value"] = np.hstack([self.batch_size,
		                                self.epochs,
		                                self.alpha,
		                                self.do_rate,
		                                self.kernel_initializer,
		                                self.bias_initializer,
		                                self.l2_weight,
		                                self.l1_weight,
		                                self.latent_dim,
		                                self.layers_enc_dim,
		                                self.layers_dec_dim,
		                                self.lr_ae,
		                                self.dr_ae
		                                ])

		self.dict["Description"] = np.hstack(["batch size",
		                                      "number of epochs",
		                                      "alpha coeff. in activation function",
		                                      "dropout rate",
		                                      "kernel initializer of all dense layers",
		                                      "bias initializer of all dense layers",
		                                      "weight of l2 kernel regularization",
		                                      "weight of l1 activity regularization",
		                                      "dimension of latent space Z",
		                                      ["dimension of encoder dense layer " + str(k + 1) for k in
		                                       range(len(self.layers_enc_dim))],
		                                      ["dimension of decoder dense layer " + str(k + 1) for k in
		                                       range(len(self.layers_dec_dim))],
		                                      "learning rate of autoencoder",
		                                      "decay rate of autoencoder"
		                                      ])

	def get_summary(self):

		"""Print Variational Autoencoder model summary.

		:return:
		"""

		print("\nEncoder Network")
		print("===============")
		self.encoder.summary()

		print("\nDecoder Network")
		print("===============")
		self.decoder.summary()

		print("\nAutoencoder Network")
		print("===================")
		self.autoencoder.summary()

	def export_graph(self, filepath):

		"""Save model graphs to PNG images.

		:param filepath:
			 path of output image files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		plot_model(self.encoder, to_file=filepath + "encoder.png", show_shapes=True)
		plot_model(self.decoder, to_file=filepath + "decoder.png", show_shapes=True)
		plot_model(self.autoencoder, to_file=filepath + "autoencoder.png", show_shapes=True)

		print("Model graphs saved.\n")

	def export_model(self, filepath):

		"""Export the network models in h5 format.

		:param filepath:
			path of model files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		self.autoencoder.save(filepath + 'autoencoder.h5')
		self.encoder.save(filepath + 'encoder.h5')
		self.decoder.save(filepath + 'decoder.h5')

		print("All networks exported in h5 format.")

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer
		# TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_enc_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		z_mean = Dense(self.latent_dim,
		               name='z_mean',
		               kernel_initializer=self.kernel_initializer,
		               bias_initializer=self.bias_initializer)(x)

		z_log_var = Dense(self.latent_dim,
		                  name='z_log_var',
		                  kernel_initializer=self.kernel_initializer,
		                  bias_initializer=self.bias_initializer)(x)

		z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

		# instantiate encoder model
		encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

		return encoder

	def _build_decoder(self):

		"""Build decoder neural network.

		:return:
			decoder
		"""

		# TODO: check impact of kernel and activity regularizer

		decoder_input = Input(shape=(self.latent_dim,), name='Z')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dec_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model(decoder_input, x, name='decoder')

		return decoder

	def build_model(self):

		"""Build Variational Autoencoder model architecture.

		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		# TODO: implement optimizer_ae = SGD(lr=0.001, decay=1e-6, momentum=0.9)

		encoder_input = Input(shape=(self.original_dim,), name='X')

		# build encoder
		self.encoder = self._build_encoder()

		# build decoder
		self.decoder = self._build_decoder()

		# build and compile variational autoencoder
		real_input = encoder_input
		compression = self.encoder(real_input)[2]
		reconstruction = self.decoder(compression)
		self.autoencoder = Model(real_input, reconstruction, name='autoencoder')

		# expected negative log-likelihood of the ii-th datapoint (reconstruction loss)
		reconstruction_loss = mse(real_input, reconstruction)
		reconstruction_loss *= self.original_dim

		# add regularizer: Kullback-Leibler divergence between the encoder’s distribution Q(z|x) and p(z)
		z_mean = self.encoder(real_input)[0]
		z_log_var = self.encoder(real_input)[1]
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = K.sum(kl_loss, axis=-1)
		kl_loss *= -0.5
		ae_loss = K.mean(reconstruction_loss + kl_loss)

		self.autoencoder.add_loss(ae_loss)
		self.autoencoder.compile(optimizer=optimizer_ae, metrics=['accuracy'])

	def train(self, val_split=0.2):

		"""Training of the Variational Autoencoder.

			The training will stop if there is no change in the validation loss after 30 epochs.

		:param val_split:
			fraction of data used for validation
		:type val_split: float

		:return:
			lists containing training loss and validation loss
		"""

		print("Start model training...")

		# def checkpoint(file):
		#
		# 	checkpoint = ModelCheckpoint(file, monitor='accuracy', save_weights_only=True, verbose=0)
		#
		# 	return checkpoint

		vae_history = self.autoencoder.fit(self.data,
		                                   epochs=self.epochs,
		                                   batch_size=self.batch_size,
		                                   validation_split=val_split,
		                                   callbacks=[  # checkpoint('../models/vae_weights.hdf5'),
			                                   EarlyStopping(monitor='val_loss',
			                                                 patience=30,
			                                                 verbose=0,
			                                                 mode='min',
			                                                 baseline=None,
			                                                 restore_best_weights=False)
		                                   ],
		                                   verbose=1)

		print("Training completed.")

		loss = vae_history.history["loss"]
		val_loss = vae_history.history["val_loss"]

		return loss, val_loss


##########################################
############### MODEL n.1 ################
##########################################
class AAE1(Base):
	""" Unsupervised adversarial autoencoder model.

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	Raises
	------
	TypeError
		If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim, layers_dis_dim.
	"""

	def __init__(self, **kwargs):
		super(AAE1, self).__init__(**kwargs)

		if self.latent_dim is None or \
				self.layers_enc_dim is None or \
				self.layers_dec_dim is None or \
				self.layers_dis_dim is None:
			raise TypeError(
				"List of mandatory arguments: latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer
		# TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_enc_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          # kernel_regularizer=regularizers.l2(self.l2_weight),
			          # activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		z_mean = Dense(self.latent_dim,
		               name='z_mean',
		               kernel_initializer=self.kernel_initializer,
		               bias_initializer=self.bias_initializer)(x)

		z_log_var = Dense(self.latent_dim,
		                  name='z_log_var',
		                  kernel_initializer=self.kernel_initializer,
		                  bias_initializer=self.bias_initializer)(x)

		z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

		# instantiate encoder model
		encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

		return encoder

	def _build_decoder(self):

		"""Build decoder neural network.

		:return:
			decoder
		"""

		# TODO: check impact of kernel and activity regularizer

		decoder_input = Input(shape=(self.latent_dim,), name='Z')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dec_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model(decoder_input, x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network.

		:return:
			discriminator
		"""
		# TODO: check impact of kernel and activity regularizer

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		discr_input = latent_input

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dis_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model(latent_input, x, name='discriminator')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def _build_generator(self, input_encoder, compression, discriminator):

		"""Build generator neural network.

		:param input_encoder:
			encoder input layer
		:param compression:
			encoder transformation
		:param discriminator:
			initialized discriminator model
		:return:
			generator
		"""

		optimizer_gen = Adam(lr=self.lr_gen, decay=self.dr_gen)

		# keep discriminator weights frozen
		discriminator.trainable = False

		generation = discriminator(compression)

		# instantiate and compile generator model
		generator = Model(input_encoder, generation)
		generator.compile(optimizer=optimizer_gen, loss="binary_crossentropy", metrics=['accuracy'])

		return generator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture.

		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		# optimizer_aae = SGD(lr=0.001, decay=1e-6, momentum=0.9)

		encoder_input = Input(shape=(self.original_dim,), name='X')

		# build encoder
		self.encoder = self._build_encoder()

		# build decoder
		self.decoder = self._build_decoder()

		# build and compile discriminator
		self.discriminator = self._build_discriminator()

		# build and compile autoencoder
		real_input = encoder_input
		compression = self.encoder(real_input)[2]
		reconstruction = self.decoder(compression)
		self.autoencoder = Model(real_input, reconstruction, name='autoencoder')
		self.autoencoder.compile(optimizer=optimizer_ae, loss='mse')

		# build and compile generator model
		self.generator = self._build_generator(real_input,
		                                       self.encoder(real_input)[2],
		                                       self.discriminator)

	def train(self, graph=False, gene=None):

		"""Training of the Adversarial Autoencoder.

		During the reconstruction phase the training of the generator proceeds with the
		discriminator weights frozen.

		:param graph:
			if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
		:type graph: bool
		:param gene:
			selected gene
		:type gene: str
		:return:
			lists containing reconstruction loss, generator loss, and discriminator loss at each epoch
		"""

		rec_loss = []
		gen_loss = []
		dis_loss = []

		val_split = 0.0

		print("Start model training...")

		for epoch in range(self.epochs):
			np.random.shuffle(self.data)

			for i in range(int(len(self.data) / self.batch_size)):
				batch = self.data[i * self.batch_size:i * self.batch_size + self.batch_size]

				# Reconstruction phase
				autoencoder_history = self.autoencoder.fit(x=batch,
				                                           y=batch,
				                                           epochs=1,
				                                           batch_size=self.batch_size,
				                                           validation_split=val_split,
				                                           verbose=0)

				# Regularization phase
				fake_pred = self.encoder.predict(batch)[2]
				real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
				discriminator_batch_x = np.concatenate([fake_pred, real_pred])
				discriminator_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
				                                        np.random.uniform(0.0, 0.1, self.batch_size)])

				discriminator_history = self.discriminator.fit(x=discriminator_batch_x,
				                                               y=discriminator_batch_y,
				                                               epochs=1,
				                                               batch_size=self.batch_size,
				                                               validation_split=val_split,
				                                               verbose=0)

				generator_history = self.generator.fit(x=batch,
				                                       y=np.zeros(self.batch_size),
				                                       epochs=1,
				                                       batch_size=self.batch_size,
				                                       validation_split=val_split,
				                                       verbose=0)

			# Update loss functions at the end of each epoch
			self.rec_loss = autoencoder_history.history["loss"][0]
			self.gen_loss = generator_history.history["loss"][0]
			self.dis_loss = discriminator_history.history["loss"][0]

			if ((epoch + 1) % 10 == 0):

				clear_output()

				print(
					"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}".format(
						*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss])
				)

				if graph and (gene is not None):
					self.plot_umap(gene_selected=[gene])

			rec_loss.append(self.rec_loss)
			gen_loss.append(self.gen_loss)
			dis_loss.append(self.dis_loss)

		print("Training completed.")

		return rec_loss, gen_loss, dis_loss


##########################################
############### MODEL n.2 ################
##########################################
class AAE2(Base):
	""" Model incorporating label information in the adversarial regularization.

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	Raises
	------
	TypeError
		If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim, layers_dis_dim.
	"""

	def __init__(self, **kwargs):
		super(AAE2, self).__init__(**kwargs)

		if self.latent_dim is None or \
				self.layers_enc_dim is None or \
				self.layers_dec_dim is None or \
				self.layers_dis_dim is None:
			raise TypeError(
				"List of mandatory arguments: latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer
		# TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_enc_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          # kernel_regularizer=regularizers.l2(self.l2_weight),
			          # activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		z_mean = Dense(self.latent_dim,
		               name='z_mean',
		               kernel_initializer=self.kernel_initializer,
		               bias_initializer=self.bias_initializer)(x)

		z_log_var = Dense(self.latent_dim,
		                  name='z_log_var',
		                  kernel_initializer=self.kernel_initializer,
		                  bias_initializer=self.bias_initializer)(x)

		z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

		# instantiate encoder model
		encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

		return encoder

	def _build_decoder(self):

		"""Build decoder neural network.

		:return:
			decoder
		"""

		# TODO: check impact of kernel and activity regularizer

		decoder_input = Input(shape=(self.latent_dim,), name='Z')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dec_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model(decoder_input, x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network.

		:return:
			discriminator
		"""
		# TODO: check impact of kernel and activity regularizer

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		labels_input = Input(shape=(labels_dim + 1,), name='Labels')  # add one category for non-labeld data
		discr_input = concatenate([latent_input, labels_input], name='Z_Labels')

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dis_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model([latent_input, labels_input], x, name='discriminator')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def _build_generator(self, input_encoder, compression, discriminator):

		"""Build generator neural network.

		:param input_encoder:
			encoder input layer
		:param compression:
			encoder transformation
		:param discriminator:
			initialized discriminator model
		:return:
			generator
		"""

		optimizer_gen = Adam(lr=self.lr_gen, decay=self.dr_gen)

		# keep discriminator weights frozen
		discriminator.trainable = False

		generation = discriminator(compression)

		# instantiate and compile generator model
		generator = Model(input_encoder, generation)
		generator.compile(optimizer=optimizer_gen, loss="binary_crossentropy", metrics=['accuracy'])

		return generator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture.

		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		encoder_input = Input(shape=(self.original_dim,), name='X')
		labels_input = Input(shape=(labels_dim + 1,), name='Labels')

		# build encoder
		self.encoder = self._build_encoder()

		# build decoder
		self.decoder = self._build_decoder()

		# build and compile discriminator
		self.discriminator = self._build_discriminator()

		# build and compile autoencoder
		real_input = encoder_input
		compression = self.encoder(real_input)[2]
		reconstruction = self.decoder(compression)
		self.autoencoder = Model(real_input, reconstruction, name='autoencoder')
		self.autoencoder.compile(optimizer=optimizer_ae, loss='mse')

		# build and compile generator model
		self.generator = self._build_generator([real_input, labels_input],
		                                       [self.encoder(real_input)[2], labels_input],
		                                       self.discriminator)

	def train(self, graph=False, gene=None):

		"""Training of the Adversarial Autoencoder.

		During the reconstruction phase the training of the generator proceeds with the
		discriminator weights frozen.

		:param graph:
			if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
		:type graph: bool
		:param gene:
			selected gene
		:type gene: str
		:return:
			lists containing reconstruction loss, generator loss, and discriminator loss at each epoch
		"""

		rec_loss = []
		gen_loss = []
		dis_loss = []

		val_split = 0.0

		labels_code = to_categorical(self.labels).astype(int)
		labels_zeros = np.zeros((len(self.labels), 1))  # extra label for training points with unknown classes
		labels_code = np.append(labels_code, labels_zeros, axis=1)

		data_ = np.concatenate([self.data, labels_code], axis=1)

		print("Start model training...")

		for epoch in range(self.epochs):
			np.random.shuffle(data_)

			for i in range(int(len(self.data) / self.batch_size)):
				batch = data_[i * self.batch_size:i * self.batch_size + self.batch_size, :self.data.shape[1]]
				labels_ = data_[i * self.batch_size:i * self.batch_size + self.batch_size, self.data.shape[1]:]

				# Reconstruction phase
				autoencoder_history = self.autoencoder.fit(x=batch,
				                                           y=batch,
				                                           epochs=1,
				                                           batch_size=self.batch_size,
				                                           validation_split=val_split,
				                                           verbose=0)

				# Regularization phase
				fake_pred = self.encoder.predict(batch)[2]
				real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
				discriminator_batch_x = [np.concatenate([fake_pred, real_pred]),
				                         np.concatenate([labels_, labels_])]
				discriminator_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
				                                        np.random.uniform(0.0, 0.1, self.batch_size)])

				discriminator_history = self.discriminator.fit(x=discriminator_batch_x,
				                                               y=discriminator_batch_y,
				                                               epochs=1,
				                                               batch_size=self.batch_size,
				                                               validation_split=val_split,
				                                               verbose=0)

				generator_history = self.generator.fit(x=[batch, labels_],
				                                       y=np.zeros(self.batch_size),
				                                       epochs=1,
				                                       batch_size=self.batch_size,
				                                       validation_split=val_split,
				                                       verbose=0)

			# Update loss functions at the end of each epoch
			self.rec_loss = autoencoder_history.history["loss"][0]
			self.gen_loss = generator_history.history["loss"][0]
			self.dis_loss = discriminator_history.history["loss"][0]

			if ((epoch + 1) % 10 == 0):

				clear_output()

				print(
					"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}".format(
						*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss])
				)

				if graph and (gene is not None):
					self.plot_umap(gene_selected=[gene])

			rec_loss.append(self.rec_loss)
			gen_loss.append(self.gen_loss)
			dis_loss.append(self.dis_loss)

		print("Training completed.")

		return rec_loss, gen_loss, dis_loss


##########################################
############### MODEL n.3 ################
##########################################
class AAE3(Base):
	"""  Supervised adversarial autoencoder model.

		 The label information is disentangled from the hidden code by providing
		 a one-hot encoded vector to the generative model.

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	Raises
	------
	TypeError
		If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim, layers_dis_dim.
	"""

	def __init__(self, **kwargs):
		super(AAE3, self).__init__(**kwargs)

		if self.latent_dim is None or \
				self.layers_enc_dim is None or \
				self.layers_dec_dim is None or \
				self.layers_dis_dim is None:
			raise TypeError(
				"List of mandatory arguments: latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer
		# TODO: implement: 1) DETERMINISTIC POSTERIOR Q(z|x); 2) UNIVERSAL APPROXIMATOR POSTERIOR

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_enc_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          # kernel_regularizer=regularizers.l2(self.l2_weight),
			          # activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		z_mean = Dense(self.latent_dim,
		               name='z_mean',
		               kernel_initializer=self.kernel_initializer,
		               bias_initializer=self.bias_initializer)(x)

		z_log_var = Dense(self.latent_dim,
		                  name='z_log_var',
		                  kernel_initializer=self.kernel_initializer,
		                  bias_initializer=self.bias_initializer)(x)

		z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

		# instantiate encoder model
		encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

		return encoder

	def _build_decoder(self):

		"""Build decoder neural network.

		:return:
			decoder
		"""

		# TODO: check impact of kernel and activity regularizer

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		labels_input = Input(shape=(labels_dim,), name='Labels')

		decoder_input = concatenate([latent_input, labels_input], name='Z_Labels')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dec_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model([latent_input, labels_input], x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network.

		:return:
			discriminator
		"""
		# TODO: check impact of kernel and activity regularizer

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		discr_input = latent_input

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dis_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model(latent_input, x, name='discriminator')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def _build_generator(self, input_encoder, compression, discriminator):

		"""Build generator neural network.

		:param input_encoder:
			encoder input layer
		:param compression:
			encoder transformation
		:param discriminator:
			initialized discriminator model
		:return:
			generator
		"""

		optimizer_gen = Adam(lr=self.lr_gen, decay=self.dr_gen)

		# keep discriminator weights frozen
		discriminator.trainable = False

		generation = discriminator(compression)

		# instantiate and compile generator model
		generator = Model(input_encoder, generation)
		generator.compile(optimizer=optimizer_gen, loss="binary_crossentropy", metrics=['accuracy'])

		return generator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture.

		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		encoder_input = Input(shape=(self.original_dim,), name='X')
		labels_input = Input(shape=(labels_dim,), name='Labels')

		# build encoder
		self.encoder = self._build_encoder()

		# build decoder
		self.decoder = self._build_decoder()

		# build and compile discriminator
		self.discriminator = self._build_discriminator()

		# build and compile autoencoder
		real_input = encoder_input
		compression = [self.encoder(real_input)[2], labels_input]
		reconstruction = self.decoder(compression)
		self.autoencoder = Model([real_input, labels_input], reconstruction, name='autoencoder')
		self.autoencoder.compile(optimizer=optimizer_ae, loss='mse')

		# build and compile generator model
		self.generator = self._build_generator(real_input,
		                                       self.encoder(real_input)[2],
		                                       self.discriminator)

	def train(self, graph=False, gene=None):

		"""Training of the Adversarial Autoencoder.

		During the reconstruction phase the training of the generator proceeds with the
		discriminator weights frozen.

		:param graph:
			if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
		:type graph: bool
		:param gene:
			selected gene
		:type gene: str
		:return:
			lists containing reconstruction loss, generator loss, and discriminator loss at each epoch
		"""

		rec_loss = []
		gen_loss = []
		dis_loss = []

		val_split = 0.0

		labels_code = to_categorical(self.labels).astype(int)

		data_ = np.concatenate([self.data, labels_code], axis=1)

		print("Start model training...")

		for epoch in range(self.epochs):
			np.random.shuffle(data_)

			for i in range(int(len(self.data) / self.batch_size)):
				batch = data_[i * self.batch_size:i * self.batch_size + self.batch_size, :self.data.shape[1]]
				labels_ = data_[i * self.batch_size:i * self.batch_size + self.batch_size, self.data.shape[1]:]

				# Reconstruction phase
				autoencoder_history = self.autoencoder.fit(x=[batch, labels_],
				                                           y=batch,
				                                           epochs=1,
				                                           batch_size=self.batch_size,
				                                           validation_split=val_split,
				                                           verbose=0)

				# Regularization phase
				fake_pred = self.encoder.predict(batch)[2]
				real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
				discriminator_batch_x = np.concatenate([fake_pred, real_pred])
				discriminator_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
				                                        np.random.uniform(0.0, 0.1, self.batch_size)])

				discriminator_history = self.discriminator.fit(x=discriminator_batch_x,
				                                               y=discriminator_batch_y,
				                                               epochs=1,
				                                               batch_size=self.batch_size,
				                                               validation_split=val_split,
				                                               verbose=0)

				generator_history = self.generator.fit(x=batch,
				                                       y=np.zeros(self.batch_size),
				                                       epochs=1,
				                                       batch_size=self.batch_size,
				                                       validation_split=val_split,
				                                       verbose=0)

			# Update loss functions at the end of each epoch
			self.rec_loss = autoencoder_history.history["loss"][0]
			self.gen_loss = generator_history.history["loss"][0]
			self.dis_loss = discriminator_history.history["loss"][0]

			if ((epoch + 1) % 10 == 0):

				clear_output()

				print(
					"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}".format(
						*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss])
				)

				if graph and (gene is not None):
					self.plot_umap(gene_selected=[gene])

			rec_loss.append(self.rec_loss)
			gen_loss.append(self.gen_loss)
			dis_loss.append(self.dis_loss)

		print("Training completed.")

		return rec_loss, gen_loss, dis_loss


##########################################
############### MODEL n.4 ################
##########################################
class AAE4(Base):
	""" Semi-supervised adversarial autoencoder model with categorical distribution of classes labels.

		The classes labels match the corresponsding data points.

	Attributes
	----------
	tau: float
		temperature parameter used in the Gumbel-softmax trick
	layers_dis_cat_dim: list
		array containing the dimension of categorical discriminator network dense layers
	lr_dis_cat: float
		learning rate for categorical discriminator optimizer
	lr_gen_cat: float
		learning rate for categorical generator optimizer
	dr_dis_cat: float
		decay rate categorical discriminator optimizer
	dr_gen_cat: float
		decay rate categorical generator optimizer
	discriminator_cat: keras.engine.training.Model
		categorical discriminator deep neural network
	generator_cat: keras.engine.training.Model
		categorical generator deep neural network
	dis_cat_loss: float
		discriminator_cat loss

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	Raises
	------
	TypeError
		If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim, layers_dis_dim, layers_dis_cat_dim.
	"""

	def __init__(self,
	             tau=0.5,
	             layers_dis_cat_dim=None,
	             lr_dis_cat=0.0001,
	             dr_dis_cat=1e-6,
	             **kwargs):

		self.tau = tau
		self.layers_dis_cat_dim = layers_dis_cat_dim
		self.lr_dis_cat = lr_dis_cat
		self.dr_dis_cat = dr_dis_cat

		Base.__init__(self, **kwargs)

		if self.latent_dim is None or \
				self.layers_enc_dim is None or \
				self.layers_dec_dim is None or \
				self.layers_dis_dim is None or \
				self.layers_dis_cat_dim is None:
			raise TypeError(
				"List of mandatory arguments: latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

		self.discriminator_cat = None
		self.generator_cat = None
		self.dis_cat_loss = None

		# update dictionary of internal parameters
		self._update_dict()

	def _update_dict(self):
		"""Update model dictionary of input parameters.

		"""

		dict2 = {"Parameter": [], "Value": [], "Description": []}

		dict2["Parameter"] = np.hstack(
			[['layer_' + str(k + 1) + '_dis_cat_dim' for k in range(len(self.layers_dis_cat_dim))],
			 'lr_dis_cat',
			 'dr_dis_cat',
			 'tau'
			 ])

		dict2["Value"] = np.hstack([self.layers_dis_cat_dim,
		                            self.lr_dis_cat,
		                            self.dr_dis_cat,
		                            self.tau
		                            ])

		dict2["Description"] = np.hstack([["dimension of cat. discriminator dense layer " + str(k + 1) for k in
		                                   range(len(self.layers_dis_cat_dim))],
		                                  "learning rate of cat. discriminator",
		                                  "decay rate of cat. discriminator",
		                                  "temperature parameter"
		                                  ])

		for k in self.dict.keys():
			self.dict[k] = np.append(self.dict[k], dict2[k])

	def get_summary(self):

		"""Print Adversarial Autoencoder model summary.

		:return:
		"""

		print("\nEncoder Network")
		print("===============")
		self.encoder.summary()

		print("\nDecoder Network")
		print("===============")
		self.decoder.summary()

		print("\nAutoencoder Network")
		print("===================")
		self.autoencoder.summary()

		print("\nGenerator Network")
		print("=================")
		# Freeze the discriminator weights during training of generator
		self.generator.summary()

		print("\nDiscriminator Network")
		print("=====================")
		self.discriminator.trainable = True  # note: model already compiled with trainable weights
		self.discriminator.summary()

		print("\nCategorical Discriminator Network")
		print("=================================")
		self.discriminator_cat.trainable = True  # note: model already compiled with trainable weights
		self.discriminator_cat.summary()

	def export_graph(self, filepath):

		"""Save model graphs to PNG images.

		:param filepath:
			 path of output image files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		plot_model(self.encoder, to_file=filepath + "encoder.png", show_shapes=True)
		plot_model(self.decoder, to_file=filepath + "decoder.png", show_shapes=True)
		plot_model(self.autoencoder, to_file=filepath + "autoencoder.png", show_shapes=True)
		plot_model(self.generator, to_file=filepath + "generator.png", show_shapes=True)
		plot_model(self.discriminator, to_file=filepath + "discriminator.png", show_shapes=True)
		plot_model(self.discriminator_cat, to_file=filepath + "discriminator_cat.png", show_shapes=True)

		print("Model graphs saved.\n")

	def export_model(self, filepath):

		"""Export the network models in h5 format.

		:param filepath:
			path of model files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		self.autoencoder.save(filepath + 'autoencoder.h5')
		self.generator.save(filepath + 'generator.h5')
		self.discriminator.save(filepath + 'discriminator.h5')
		self.discriminator_cat.save(filepath + 'discriminator_cat.h5')
		self.encoder.save(filepath + 'encoder.h5')
		self.decoder.save(filepath + 'decoder.h5')

		print("All networks exported in h5 format.")

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_enc_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer)(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		z_mean = Dense(self.latent_dim,
		               name='z_mean',
		               kernel_initializer=self.kernel_initializer,
		               bias_initializer=self.bias_initializer)(x)

		z_log_var = Dense(self.latent_dim,
		                  name='z_log_var',
		                  kernel_initializer=self.kernel_initializer,
		                  bias_initializer=self.bias_initializer)(x)

		z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		y = Dense(labels_dim,
		          name='logits',
		          kernel_initializer=self.kernel_initializer,
		          bias_initializer=self.bias_initializer)(x)

		# y = Softmax(axis=-1, name='y')(y)

		y = Lambda(sampling_gumbel, arguments={'tau': self.tau}, output_shape=(labels_dim,), name='y')(y)

		# instantiate encoder model
		encoder = Model(encoder_input, [z_mean, z_log_var, z, y], name='encoder')

		return encoder

	def _build_decoder(self):

		"""Build decoder neural network.

		:return:
			decoder
		"""

		# TODO: check impact of kernel and activity regularizer

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		labels_input = Input(shape=(labels_dim,), name='y')

		decoder_input = concatenate([latent_input, labels_input], name='Z_y')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dec_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model([latent_input, labels_input], x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network.

		:return:
			discriminator
		"""
		# TODO: check impact of kernel and activity regularizer

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		discr_input = latent_input

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dis_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model(latent_input, x, name='discriminator')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def _build_generator(self, input_encoder, compression, compression_cat, discriminator, discriminator_cat):

		"""Build generator neural network.

		:param input_encoder:
			encoder input layer
		:param compression:
			encoder transformation
		:param discriminator:
			initialized discriminator model
		:return:
			generator
		"""

		optimizer_gen = Adam(lr=self.lr_gen, decay=self.dr_gen)

		# keep discriminator weights frozen
		discriminator.trainable = False
		discriminator_cat.trainable = False

		generation = discriminator(compression)
		generation_cat = discriminator_cat(compression_cat)

		# instantiate and compile generator model
		generator = Model(input_encoder, [generation, generation_cat])
		generator.compile(optimizer=optimizer_gen,
		                  loss=["binary_crossentropy", "binary_crossentropy"],
		                  metrics=['accuracy', 'accuracy'])

		return generator

	def _build_discriminator_cat(self):

		"""Build categorical discriminator neural network.

		:return:
			discriminator_cat

		"""
		# TODO: check impact of kernel and activity regularizer

		optimizer_dis = Adam(lr=self.lr_dis_cat, decay=self.dr_dis_cat)

		labels_dim = np.max(np.unique(self.labels)) + 1  # labels start from 0

		latent_input = Input(shape=(labels_dim,), name='y')
		discr_input = latent_input

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dis_cat_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          use_bias=False,
			          kernel_initializer=self.kernel_initializer,
			          kernel_regularizer=regularizers.l2(self.l2_weight),
			          activity_regularizer=regularizers.l1(self.l1_weight))(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model(latent_input, x, name='discriminator_cat')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture.

		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		encoder_input = Input(shape=(self.original_dim,), name='X')

		# build encoder
		self.encoder = self._build_encoder()

		# build decoder
		self.decoder = self._build_decoder()

		# build and compile discriminator
		self.discriminator = self._build_discriminator()

		# build and compile categorical discriminator
		self.discriminator_cat = self._build_discriminator_cat()

		# build and compile autoencoder
		real_input = encoder_input
		compression = [self.encoder(real_input)[2], self.encoder(real_input)[3]]
		reconstruction = self.decoder(compression)
		self.autoencoder = Model(real_input, reconstruction, name='autoencoder')
		self.autoencoder.compile(optimizer=optimizer_ae, loss='mse')

		# build and compile generator model
		self.generator = self._build_generator(real_input,
		                                       self.encoder(real_input)[2],
		                                       self.encoder(real_input)[3],
		                                       self.discriminator,
		                                       self.discriminator_cat)

	def train(self, graph=False, gene=None):

		"""Training of the semisupervised adversarial autoencoder.

		During the reconstruction phase the training of the generator proceeds with the
		discriminator weights frozen.

		:param graph:
			if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
		:type graph: bool
		:param gene:
			selected gene
		:type gene: str
		:return:
			lists containing reconstruction loss, generator loss, discriminator loss,
			categorical generator loss, and categorical discriminator loss at each epoch
		"""

		rec_loss = []
		gen_loss = []
		dis_loss = []
		dis_cat_loss = []

		val_split = 0.0

		labels_code = to_categorical(self.labels).astype(int)

		data_ = np.concatenate([self.data, labels_code], axis=1)

		print("Start model training...")

		for epoch in range(self.epochs):
			np.random.shuffle(data_)

			for i in range(int(len(self.data) / self.batch_size)):
				batch = data_[i * self.batch_size:i * self.batch_size + self.batch_size, :self.data.shape[1]]
				labels_ = data_[i * self.batch_size:i * self.batch_size + self.batch_size, self.data.shape[1]:]

				# Reconstruction phase
				autoencoder_history = self.autoencoder.fit(x=batch,
				                                           y=batch,
				                                           epochs=1,
				                                           batch_size=self.batch_size,
				                                           validation_split=val_split,
				                                           verbose=0)

				# Regularization phase
				fake_pred = self.encoder.predict(batch)[2]
				real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
				discriminator_batch_x = np.concatenate([fake_pred, real_pred])
				discriminator_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
				                                        np.random.uniform(0.0, 0.1, self.batch_size)])

				discriminator_history = self.discriminator.fit(x=discriminator_batch_x,
				                                               y=discriminator_batch_y,
				                                               epochs=1,
				                                               batch_size=self.batch_size,
				                                               validation_split=val_split,
				                                               verbose=0)

				fake_pred_cat = self.encoder.predict(batch)[3]
				real_pred_cat = labels_

				discriminator_cat_batch_x = np.concatenate([fake_pred_cat, real_pred_cat])
				discriminator_cat_batch_y = np.concatenate([np.random.uniform(0.9, 1.0, self.batch_size),
				                                            np.random.uniform(0.0, 0.1, self.batch_size)])

				discriminator_cat_history = self.discriminator_cat.fit(x=discriminator_cat_batch_x,
				                                                       y=discriminator_cat_batch_y,
				                                                       epochs=1,
				                                                       batch_size=self.batch_size,
				                                                       validation_split=val_split,
				                                                       verbose=0)

				generator_history = self.generator.fit(x=batch,
				                                       y=[np.zeros(self.batch_size), np.zeros(self.batch_size)],
				                                       epochs=1,
				                                       batch_size=self.batch_size,
				                                       validation_split=val_split,
				                                       verbose=0)

			# Update loss functions at the end of each epoch
			self.rec_loss = autoencoder_history.history["loss"][0]
			self.gen_loss = generator_history.history["loss"][0]
			self.dis_loss = discriminator_history.history["loss"][0]
			self.dis_cat_loss = discriminator_cat_history.history["loss"][0]

			if ((epoch + 1) % 10 == 0):

				clear_output()

				print(
					"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, generation loss: {3:.6f}, discriminator loss: {4:.6f}, cat. discriminator loss: {5:.6f}"
						.format(
						*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss, self.dis_cat_loss])
				)

				if graph and (gene is not None):
					self.plot_umap(gene_selected=[gene])

			rec_loss.append(self.rec_loss)
			gen_loss.append(self.gen_loss)
			dis_loss.append(self.dis_loss)
			dis_cat_loss.append(self.dis_cat_loss)

		print("Training completed.")

		return rec_loss, gen_loss, dis_loss, dis_cat_loss


##########################################
############### MODEL n.5 ################
##########################################
class AAE5(Base):
	""" Unsupervised adversarial autoencoder model with arebitrary number of clusters.

	Attributes
	----------
	clusters: int
		number of clusters in the dataset
	tau: float
		temperature parameter used in the Gumbel-softmax trick
	layers_dis_cat_dim: list
		array containing the dimension of categorical discriminator network dense layers
	lr_dis_cat: float
		learning rate for categorical discriminator optimizer
	lr_gen_cat: float
		learning rate for categorical generator optimizer
	dr_dis_cat: float
		decay rate categorical discriminator optimizer
	dr_gen_cat: float
		decay rate categorical generator optimizer
	discriminator_cat: keras.engine.training.Model
		categorical discriminator deep neural network
	generator_cat: keras.engine.training.Model
		categorical generator deep neural network
	dis_cat_loss: float
		discriminator_cat loss

	Methods
	-------
	get_parameters()
		Print the list of network parameters
	get_data(datapath)
		Read data file and initialize cell gene counts, gene name list and cell subgroups
	rescale_data()
		Rescale gene expression data to zero mean and unit variance
	get_summary(model)
		print model summary
	export_graph(model, filename)
		save model graph to file
	plot_umap(gene_selected, louvain=False)
		plot the gene expression in the 2-D latent space using UMAP clustering algorithm
	export_model(filepath)
		export the network models in h5 format

	Raises
	------
	TypeError
		If one of the following argument is null:  latent_dim, layers_enc_dim, layers_dec_dim, layers_dis_dim, layers_dis_cat_dim.
	"""

	def __init__(self,
	             num_clusters=None,
	             tau=0.5,
	             layers_dis_cat_dim=None,
	             lr_dis_cat=0.0001,
	             dr_dis_cat=1e-6,
	             **kwargs):

		self.num_clusters = num_clusters
		self.tau = tau
		self.layers_dis_cat_dim = layers_dis_cat_dim
		self.lr_dis_cat = lr_dis_cat
		self.dr_dis_cat = dr_dis_cat

		Base.__init__(self, **kwargs)

		if self.latent_dim is None or \
				self.layers_enc_dim is None or \
				self.layers_dec_dim is None or \
				self.layers_dis_dim is None or \
				self.layers_dis_cat_dim is None or \
				self.num_clusters is None:
			raise TypeError(
				"List of mandatory arguments: num_clusters, latent_dim, layers_enc_dim, layers_dec_dim, and layers_dis_dim.")

		self.discriminator_cat = None
		self.generator_cat = None
		self.dis_cat_loss = None

		# update dictionary of internal parameters
		self._update_dict()

	def _update_dict(self):
		"""Update model dictionary of input parameters.

		"""

		dict2 = {"Parameter": [], "Value": [], "Description": []}

		dict2["Parameter"] = np.hstack(
			[['layer_' + str(k + 1) + '_dis_cat_dim' for k in range(len(self.layers_dis_cat_dim))],
			 'lr_dis_cat',
			 'dr_dis_cat',
			 'tau',
			 'num_clusters'
			 ])

		dict2["Value"] = np.hstack([self.layers_dis_cat_dim,
		                            self.lr_dis_cat,
		                            self.dr_dis_cat,
		                            self.tau,
		                            self.num_clusters
		                            ])

		dict2["Description"] = np.hstack([["dimension of cat. discriminator dense layer " + str(k + 1) for k in
		                                   range(len(self.layers_dis_cat_dim))],
		                                  "learning rate of cat. discriminator",
		                                  "decay rate of cat. discriminator",
		                                  "temperature parameter",
		                                  "number of clusters in the dateset"
		                                  ])

		for k in self.dict.keys():
			self.dict[k] = np.append(self.dict[k], dict2[k])

	def get_summary(self):

		"""Print Adversarial Autoencoder model summary.

		:return:
		"""

		print("\nEncoder Network")
		print("===============")
		self.encoder.summary()

		print("\nDecoder Network")
		print("===============")
		self.decoder.summary()

		print("\nAutoencoder Network")
		print("===================")
		self.autoencoder.summary()

		print("\nDiscriminator Network")
		print("=====================")
		self.discriminator.trainable = True  # note: model already compiled with trainable weights
		self.discriminator.summary()

		print("\nCategorical Discriminator Network")
		print("=================================")
		self.discriminator_cat.trainable = True  # note: model already compiled with trainable weights
		self.discriminator_cat.summary()

	def export_graph(self, filepath):

		"""Save model graphs to PNG images.

		:param filepath:
			 path of output image files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		plot_model(self.encoder, to_file=filepath + "encoder.png", show_shapes=True)
		plot_model(self.decoder, to_file=filepath + "decoder.png", show_shapes=True)
		plot_model(self.autoencoder, to_file=filepath + "autoencoder.png", show_shapes=True)
		plot_model(self.discriminator, to_file=filepath + "discriminator.png", show_shapes=True)
		plot_model(self.discriminator_cat, to_file=filepath + "discriminator_cat.png", show_shapes=True)

		print("Model graphs saved.\n")

	def export_model(self, filepath):

		"""Export the network models in h5 format.

		:param filepath:
			path of model files
		:type filepath: str
		:return:
		"""

		if filepath[-1] != "/":
			filepath = filepath + "/"

		self.autoencoder.save(filepath + 'autoencoder.h5')
		self.discriminator.save(filepath + 'discriminator.h5')
		self.discriminator_cat.save(filepath + 'discriminator_cat.h5')
		self.encoder.save(filepath + 'encoder.h5')
		self.decoder.save(filepath + 'decoder.h5')

		print("All networks exported in h5 format.")

	def _build_encoder(self):

		"""Build encoder neural network.

		:return:
			encoder
		"""
		# NB: no kernel regularizer and activity regularizer

		# GAUSSIAN POSTERIOR

		encoder_input = Input(shape=(self.original_dim,), name="X")

		x = Dropout(rate=self.do_rate, name='D_O')(encoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_enc_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          # use_bias=False,
			          # kernel_initializer=self.kernel_initializer
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		z0 = Dense(self.layers_enc_dim[-1],
		           name="H_z",
		           # use_bias=False,
		           # kernel_initializer=self.kernel_initializer
		           )(x)

		z0 = BatchNormalization(name='BN_z')(z0)

		z0 = LeakyReLU(alpha=self.alpha, name='LR_z')(z0)

		z0 = Dropout(rate=self.do_rate, name='D_z')(z0)

		z_mean = Dense(self.latent_dim,
		               name='z_mean',
		               # kernel_initializer=self.kernel_initializer,
		               # bias_initializer=self.bias_initializer
		               )(z0)

		z_log_var = Dense(self.latent_dim,
		                  name='z_log_var',
		                  # kernel_initializer=self.kernel_initializer,
		                  # bias_initializer=self.bias_initializer
		                  )(z0)

		z = Lambda(sampling, output_shape=(self.latent_dim,), name='Z')([z_mean, z_log_var])

		y0 = Dense(self.layers_enc_dim[-1],
		           name="H_y",
		           use_bias=False,
		           kernel_initializer=self.kernel_initializer)(x)

		y0 = BatchNormalization(name='BN_y')(y0)

		y0 = LeakyReLU(alpha=self.alpha, name='LR_y')(y0)

		y0 = Dropout(rate=self.do_rate, name='D_y')(y0)

		y = Dense(self.num_clusters,
		          name='logits',
		          # kernel_initializer=self.kernel_initializer,
		          # bias_initializer=self.bias_initializer
		          )(y0)

		y = Lambda(sampling_gumbel, arguments={'tau': self.tau}, output_shape=(self.num_clusters,), name='y')(y)

		# instantiate encoder model
		encoder = Model(encoder_input, [z_mean, z_log_var, z, y], name='encoder')

		return encoder

	def _build_decoder(self):

		"""Build decoder neural network.

		:return:
			decoder
		"""

		latent_input = Input(shape=(self.latent_dim,), name='Z')
		classes = Input(shape=(self.num_clusters,), name='y')

		decoder_input = concatenate([latent_input, classes], name='Z_y')

		x = Dropout(rate=self.do_rate, name='D_O')(decoder_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dec_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          # use_bias=False,
			          # kernel_initializer=self.kernel_initializer,
			          # kernel_regularizer=regularizers.l2(self.l2_weight),
			          # activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(self.original_dim, activation='sigmoid', name="Xp")(x)

		# instantiate decoder model
		decoder = Model([latent_input, classes], x, name='decoder')

		return decoder

	def _build_discriminator(self):

		"""Build discriminator neural network.

		:return:
			discriminator
		"""

		optimizer_dis = Adam(lr=self.lr_dis, decay=self.dr_dis)

		discr_input = Input(shape=(self.latent_dim,), name='Z')

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dis_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          # use_bias=False,
			          # kernel_initializer=self.kernel_initializer,
			          # kernel_regularizer=regularizers.l2(self.l2_weight),
			          # activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model(discr_input, x, name='discriminator')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def _build_generator(self, compression, compression_cat, discriminator, discriminator_cat):

		"""Build generator neural network.

		:param compression:
			encoder transformation
		:param discriminator:
			initialized discriminator model
		:return:
			generator
		"""

		# keep discriminator weights frozen
		discriminator.trainable = False
		discriminator_cat.trainable = False

		generation = discriminator(compression)
		generation_cat = discriminator_cat(compression_cat)

		return generation, generation_cat

	def _build_discriminator_cat(self):

		"""Build categorical discriminator neural network.

		:return:
			discriminator_cat

		"""
		optimizer_dis = Adam(lr=self.lr_dis_cat, decay=self.dr_dis_cat)

		discr_input = Input(shape=(self.num_clusters,), name='y')

		x = Dropout(rate=self.do_rate, name='D_O')(discr_input)

		# add dense layers
		for i, nodes in enumerate(self.layers_dis_cat_dim):
			x = Dense(nodes,
			          name="H_" + str(i + 1),
			          # use_bias=False,
			          # kernel_initializer=self.kernel_initializer,
			          # kernel_regularizer=regularizers.l2(self.l2_weight),
			          # activity_regularizer=regularizers.l1(self.l1_weight)
			          )(x)

			x = BatchNormalization(name='BN_' + str(i + 1))(x)

			x = LeakyReLU(alpha=self.alpha, name='LR_' + str(i + 1))(x)

			x = Dropout(rate=self.do_rate, name='D_' + str(i + 1))(x)

		x = Dense(1, activation='sigmoid', name="Check")(x)

		# instantiate and compile discriminator model
		discriminator = Model(discr_input, x, name='discriminator_cat')
		discriminator.compile(optimizer=optimizer_dis, loss="binary_crossentropy", metrics=['accuracy'])

		return discriminator

	def build_model(self):

		"""Build Adversarial Autoencoder model architecture.

		"""

		optimizer_ae = Adam(lr=self.lr_ae, decay=self.dr_ae)

		encoder_input = Input(shape=(self.original_dim,), name='X')

		# build encoder
		self.encoder = self._build_encoder()

		# build decoder
		self.decoder = self._build_decoder()

		# build and compile discriminator
		self.discriminator = self._build_discriminator()

		# build and compile categorical discriminator
		self.discriminator_cat = self._build_discriminator_cat()

		# build generators
		compression = [self.encoder(encoder_input)[2], self.encoder(encoder_input)[3]]
		generation, generation_cat = self._build_generator(compression[0],
		                                                   compression[1],
		                                                   self.discriminator,
		                                                   self.discriminator_cat)

		# build and compile autoencoder
		reconstruction = self.decoder(compression)
		self.autoencoder = Model(encoder_input,
		                         [reconstruction, generation, generation_cat],
		                         name='autoencoder')
		self.autoencoder.compile(optimizer=optimizer_ae,
		                         loss=['mse', 'binary_crossentropy', 'binary_crossentropy'],
		                         loss_weights=[0.99, 0.005, 0.005], )

	def train(self, graph=False, gene=None, val_split=0.2):

		"""Training of the semisupervised adversarial autoencoder.

		During the reconstruction phase the training of the generator proceeds with the
		discriminator weights frozen.

		:param graph:
			if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
		:type graph: bool
		:param gene:
			selected gene
		:type gene: str
		:param val_split:
			fraction of the training data to be used as validation data
		:type val_split: float
		:return:
			lists containing reconstruction training loss, reconstruction validation loss,
			discriminator loss, and categorical discriminator loss at each epoch
		"""

		rec_loss = []
		val_loss = [] # validation loss
		dis_loss = []
		dis_cat_loss = []

		print("Start model training...")

		for epoch in range(self.epochs):
			np.random.shuffle(self.data)

			for i in range(int(len(self.data) / self.batch_size)):
				batch = self.data[i * self.batch_size:i * self.batch_size + self.batch_size]

				# Regularization phase
				real = np.random.uniform(0.0, 0.1, self.batch_size)
				fake = np.random.uniform(0.9, 1.0, self.batch_size)

				fake_pred = self.encoder.predict(batch)[2]
				real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
				discriminator_batch_x = np.concatenate([fake_pred, real_pred])
				discriminator_batch_y = np.concatenate([fake, real])

				self.discriminator.trainable = True
				discriminator_history = self.discriminator.fit(x=discriminator_batch_x,
				                                               y=discriminator_batch_y,
				                                               epochs=1,
				                                               batch_size=self.batch_size,
				                                               validation_split=0.0,
				                                               verbose=0)
				self.discriminator.trainable = False
				#print(self.discriminator.layers[1].get_weights())

				fake_pred_cat = self.encoder.predict(batch)[3]
				class_sample = np.random.randint(low=0, high=self.num_clusters, size=self.batch_size)
				real_pred_cat = to_categorical(class_sample, num_classes=self.num_clusters).astype(int)

				discriminator_cat_batch_x = np.concatenate([fake_pred_cat, real_pred_cat])
				discriminator_cat_batch_y = np.concatenate([fake, real])

				self.discriminator_cat.trainable = True
				discriminator_cat_history = self.discriminator_cat.fit(x=discriminator_cat_batch_x,
				                                                       y=discriminator_cat_batch_y,
				                                                       epochs=1,
				                                                       batch_size=self.batch_size,
				                                                       validation_split=0.0,
				                                                       verbose=0)
				self.discriminator_cat.trainable = False

				# Reconstruction phase
				real = np.zeros((self.batch_size,), dtype=int)
				autoencoder_history = self.autoencoder.fit(x=batch,
				                                           y=[batch, real, real],
				                                           epochs=1,
				                                           batch_size=self.batch_size,
				                                           validation_split=val_split,
				                                           verbose=0)

				#print(self.discriminator.layers[1].get_weights())

			# Update loss functions at the end of each epoch
			self.rec_loss = autoencoder_history.history["loss"][0]
			self.gen_loss = autoencoder_history.history["val_loss"][0]
			self.dis_loss = discriminator_history.history["loss"][0]
			self.dis_cat_loss = discriminator_cat_history.history["loss"][0]

			if ((epoch + 1) % 1 == 0):

				clear_output()

				print(
					"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, validation loss: {3:.6f}, discriminator loss: {4:.6f}, cat. discriminator loss: {5:.6f}"
						.format(*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss, self.dis_cat_loss])
				)

				if graph and (gene is not None):
					self.plot_umap(gene_selected=[gene], louvain=True)

			rec_loss.append(self.rec_loss)
			val_loss.append(self.gen_loss)
			dis_loss.append(self.dis_loss)
			dis_cat_loss.append(self.dis_cat_loss)

		print("Training completed.")

		return rec_loss, val_loss, dis_loss, dis_cat_loss

	# def train(self, graph=False, gene=None, val_split=0.2):
	#
	# 	"""Training of the semisupervised adversarial autoencoder.
	#
	# 	During the reconstruction phase the training of the generator proceeds with the
	# 	discriminator weights frozen.
	#
	# 	:param graph:
	# 		if true, then shows every 10 epochs 2-D cluster plot with selected gene expression
	# 	:type graph: bool
	# 	:param gene:
	# 		selected gene
	# 	:type gene: str
	# 	:param val_split:
	# 		fraction of the training data to be used as validation data
	# 	:type val_split: float
	# 	:return:
	# 		lists containing reconstruction training loss, reconstruction validation loss,
	# 		discriminator loss, and categorical discriminator loss at each epoch
	# 	"""
	#
	# 	rec_loss = []
	# 	val_loss = []  # validation loss
	# 	dis_loss = []
	# 	dis_cat_loss = []
	#
	# 	print("Start model training...")
	#
	# 	for epoch in range(self.epochs):
	# 		np.random.shuffle(self.data)
	#
	# 		for i in range(int(len(self.data) / self.batch_size)):
	# 			batch = self.data[i * self.batch_size:i * self.batch_size + self.batch_size]
	#
	# 			# Regularization phase
	# 			real = np.random.uniform(0.0, 0.1, self.batch_size)
	# 			fake = np.random.uniform(0.9, 1.0, self.batch_size)
	#
	# 			fake_pred = self.encoder.predict(batch)[2]
	# 			real_pred = np.random.normal(size=(self.batch_size, self.latent_dim))  # prior distribution
	# 			discriminator_batch_x = np.concatenate([fake_pred, real_pred])
	# 			discriminator_batch_y = np.concatenate([fake, real])
	#
	# 			discriminator_loss = self.discriminator.train_on_batch(x=discriminator_batch_x,
	# 			                                                       y=discriminator_batch_y)
	#
	# 			print(self.discriminator.layers[1].get_weights())
	#
	# 			fake_pred_cat = self.encoder.predict(batch)[3]
	# 			class_sample = np.random.randint(low=0, high=self.num_clusters, size=self.batch_size)
	# 			real_pred_cat = to_categorical(class_sample, num_classes=self.num_clusters).astype(int)
	#
	# 			discriminator_cat_batch_x = np.concatenate([fake_pred_cat, real_pred_cat])
	# 			discriminator_cat_batch_y = np.concatenate([fake, real])
	#
	# 			discriminator_cat_loss = self.discriminator_cat.train_on_batch(x=discriminator_cat_batch_x,
	# 			                                                               y=discriminator_cat_batch_y)
	#
	# 			# Reconstruction phase
	# 			real = np.zeros((self.batch_size,), dtype=int)
	# 			autoencoder_loss = self.autoencoder.train_on_batch(x=batch,
	# 			                                                   y=[batch, real, real])
	#
	# 			autoencoder_val_loss = self.autoencoder.test_on_batch(x=batch,
	# 			                                                      y=[batch, real, real])
	#
	# 			print(self.discriminator.layers[1].get_weights())
	# 			print(self.autoencoder.layers[1].get_weights())
	#
	# 		# Update loss functions at the end of each epoch
	# 		self.rec_loss = autoencoder_loss[1]
	# 		self.gen_loss = autoencoder_val_loss[1]
	# 		self.dis_loss = discriminator_loss[1]
	# 		self.dis_cat_loss = discriminator_cat_loss[1]
	#
	# 		if ((epoch + 1) % 1 == 0):
	#
	# 			clear_output()
	#
	# 			print(
	# 				"Epoch {0:d}/{1:d}, reconstruction loss: {2:.6f}, validation loss: {3:.6f}, discriminator loss: {4:.6f}, cat. discriminator loss: {5:.6f}"
	# 					.format(
	# 					*[epoch + 1, self.epochs, self.rec_loss, self.gen_loss, self.dis_loss, self.dis_cat_loss])
	# 			)
	#
	# 			if graph and (gene is not None):
	# 				self.plot_umap(gene_selected=[gene], louvain=True)
	#
	# 		rec_loss.append(self.rec_loss)
	# 		val_loss.append(self.gen_loss)
	# 		dis_loss.append(self.dis_loss)
	# 		dis_cat_loss.append(self.dis_cat_loss)
	#
	# 	print("Training completed.")
	#
	# 	return rec_loss, val_loss, dis_loss, dis_cat_loss
