# from keras.utils.vis_utils import model_to_dot
#
import tensorflow as tf
# from tensorflow.keras.utils import plot_model
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
import os, sys
import argparse
import json
import textwrap
from IPython.display import clear_output

sys.path.append('../src')

# from imp import load_source
from models import *
# from utils import *

# tf.enable_eager_execution()
# tf.executing_eagerly()

import warnings

warnings.filterwarnings("ignore")


# from tensorflow.python.util import deprecation

# deprecation._PRINT_DEPRECATION_WARNINGS = False


def header():
	version = '1.0.0'
	program = 'Single Cell RNA Sequencing Analysis'
	print(" ")
	header = """
    ============================================
    {} (v{}) 
    ============================================
    """.format(program, version)

	list = textwrap.wrap(header, width=45)
	for element in list:
		print(element)
	print("\n")


def init_model(param_file, model_type):
	# Model selection
	if model_type not in ['VAE', 'AAE1', 'AAE2']:
		print("model type must be one of 'VAE', 'AAE1', and 'AAE2'.")
		sys.exit()

	# Initialize network parameters
	parameters = init_parameters(param_file, model_type)

	if model_type == 'VAE':
		model = VAE(**parameters)
		print(" Model: Variational Autoencoder ")
		print(" ==============================\n")

	elif model_type == 'AAE1':
		model = AAE1(**parameters)
		print(" Model: Adversarial Autoencoder 1 ")
		print(" ================================\n")

	elif model_type == 'AAE2':
		model = AAE2(**parameters)
		print(" Model: Adversarial Autoencoder 2 ")
		print(" ================================\n")

	print(" list of network parameters")
	print(" ^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
	print(model.get_parameters())
	print("\n")

	return model


def init_parameters(param_file, model_type):
	if param_file is None:

		if model_type == 'VAE':

			parameters = {
				'latent_dim': 100,
				'layers_enc_dim': [1000, 500, 400, 300, 200],
				'layers_dec_dim': [200, 300, 400, 500, 1000],
				'batch_size': 1000,
				'epochs': 50
			}

		elif model_type == 'AAE1':

			parameters = {
				'latent_dim': 100,
				'layers_enc_dim': [1000, 500, 400, 300, 200],
				'layers_dec_dim': [200, 300, 400, 500, 1000],
				'layers_dis_dim': [1000, 500, 400, 300, 200],
				'batch_size': 1000,
				'epochs': 10
			}

		elif model_type == 'AAE2':

			parameters = {
				'latent_dim': 100,
				'num_clusters': 17,
				'layers_enc_dim': [1000, 500, 400, 300, 200],
				'layers_dec_dim': [200, 300, 400, 500, 1000],
				'layers_dis_dim': [1000, 500, 400, 300, 200],
				'layers_dis_cat_dim': [1000, 500, 400, 300, 200],
				'batch_size': 1000,
				'epochs': 10,
				'tau': 0.05  # temperature parameter
			}

	else:

		file_settings = open(param_file, "r")
		with file_settings as f:
			parameters = json.load(f)

	return parameters


parser = argparse.ArgumentParser(
	prog='train.py',
	usage='python train.py --gps-path GPS_PATH --acc-path ACC_PATH [GPS options] [accelerometer options] [Spark options]',
	description="%(prog)s is an implementation of the Personal Activity and Location Measurement System (PALMS), written\
                 in Python and integrated with Apache Spark for cluster-computing.",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

requiredargs = parser.add_argument_group('required arguments')
optionalargs = parser.add_argument_group('options')

requiredargs.add_argument(
	"-f",
	"--data-file",
	type=str,
	dest="data_file",
	default=None,
	help="path to processed single cell RNA data file in h5da format."
)

requiredargs.add_argument(
	"-m",
	"--model",
	type=str,
	dest="model_type",
	default="VAE",
	help="select model type: 'VAE', 'AAE1', 'AAE2'."
)

optionalargs.add_argument(
	"-p",
	"--param",
	type=str,
	dest="param_file",
	default=None,
	help="JSON file with input parameters."
)

optionalargs.add_argument(
	"-ap",
	"--add-param",
	type=str,
	dest="add_param_file",
	default=None,
	help="JSON file with additional model training parameters."
)

optionalargs.add_argument(
	"-tf",
	"--tfrecord-data-file",
	type=str,
	dest="tfrecord_data_file",
	default=None,
	help="path to TFRecord dataset file."
)

optionalargs.add_argument(
	"-ds",
	"--distribution-strategy",
	type=str,
	dest="strategy_type",
	default='MirroredStrategy',
	help="TensorFlow distribution strategy: 'MirroredStrategy', 'MultiWorkerMirroredStrategy'."
)

optionalargs.add_argument(
	"-o",
	"--output",
	type=str,
	dest="output_folder",
	default='output',
	help="name of the output folder."
)


def main(data_file, model_type, strategy_type, param_file, add_param_file, tfrecord_data_file, output_folder):
	if data_file is None:
		print("Specify path to h5da data file (-f or --data-file).\n")
		sys.exit()

	header()
	# print(data_file)
	# print(model_type)
	# print(param_file)
	# print(tfrecord_data_file)
	# print(strategy_type)

	if output_folder[-1] != "/":
		output_folder = output_folder + "/"

	# Initialize model
	model = init_model(param_file, model_type)

	print(" load and transform the data ")
	print(" ^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
	# Load data
	model.load_data(data_file)

	# Rescale data
	model.rescale_data()

	print("\n")

	# Build and compile the model
	print(" build and compile model ")
	print(" ^^^^^^^^^^^^^^^^^^^^^^^\n")

	# Set distribution strategy
	if strategy_type == 'MirroredStrategy':
		CROSS_DEVICE_OPS = tf.distribute.NcclAllReduce()
		# CROSS_DEVICE_OPS = tf.distribute.ReductionToOneDevice()
		# CROSS_DEVICE_OPS = tf.distribute.HierarchicalCopyAllReduce()
		strategy = tf.distribute.MirroredStrategy(
			cross_device_ops=CROSS_DEVICE_OPS)

		with strategy.scope():
			if model_type == 'VAE':
				BATCH_SIZE_PER_REPLICA = model.batch_size
				global_batch_size = (BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync)
				model.batch_size = global_batch_size
				model.build_model()

			else:
				model.build_model()

			model.get_summary()

	elif strategy_type == 'MultiWorkerMirroredStrategy':
		pass

	else:
		print('Invalid distribution strategy')
		sys.exit()

	print("\n")

	# Export model graphs
	print(' export model graphs ')
	print(' ^^^^^^^^^^^^^^^^^^^\n')

	log_dir = '../results/' + model_type + '/' + output_folder
	os.makedirs(log_dir + 'graphs', exist_ok=True)
	model.export_graph(log_dir + 'graphs')

	# Training
	print(' model training ')
	print(' ^^^^^^^^^^^^^^\n')

	os.makedirs(log_dir + '/losses', exist_ok=True)

	if model_type == 'VAE':

		if add_param_file is None:
			additional_parameters = {'val_split': 0.2,
			                         'log_dir': log_dir,
			                         'num_workers': 24}
		else:
			additional_parameters = json.load(add_param_file)

		if tfrecord_data_file is None:
			loss, val_loss = model.train(mode='Dataset',
			                             **additional_parameters)

		else:
			loss, val_loss = model.train(mode='TFRecord',
			                             data_file=tfrecord_data_file,
			                             **additional_parameters)

		with open(log_dir + '/losses/loss.txt', 'w') as f:
			for k in loss:
				f.write("%s\n" % k)

		with open(log_dir + '/losses/val_loss.txt', 'w') as f:
			for k in val_loss:
				f.write("%s\n" % k)

		# Save training loss and validation loss to file
		print('Train loss and validation loss exported to file.')

	elif model_type == 'AAE1':
		if add_param_file is None:

			additional_parameters = {'enable_function': True,
			                         'graph': False,
			                         'gene': None}
		else:
			additional_parameters = json.load(add_param_file)

		BATCH_SIZE_PER_REPLICA = model.batch_size
		global_batch_size = (BATCH_SIZE_PER_REPLICA *
		                     strategy.num_replicas_in_sync)

		# Create input dataset
		if tfrecord_data_file is None:
			train_dataset = tf.data.Dataset.from_tensor_slices(model.data).shuffle(
				len(model.data)).repeat(model.epochs).batch(global_batch_size, drop_remainder=True).prefetch(
				buffer_size=1)
		else:
			train_dataset = data_generator(tfrecord_data_file,
			                               batch_size=global_batch_size,
			                               epochs=model.epochs,
			                               is_training=True)

		with strategy.scope():
			distributed_train_dataset = strategy.experimental_distribute_dataset(train_dataset)

			rec_loss, dis_loss = model.distributed_train(distributed_train_dataset, strategy,
			                                             log_dir=log_dir,
			                                             **additional_parameters)

			with open(log_dir + '/losses/rec_loss.txt', 'w') as f:
				for k in rec_loss:
					f.write("%s\n" % k)

			with open(log_dir + '/losses/dis_loss.txt', 'w') as f:
				for k in dis_loss:
					f.write("%s\n" % k)

			# Save training loss and validation loss to file
			print('Reconstruction loss and discriminator loss exported to file.')

	elif model_type == 'AAE2':
		if add_param_file is None:

			additional_parameters = {'enable_function': True,
			                         'graph': False,
			                         'gene': None}
		else:
			additional_parameters = json.load(add_param_file)

		BATCH_SIZE_PER_REPLICA = model.batch_size
		global_batch_size = (BATCH_SIZE_PER_REPLICA *
		                     strategy.num_replicas_in_sync)

		# Create input dataset
		if tfrecord_data_file is None:
			train_dataset = tf.data.Dataset.from_tensor_slices(model.data).shuffle(
				len(model.data)).repeat(model.epochs).batch(global_batch_size, drop_remainder=True).prefetch(
				buffer_size=1)
		else:
			train_dataset = data_generator(tfrecord_data_file,
			                               batch_size=global_batch_size,
			                               epochs=model.epochs,
			                               is_training=True)

		with strategy.scope():

			distributed_train_dataset = strategy.experimental_distribute_dataset(train_dataset)

			rec_loss, dis_loss, dis_cat_loss = model.distributed_train(distributed_train_dataset, strategy,
			                                                           log_dir=log_dir,
			                                                           **additional_parameters)

			with open(log_dir + '/losses/rec_loss.txt', 'w') as f:
				for k in rec_loss:
					f.write("%s\n" % k)

			with open(log_dir + '/losses/dis_loss.txt', 'w') as f:
				for k in dis_loss:
					f.write("%s\n" % k)

			with open(log_dir + '/losses/dis_cat_loss.txt', 'w') as f:
				for k in dis_cat_loss:
					f.write("%s\n" % k)

			# Save training loss and validation loss to file
			print('Reconstruction loss, discriminator loss, and categorical discriminator loss exported to file.')


if __name__ == "__main__":
	arguments = parser.parse_args()
	main(**vars(arguments))
