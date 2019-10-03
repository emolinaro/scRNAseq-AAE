import tensorflow as tf

class PColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



# The following class is a workaround to solve the following error when using Tensorboard:
# AttributeError: 'TensorBoard' object has no attribute 'sess'
# Issue reported here:
# https://github.com/tensorflow/tensorboard/issues/1666

class TensorBoardWithSession(tf.keras.callbacks.TensorBoard):

	def __init__(self, **kwargs):
		from tf.python.keras import backend as K
		self.sess = K.get_session()

		super().__init__(**kwargs)


# Helperfunctions to make your feature definition more readable
def _int64_feature(value):
	"""
		Returns an int64_list from a bool / enum / int / uint.

	"""
	return tf.train.Feature(
		int64_list=tf.train.Int64List(value=[value])
	)


def _floats_feature(value):
	"""
		Returns a float_list from a float / double.

	"""
	return tf.train.Feature(
		float_list=tf.train.FloatList(value=[value])
	)


def _bytes_feature(value):
	"""
		Returns a bytes_list from a string / byte.

	"""
	return tf.train.Feature(
		bytes_list=tf.train.BytesList(value=[value])
	)


def export_to_tfrecord(filepath, adata, val_split=0.2):
	"""Export dataset from h5ad format to TFRecord format.

	   :param filepath:
			  path of output file
	   :type filepath: str

	   :param adata:
			  an annotated data matrix
	   :type adata: AnnData object

	   :param val_split:
			  percentage of data assigned to validation dataset

	   :return: TFRecord formatted files
	"""

	data = adata.X

	dim = data.shape[0]

	train_size = int(dim * (1 - val_split))
	val_size = dim - train_size

	labels = adata.obs['louvain'].values.astype(int)

	# create train dataset
	writer = tf.python_io.TFRecordWriter(filepath + ".train")
	for i in range(train_size):
		feature = {'data': _bytes_feature(tf.compat.as_bytes(data[i].tostring())),
		           'label': _int64_feature(int(labels[i]))}

		# Serialize to string and write to file
		single = tf.train.Example(features=tf.train.Features(feature=feature))

		writer.write(single.SerializeToString())

	# create train dataset
	writer = tf.python_io.TFRecordWriter(filepath + ".val")
	for i in range(val_size):
		feature = {'data': _bytes_feature(tf.compat.as_bytes(data[train_size + i].tostring())),
		           'label': _int64_feature(int(labels[train_size + i]))}

		# Serialize to string and write to file

		single = tf.train.Example(features=tf.train.Features(feature=feature))

		writer.write(single.SerializeToString())


def data_generator(filepath, batch_size=35, epochs=200, is_training=True):
	"""Build data pipeline.

	:param filepath:
	:param batch_size:
	:param data_size:
	:param epochs:
	:return:
	"""

	def _parse_function(proto):
		"""Parse TFExample records.

		:param proto:
		:return:
		"""
		# data fetures
		keys_to_features = {'data': tf.io.FixedLenFeature((), tf.string, ""),
		                    'label': tf.io.FixedLenFeature((), tf.int64, -1)}

		# Load one example
		parsed_features = tf.io.parse_single_example(proto, keys_to_features)

		# turn saved data string into an array
		parsed_features['data'] = tf.decode_raw(parsed_features['data'], tf.float32)

		return parsed_features['data']

	num_cpus = 24

	# load TFRecords and create a Dataset object
	files = tf.data.Dataset.list_files(filepath)
	# dataset = files.interleave(tf.data.TFRecordDataset)

	dataset = tf.data.TFRecordDataset(files, num_parallel_reads=num_cpus)

	# set the number of datapoints you want to load and shuffle
	if is_training:
		dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

	# repeat the dataset
	dataset = dataset.repeat(epochs)

	# parallelize data transformation
	dataset = dataset.map(_parse_function, num_parallel_calls=num_cpus)

	# set the batchsize
	dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)

	# prefetch elements from the input dataset ahead of the time they are requested
	dataset = dataset.prefetch(buffer_size=1)

	return dataset
