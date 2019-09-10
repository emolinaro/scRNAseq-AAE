import tensorflow as tf


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


def export_to_tfrecord(filepath, adata):
	"""Export dataset from h5ad format to TFRecord format.

	   :param filepath:
	          path of output file
	   :type filepath: str

	   :param adata:
	          an annotated data matrix
	   :type adata: AnnData object

	   :return: TFRecord formatted file
	"""

	data = adata.X

	dim = data.shape[0]

	labels = adata.obs['louvain'].values.astype(int)

	writer = tf.python_io.TFRecordWriter(filepath)

	for i in range(dim):
		feature = {'data': _bytes_feature(tf.compat.as_bytes(data[i].tostring())),
		           'label': _int64_feature(int(labels[i]))}

		# Serialize to string and write to file

		single = tf.train.Example(features=tf.train.Features(feature=feature))

		writer.write(single.SerializeToString())


def data_generator(filepath, batch_size=35, data_size=1000, epochs=200):
	"""

	:param filepath:
	:param batch_size:
	:param data_size:
	:param epochs:
	:return:
	"""

	def _parse_function(proto):
		"""

		:param proto:
		:return:
		"""
		# define your tfrecord again. Remember that you saved your data as a string.
		keys_to_features = {'data': tf.io.FixedLenFeature([], tf.string),
		                    'label': tf.io.FixedLenFeature([], tf.int64)}

		# Load one example
		parsed_features = tf.io.parse_single_example(proto, keys_to_features)

		# Turn saved data string into an array
		parsed_features['data'] = tf.decode_raw(
			parsed_features['data'], tf.float32)

		return parsed_features['data']

	# This works with arrays as well
	dataset = tf.data.TFRecordDataset(filepath)

	# Maps the parser on every filepath in the array. You can set the number of parallel loaders here
	dataset = dataset.map(_parse_function, num_parallel_calls=1)

	# This dataset will go on forever
	dataset = dataset.repeat(epochs)

	# Set the number of datapoints you want to load and shuffle
	dataset = dataset.shuffle(data_size)

	dataset = dataset.take(data_size)

	# Set the batchsize
	dataset = dataset.batch(batch_size)

	# Create an iterator
	# iterator = dataset.make_one_shot_iterator()

	return dataset  # iterator
