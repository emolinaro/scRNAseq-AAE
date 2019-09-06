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


def export_to_tfrecord(filepath, data, labels):
	"""
		Export dataset from h5ad format to tfrecord format.

	"""

	writer = tf.python_io.TFRecordWriter(filepath)

	dim = len(labels)

	for i in range(dim):
		feature = {'data': _bytes_feature(tf.compat.as_bytes(data[i].tostring())),
		           'label': _int64_feature(int(labels[i]))}

		# Serialize to string and write to file

		single = tf.train.Example(features=tf.train.Features(feature=feature))

		writer.write(single.SerializeToString())
