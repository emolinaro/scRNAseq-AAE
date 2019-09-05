import tensorflow


# The following class is a workaround to solve the following error when using Tensorboard:
# AttributeError: 'TensorBoard' object has no attribute 'sess'
# Issue reported here:
# https://github.com/tensorflow/tensorboard/issues/1666

class TensorBoardWithSession(tensorflow.keras.callbacks.TensorBoard):

	def __init__(self, **kwargs):
		from tensorflow.python.keras import backend as K
		self.sess = K.get_session()

		super().__init__(**kwargs)

