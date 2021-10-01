import tensorflow as tf


def xception_model(input_shape=(480,270,3), weights='imagenet', include_top=False, num_labels=9):
	# Xception base model for self driving cars in GTA-V (2021)
	
	_base  = tf.keras.applications.xception.Xception(weights=weights, input_shape=input_shape, include_top=include_top)

	x = _base.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dense(1024, activation='relu')(x)
	output_fc = tf.keras.layers.Dense(num_labels, activation='softmax', name='output_layer')(x)

	model = tf.keras.models.Model(inputs=_base.input, outputs=output_fc)

	return model

if __name__ == '__main__':
	print('Model is:')
	model = xception_model(weights=None)
	print(model.summary())
