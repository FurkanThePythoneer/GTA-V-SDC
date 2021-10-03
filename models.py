import tensorflow as tf
import keras_efficientnet_v2

def xception_model(input_shape=(480,270,3), weights='imagenet', include_top=False, num_labels=9):
	# Xception base model for self driving cars in GTA-V (2021)
	
	_base  = tf.keras.applications.xception.Xception(weights=weights, input_shape=input_shape, include_top=include_top)

	x = _base.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dense(1024, activation='relu')(x)
	output_fc = tf.keras.layers.Dense(num_labels, activation='softmax', name='output_layer')(x)

	model = tf.keras.models.Model(inputs=_base.input, outputs=output_fc)

	return model


def effnetv2_b2_model(input_shape=(480,270,3), weights='imagenet', include_top=False, num_labels=9):
	# Xception base model for self driving cars in GTA-V (2021)
	
	_base = keras_efficientnet_v2.EfficientNetV2B2(input_shape=input_shape, num_classes=0, pretrained=weights)


	x = _base.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dense(1024, activation='relu')(x)
	output_fc = tf.keras.layers.Dense(num_labels, activation='softmax', name='output_layer')(x)

	model = tf.keras.models.Model(inputs=_base.input, outputs=output_fc)

	return model



if __name__ == '__main__':
	print('Model is:')
	model = effnetv2_b2_model(weights=None)
	print(model.summary())
