import numpy as np
import tensorflow as tf

# same random input for both
x = np.random.random((1, 30, 63)).astype(np.float32)

# 1. Keras .h5 prediction
keras_model = tf.keras.models.load_model('gesto_model.h5')
keras_out = keras_model.predict(x, verbose=0)[0]

# 2. TFLite prediction
interp = tf.lite.Interpreter(model_path='gesto_new.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]
outp = interp.get_output_details()[0]
interp.set_tensor(inp['index'], x)
interp.invoke()
tflite_out = interp.get_tensor(outp['index'])[0]

print("Keras :", np.round(keras_out, 4))
print("TFLite:", np.round(tflite_out, 4))
print("Max diff:", np.abs(keras_out - tflite_out).max())