from keras.models import load_model
import numpy as np

# read the whole image data and reshape it
X = np.load('img.npy')
X = X.astype('float32') / 255
X = np.reshape(X, (40000, 32, 32, 3))
print("Shape of X is: ", X.shape)

# load our autoencoder model, predict the result and save it
encoder0 = load_model('encoder.h5')
encoder0.summary()
encode_img0 = encoder0.predict(X)
encode_img = encode_img0.reshape(encode_img0.shape[0], -1)
np.save('predict.npy', encode_img)
