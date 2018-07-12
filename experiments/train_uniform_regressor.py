import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from coord import CoordinateChannel2D

sess = tf.Session()
K.set_session(sess)

np.random.seed(0)


if not os.path.exists('data-uniform/train_images.npy'):
    print("Please run `make_dataset.py` first")
    exit()

train_onehot = np.load('data-uniform/train_onehot.npy').astype('float32')
test_onehot = np.load('data-uniform/test_onehot.npy').astype('float32')

# train dataset
pos = np.where(train_onehot == 1.0)
X = pos[1]
Y = pos[2]

train_set = np.zeros((len(X), 2), dtype='float32')

for i, (x, y) in enumerate(zip(X, Y)):
    train_set[i, 0] = x
    train_set[i, 1] = y

# test dataset
pos = np.where(test_onehot == 1.0)
X = pos[1]
Y = pos[2]

test_set = np.zeros((len(X), 2), dtype='float32')

for i, (x, y) in enumerate(zip(X, Y)):
    test_set[i, 0] = x
    test_set[i, 1] = y

train_set /= (64. - 1.)  # 64x64 grid, 0-based index
test_set /= (64. - 1.)  # 64x64 grid, 0-based index

print('Train set : ', train_set.shape, train_set.max(), train_set.min())
print('Test set : ', test_set.shape, test_set.max(), test_set.min())

# plt.imshow(np.sum(train_onehot, axis=0)[:, :, 0], cmap='gray')
# plt.show()
# plt.imshow(np.sum(test_onehot, axis=0)[:, :, 0], cmap='gray')
# plt.show()
# plt.imshow(np.sum(np.concatenate((train_onehot, test_onehot)), axis=0)[:, :, 0], cmap='gray')
# plt.show()

# model definition

ip = Input(shape=(64, 64, 1))
x = CoordinateChannel2D()(ip)
x = Conv2D(8, (1, 1), padding='same', activation='relu')(x)

x = Conv2D(8, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(8, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(2, activation='sigmoid')(x)

model = Model(ip, x)
print(model.summary())


optimizer = Adam(lr=1e-3)
model.compile(optimizer, 'binary_crossentropy')

checkpoint = ModelCheckpoint('data-uniform/coordconv-reg.h5', monitor='val_loss',
                             verbose=1, save_best_only=True, save_weights_only=True)

# # train model
model.fit(train_onehot, train_set, batch_size=32, epochs=10,
          verbose=1, callbacks=[checkpoint],
          validation_data=(test_onehot, test_set))

# evaluate model
model.load_weights('data-uniform/coordconv-reg.h5')

preds = model.predict(test_onehot)
preds *= 64
preds = preds.astype('int32')
print(np.min(preds), np.max(preds))

images = np.zeros((len(preds), 64, 64, 1), dtype='float32')
for i, pred in enumerate(preds):
    images[i, pred[0], pred[1], 0] = 1.

plt.imshow(np.sum(images, axis=0)[:, :, 0], cmap='gray')
plt.show()

