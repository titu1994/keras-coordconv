import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Conv2D, Flatten, Softmax
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

sys.path.append('..')
from coord import CoordinateChannel2D


if not os.path.exists('data-uniform/train_images.npy'):
    print("Please run `make_dataset.py` first")
    exit()

# Load the one hot datasets
train_onehot = np.load('data-uniform/train_onehot.npy').astype('float32')
test_onehot = np.load('data-uniform/test_onehot.npy').astype('float32')

# make the train and test datasets
pos = np.where(train_onehot == 1.0)
X = pos[1]
Y = pos[2]

train_set = np.zeros((len(X), 1, 1, 2), dtype='float32')

for i, (x, y) in enumerate(zip(X, Y)):
    train_set[i, 0, 0, 0] = x
    train_set[i, 0, 0, 1] = y

# test dataset
pos = np.where(test_onehot == 1.0)
X = pos[1]
Y = pos[2]

test_set = np.zeros((len(X), 1, 1, 2), dtype='float32')

for i, (x, y) in enumerate(zip(X, Y)):
    test_set[i, 0, 0, 0] = x
    test_set[i, 0, 0, 1] = y

train_set = np.tile(train_set, [1, 64, 64, 1])
test_set = np.tile(test_set, [1, 64, 64, 1])

# Normalize the datasets
train_set /= (64. - 1.)  # 64x64 grid, 0-based index
test_set /= (64. - 1.)  # 64x64 grid, 0-based index

print('Train set : ', train_set.shape, train_set.max(), train_set.min())
print('Test set : ', test_set.shape, test_set.max(), test_set.min())

# Visualize the datasets

# plt.imshow(np.sum(train_onehot, axis=0)[:, :, 0], cmap='gray')
# plt.title('Train One-hot dataset')
# plt.show()
# plt.imshow(np.sum(test_onehot, axis=0)[:, :, 0], cmap='gray')
# plt.title('Test One-hot dataset')
# plt.show()

# flatten the datasets
train_onehot = train_onehot.reshape((-1, 64 * 64))
test_onehot = test_onehot.reshape((-1, 64 * 64))

# model definition

ip = Input(shape=(64, 64, 2))
x = CoordinateChannel2D()(ip)
x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)

x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
x = Conv2D(1, (1, 1), padding='same', activation='linear')(x)
x = Flatten()(x)
x = Softmax(axis=-1)(x)

model = Model(ip, x)
model.summary()


optimizer = Adam(lr=1e-3)
model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('data-uniform/coordconv.h5', monitor='val_loss',
                             verbose=1, save_best_only=True, save_weights_only=True)

# train model
# model.fit(train_set, train_onehot, batch_size=32, epochs=10,
#           verbose=1, callbacks=[checkpoint],
#           validation_data=(test_set, test_onehot))

# evaluate model
model.load_weights('data-uniform/coordconv.h5')

# visualize the predictions
preds = model.predict(test_set)
print(np.min(preds), np.max(preds))

preds = preds.reshape((-1, 64, 64, 1))

plt.imshow(np.sum(preds, axis=0)[:, :, 0], cmap='gray')
plt.title('Predictions')
plt.show()

scores = model.evaluate(test_set, test_onehot, batch_size=128, verbose=1)

print()
for name, score in zip(model.metrics_names, scores):
    print(name, score)

