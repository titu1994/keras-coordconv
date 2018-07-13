import os
import numpy as np
from sklearn.model_selection import train_test_split

# Can be either `quadrant` or `uniform`
type = 'quadrant'

assert type in ['uniform', 'quadrant']

if not os.path.exists('data-uniform/'):
    os.makedirs('data-uniform/')

if not os.path.exists('data-quadrant/'):
    os.makedirs('data-quadrant/')


if __name__ == '__main__':
    import tensorflow as tf
    np.random.seed(0)
    tf.set_random_seed(0)

    # From https://arxiv.org/pdf/1807.03247.pdf
    onehots = np.pad(np.eye(3136, dtype='float32').reshape((3136, 56, 56, 1)),
                     ((0, 0), (4, 4), (4, 4), (0, 0)), mode="constant")

    images = tf.nn.conv2d(onehots, np.ones((9, 9, 1, 1)), [1] * 4, "SAME")

    # Get the images
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        images = sess.run(images)

    if type == 'uniform':
        # Create the uniform datasets
        indices = np.arange(0, len(onehots), dtype='int32')
        train, test = train_test_split(indices, test_size=0.2, random_state=0)

        train_onehot = onehots[train]
        train_images = images[train]

        test_onehot = onehots[test]
        test_images = images[test]

        np.save('data-uniform/train_onehot.npy', train_onehot)
        np.save('data-uniform/train_images.npy', train_images)
        np.save('data-uniform/test_onehot.npy', test_onehot)
        np.save('data-uniform/test_images.npy', test_images)

    else:
        # Create the quadrant datasets
        pos = np.where(onehots == 1.0)
        X = pos[1]
        Y = pos[2]

        train_set = []
        test_set = []

        train_ids = []
        test_ids = []

        for i, (x, y) in enumerate(zip(X, Y)):
            if x > 32 and y > 32:  # 4th quadrant
                test_ids.append(i)
                test_set.append([x, y])
            else:
                train_ids.append(i)
                train_set.append([x, y])

        train_set = np.array(train_set)
        test_set = np.array(test_set)

        train_set = train_set[:, None, None, :]
        test_set = test_set[:, None, None, :]

        print(train_set.shape)
        print(test_set.shape)

        train_onehot = onehots[train_ids]
        test_onehot = onehots[test_ids]

        train_images = images[train_ids]
        test_images = images[test_ids]

        print(train_onehot.shape, test_onehot.shape)
        print(train_images.shape, test_images.shape)

        np.save('data-quadrant/train_set.npy', train_set)
        np.save('data-quadrant/test_set.npy', test_set)
        np.save('data-quadrant/train_onehot.npy', train_onehot)
        np.save('data-quadrant/train_images.npy', train_images)
        np.save('data-quadrant/test_onehot.npy', test_onehot)
        np.save('data-quadrant/test_images.npy', test_images)
