import numpy as np

ip_shape = (10, 5, 20)
batch, d, c = ip_shape
ip = np.ones(ip_shape, dtype='float32')

# 1D conv

xx_range = np.tile(np.expand_dims(np.arange(0, d), axis=0),
                   [batch, 1])
print('xx range', xx_range.shape)
xx_range = np.expand_dims(xx_range, axis=-1)
print('xx range expand dims', xx_range.shape, xx_range[0])

final = np.concatenate([ip, xx_range], axis=-1)

print("final", final.shape)
print(final[0])
