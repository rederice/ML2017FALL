from skimage import io
from skimage import transform
import numpy as np
import os
import sys
jpgfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk(sys.argv[1])
             for name in files
             if name.endswith((".jpg"))]
ims = []
for i in range(len(jpgfiles)):
	img = io.imread(jpgfiles[i])
	ims.append(img.flatten())

ims = np.array(ims).T
x_mean = np.mean(ims, axis=1)

eigenface, eigenvalue, v = np.linalg.svd( (ims.T - x_mean).T, full_matrices=False )
# np.save("eigenface",eigenface[:, :4])
# np.save("eigenvalue",eigenvalue)

# eigenface = np.load("eigenface.npy")
# eigenvalue = np.load("eigenvalue.npy")

img0 = io.imread(jpgfiles[0])

w0 = img0.flatten().dot(eigenface[:,0])*eigenface[:,0]
w1 = img0.flatten().dot(eigenface[:,1])*eigenface[:,1]
w2 = img0.flatten().dot(eigenface[:,2])*eigenface[:,2]
w3 = img0.flatten().dot(eigenface[:,3])*eigenface[:,3]
w = w0+w1+w2+w3+x_mean.T
tmp = w.T
tmp -= np.min(tmp)
tmp /= np.max(tmp)
tmp=(tmp*255).astype(np.uint8)
tmp=tmp.reshape(600,600,3)
io.imsave(sys.argv[2], tmp)
