# encoding: utf-8
#

import os, sys
import numpy as np
import scipy


from scipy.cluster.vq import kmeans, vq
import numpy as np


assert len(sys.argv) == 4
K = int(sys.argv[1])
scpfn = sys.argv[2]
outfn = sys.argv[3]

# load feature
scplist = []
with open(scpfn, 'rt') as f:
	for line in f:
		line = line.strip()
		if line != "" and line[0] != "#":
			scplist.append(line)
emo = []
for emofn in scplist:
	emo.append(np.fromfile(emofn, dtype=np.float32))
print("load from", scpfn, "len =", len(emo))

# input limited
emo = np.array(emo) # (L, 1024)
np.random.shuffle(emo)
emo = emo[:5000]
print("shuffle, len =", len(emo))

# remove 10% of outliers
mean = np.mean(emo)
dist = np.linalg.norm(emo - mean, 2, -1)
x = np.argsort(dist)
emo = emo[x]
emo = emo[:int(0.9*len(emo))]
print("remove, len =", len(emo))

# cluster
center, _ = kmeans(emo, min(K, len(emo)))
print("center.shape =", center.shape)
#cluster, _ = vq(emo, center)

'''
# nearest
dist = np.expand_dims(emo, 0) - np.expand_dims(center, 1) # (K, L, 1024)
dist = np.linalg.norm(dist, 2, -1) # (K, L)
idx = dist.argmin(1) # (K,)
print("nearest indices =", idx)
nearest = emo[idx]
'''

# save k-means
center.tofile(outfn)
#nearest.tofile(outfn)
print("save to", outfn)


