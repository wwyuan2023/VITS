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
print("load from", scpfn, "len=", len(emo))

# cluster
center, _ = kmeans(emo, min(K, len(emo)))
print(center.shape)
#cluster, _ = vq(emo, center)

# save k-means
center.tofile(outfn)
print("save to", outfn)


