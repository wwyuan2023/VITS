# encoding: utf-8
#

import os, sys
import numpy as np
import scipy


from scipy.cluster.vq import kmeans, vq
import numpy as np


assert len(sys.argv) == 4
clusterfn = sys.argv[1]
scpfn = sys.argv[2]
outdir = sys.argv[3]

# load .emo
center = np.fromfile(clusterfn, np.float32).reshape(-1, 1024)

# vq
with open(scpfn, 'rt') as f:
    for emofn in f:
        emofn = emofn.strip()
        if emofn == "" or emofn[0] == "#":
            continue
        emo = np.fromfile(emofn, dtype=np.float32).reshape(1, 1024)
        code, _ = vq(emo, center)
        emo = center[code]
        outfn = os.path.join(outdir, os.path.basename(emofn))
        emo.tofile(outfn)
        print(f"{code}, Output to {outfn}")
