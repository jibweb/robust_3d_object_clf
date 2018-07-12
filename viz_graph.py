import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from dataset import get_dataset, DATASETS
from utils.params import params as p
from preprocessing import get_graph_preprocessing_fn


if __name__ == "__main__":
    DATASET = DATASETS.ModelNet10
    Dataset, CLASS_DICT = get_dataset(DATASET)
    d = Dataset(batch_size=1, val_set_pct=0)

    p.gridsize = 64
    p.nodes_nb = 128
    p.neigh_size = 0.15
    p.neigh_nb = 5
    p.feats_3d = True
    if p.feats_3d:
        p.feat_nb = 4
    else:
        p.feat_nb = 33

    p.debug = True
    p.viz = True
    p.viz_small_spheres = True

    p.to_remove = 0.
    p.to_keep = 5000

    feat_compute = get_graph_preprocessing_fn(p)

    d.prepare_sets()
    train_set = shuffle(d.train_x)

    for fn in train_set[:5]:
        print fn
        feats, bias = feat_compute(fn)
        print np.mean(np.isnan(feats))
        # plt.imshow(bias)
        # plt.show()
