from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from dataset import get_dataset, DATASETS
from utils.params import params as p
from preprocessing import get_graph_preprocessing_fn


def viz_n_hop(adj, hop_nb):
    mt = adj
    for idx in range(hop_nb):
        mt = (np.matmul(adj, mt) > 0).astype(adj.dtype)

    plt.imshow(mt)
    plt.show()


if __name__ == "__main__":
    DATASET = DATASETS.ModelNet10
    Dataset, CLASS_DICT = get_dataset(DATASET)
    d = Dataset(batch_size=1, val_set_pct=0)

    p.gridsize = 64
    p.nodes_nb = 512
    p.neigh_size = 0.01
    p.neigh_nb = 9
    p.edge_feats = True
    p.feats_3d = False
    if p.feats_3d:
        p.feat_nb = 4
    else:
        p.feat_nb = 3

    p.debug = True
    p.viz = False
    p.viz_small_spheres = True

    p.to_remove = 0.
    p.to_keep = 100000

    feat_compute = get_graph_preprocessing_fn(p)

    d.prepare_sets()
    train_set = shuffle(d.train_x)

    for fn in train_set[:10]:
        print fn
        if p.edge_feats:
            feats, bias, edge_feats = feat_compute(fn)
        else:
            feats, bias = feat_compute(fn)
        print "% of NaN:", np.mean(np.isnan(feats))
        # plt.imshow(bias)
        # plt.show()

        # for node_idx in range(p.nodes_nb)[:40]:
        #     # arr = feats[node_idx, ..., 0]
        #     # xdata, ydata, zdata = arr.nonzero()
        #     # cdata = arr[arr.nonzero()]
        #     xdata = [[i]*16 for i in range(4)]
        #     xdata = [elt for sublist in xdata for elt in sublist]
        #     ydata = [[i]*4 for i in range(4)]*4
        #     ydata = [elt for sublist in ydata for elt in sublist]
        #     zdata = range(4)*16
        #     cdata = feats[node_idx, ..., 0]

        #     fig = plt.figure()
        #     ax = plt.axes(projection='3d')
        #     ax.scatter3D(xdata, ydata, zdata, c=cdata, cmap='viridis')
        #     plt.show()
