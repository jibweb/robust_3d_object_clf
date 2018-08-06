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


def viz_lrf_coords(edge_feats):
    coords_lrf = edge_feats[:, :, 1:4]
    for i in range(p.nodes_nb):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        ax.scatter3D(coords_lrf[i, :, 0],
                     coords_lrf[i, :, 1],
                     coords_lrf[i, :, 2])
        plt.show()


def viz_pc(feats):
    assert p.feat_nb == 3
    xdata = feats[:, 0]
    ydata = feats[:, 1]
    zdata = feats[:, 2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.scatter3D(xdata, ydata, zdata)
    plt.show()

if __name__ == "__main__":
    DATASET = DATASETS.ModelNet10
    Dataset, CLASS_DICT = get_dataset(DATASET)
    d = Dataset(batch_size=1, val_set_pct=0)

    p.gridsize = 64
    p.nodes_nb = 256
    p.neigh_size = 0.01
    p.neigh_nb = -1
    p.edge_feats = True
    p.edge_feats_nb = 5
    p.feats_3d = False
    if p.feats_3d:
        p.feat_nb = 4
    else:
        p.feat_nb = 3

    p.debug = True
    p.viz = True
    p.viz_small_spheres = True

    p.to_remove = 0.
    p.to_keep = 5000
    p.rotation_deg = 180

    feat_compute = get_graph_preprocessing_fn(p)

    d.prepare_sets()
    # train_set = shuffle(d.train_x)
    train_set = ["/home/jbweibel/dataset/ModelNet/ModelNet10_TrainPc/monitor/monitor_0428_dist_2.000000_full_wnormals_wattention.pcd"] * 10

    for fn in train_set[:10]:
        print fn
        feats, bias, edge_feats, valid_indices = feat_compute(fn)
        print "% of NaN:", np.mean(np.isnan(feats))
        viz_pc(feats)
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
