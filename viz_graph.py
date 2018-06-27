from sklearn.utils import shuffle

from dataset import get_dataset, DATASETS
from utils.params import params as p
from preprocessing import get_graph_preprocessing_fn


if __name__ == "__main__":
    DATASET = DATASETS.ModelNet10
    Dataset, CLASS_DICT = get_dataset(DATASET)
    d = Dataset(batch_size=1, val_set_pct=0)

    p.gridsize = 128
    p.nodes_nb = 128
    p.neigh_size = 0.1
    p.neigh_nb = 5
    p.debug = True
    p.viz = True

    feat_compute = get_graph_preprocessing_fn(p)

    d.prepare_sets()
    train_set = shuffle(d.train_x)

    for fn in train_set[:5]:
        print fn
        feats, bias = feat_compute(fn)
