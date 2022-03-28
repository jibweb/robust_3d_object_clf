from functools import partial
from dataset import get_dataset, DATASETS
import os
from utils.logger import log

EXTRACTOR_BIN_DIR = "/home/jbweibel/code/grocon/cmake_build/"
MESHING_EXE = EXTRACTOR_BIN_DIR + "meshing"
VIEW_EXTRACTOR_EXE = EXTRACTOR_BIN_DIR + "view_extractor"
NORM_PRECOMP_EXE = EXTRACTOR_BIN_DIR + "normal_precomputation"

# --- ModelNet 10 -------------------------------------------------------------
MN_DIR = "/home/jbweibel/dataset/ModelNet/"
MN10_SAVE_DIR_TRAIN = MN_DIR + "ModelNet10_TrainPc/"
MN10_SAVE_DIR_TEST = MN_DIR + "ModelNet10_TestPc/"


def precompute(in_fn, out_dir, exe):
    # For ModelNet10
    # cls_name = in_fn.split("/")[-2] + "/"

    # For ModelNet10OFF
    # cls_name = in_fn.split("/")[-3] + "/"
    # os.system("{} -i {} -o {}".format(exe, in_fn, out_dir+cls_name))

    # For ScanNet
    # in_fn = '\\ '.join(in_fn.split(" "))
    # out_dir = "/".join(in_fn.split("/")[:-1]) + "/"

    # For S3DIS
    in_fn = in_fn[:-4] + ".pcd"
    out_dir = "/".join(in_fn.split("/")[:-1]) + "/"

    os.system("{} -i {} -o {}".format(exe, in_fn, out_dir))
    return in_fn


if __name__ == "__main__":
    DATASET = DATASETS.S3DIS
    Dataset, CLASS_DICT = get_dataset(DATASET)
    dataset = Dataset(balance_train_set=False,
                      balance_test_set=False,
                      batch_size=10,
                      val_set_pct=0.)
    executable = NORM_PRECOMP_EXE

    # mn10_train = partial(precompute, exe=executable,
    #                      out_dir=MN10_SAVE_DIR_TRAIN)
    # mn10_test = partial(precompute, exe=executable,
    #                     out_dir=MN10_SAVE_DIR_TEST)
    scannet_all = partial(precompute, exe=executable,
                          out_dir="")

    train_it = dataset.train_batch(process_fn=scannet_all)
    test_it = dataset.test_batch(process_fn=scannet_all)

    for idx, (fns, yns) in enumerate(train_it):
        log("{:.1f} % ", 100.*idx/dataset.train_batch_no)

    print "\n\nFINISHED TRAIN SET\n\n"

    for idx, (fns, yns) in enumerate(test_it):
        log("{:.1f} % ", 100.*idx/dataset.test_batch_no)

    print "\n\nFINISHED TEST SET\n\n"

    if DATASET == DATASETS.ScanNet:
        val_it = dataset.val_batch(process_fn=scannet_all)
        for idx, (fns, yns) in enumerate(val_it):
            log("{:.1f} % ", 100.*idx/dataset.val_batch_no)
