import argparse
import os
from train import p, DEFAULT_PARAMS_FILE, DATASET, MODEL_NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="port to run on")
    # parser.add_argument("-d", "--save_dir", help="Choose experiment manually",
    #                     default=SAVE_DIR)
    args = parser.parse_args()

    p.load(DEFAULT_PARAMS_FILE)
    EXPERIMENT_VERSION = p.get_hash()
    SAVE_DIR = "output_save/{}_{}_{}/".format(DATASET.name,
                                              MODEL_NAME,
                                              EXPERIMENT_VERSION)
    logdir_opts = "--logdir=train:{}train_tb/".format(SAVE_DIR)
    logdir_opts += ",val:{}val_tb/".format(SAVE_DIR)
    port_opts = ""
    if args.port:
        port_opts = "--port {}".format(args.port)

    called = ["tensorboard", logdir_opts, port_opts]
    print "\n\n", " ".join(called)
    os.system(" ".join(called))
