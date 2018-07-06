import argparse
import os
from train import SAVE_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", help="port to run on")
    parser.add_argument("-d", "--save_dir", help="Choose experiment manually",
                        default=SAVE_DIR)
    args = parser.parse_args()

    logdir_opts = "--logdir=train:{}train_tb/".format(args.save_dir)
    logdir_opts += ",val:{}val_tb/".format(args.save_dir)
    port_opts = ""
    if args.port:
        port_opts = "--port {}".format(args.port)

    called = ["tensorboard", logdir_opts, port_opts]
    print "\n\n", " ".join(called)
    os.system(" ".join(called))
