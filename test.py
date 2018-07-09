import itertools
from utils.logger import log, set_log_level, TimeScope
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from dataset import get_dataset
from preprocessing import get_graph_preprocessing_fn
from train import Model, MODEL_NAME, DATASET, SAVE_DIR
from utils.params import params as p

Dataset, CLASS_DICT = get_dataset(DATASET)
# ---- Parameters ----

# Model parameters
# ...
# SAVE_DIR = "../saved_output/ModelNet10_clean/"

# Generic
# p.batch_size = 20
set_log_level("INFO")
p.define("test_repeat", 5)
p.define("model_ckpt", "model_2990/model.ckpt")


# --------------------


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    # Clean previous experiments logs
    os.system("rm -rf {}test_tb/*".format(SAVE_DIR))
    p.load("params/{}_{}.yaml".format(DATASET.name, MODEL_NAME))

    # === SETUP ===============================================================
    # --- Pre processing function setup -----------------------------------
    feat_compute = get_graph_preprocessing_fn(p)
    # --- Dataset setup -------------------------------------------------------
    dataset = Dataset(batch_size=p.batch_size,
                      val_set_pct=p.val_set_pct)

    # --- Model Setup ---------------------------------------------------------
    model = Model()

    # --- Accuracy setup ------------------------------------------------------
    with tf.variable_scope('accuracy'):
        correct_prediction = tf.equal(
                tf.argmax(model.inference, 1),
                tf.argmax(model.y, 1))
        confusion = tf.confusion_matrix(
                labels=tf.argmax(model.y, 1),
                predictions=tf.argmax(model.inference, 1),
                num_classes=p.num_classes)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                          tf.float32))
        tf.summary.scalar('avg', accuracy)

    # --- Summaries and saver setup -------------------------------------------
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    # === GRAPH COMPUTATION ===================================================
    with tf.Session() as sess:
        saver.restore(sess, SAVE_DIR + p.model_ckpt)

        # Summaries Writer
        test_writer = tf.summary.FileWriter(SAVE_DIR + 'test_tb')

        # Testing
        log("Setup finished, starting testing now ... \n\n")
        print "Parameters:"
        print p, "\n"
        test_iter = 0
        total_acc = 0.
        total_cm = np.zeros((p.num_classes, p.num_classes), dtype=np.int32)

        for repeat in range(p.test_repeat):
            for xs, ys in dataset.test_batch(process_fn=feat_compute):
                with TimeScope("accuracy", debug_only=True):
                    summary, acc, loss, cm = sess.run(
                        [merged,
                         accuracy,
                         model.loss,
                         confusion],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=False))
                    test_writer.add_summary(summary, test_iter)

                total_acc = (test_iter*total_acc + acc) / (test_iter + 1)
                total_cm += cm

                log("Accurracy: {:.1f} / {:.1f} (loss: {:.3f})\n",
                    100.*total_acc,
                    100.*acc,
                    loss)

                test_iter += 1

        plot_confusion_matrix(total_cm, sorted(CLASS_DICT.keys()))
        plt.show()
