import os
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_mean

from dataset import get_dataset, DATASETS
from preprocessing import get_graph_preprocessing_fn
from model.VanillaGCN_PointNetPool import Model, MODEL_NAME
from progress.bar import Bar
from utils.logger import log, logd, set_log_level, TimeScope
from utils.params import params as p


DATASET = DATASETS.ModelNet10
Dataset, CLASS_DICT = get_dataset(DATASET)

# === PARAMETERS ==============================================================
# Training parameters
p.define("max_epochs", 500)
p.define("batch_size", 32)
p.define("learning_rate", 0.001)
p.define("reg_constant", 0.01)
p.define("decay_steps", 10000)
p.define("decay_rate", 0.96)
p.define("val_set_pct", 0.05)

# Generic
set_log_level("INFO")
p.define("num_classes", len(CLASS_DICT))


p.load("params/{}_{}.yaml".format(DATASET.name, MODEL_NAME))
EXPERIMENT_VERSION = p.get_hash()
SAVE_DIR = "output_save/{}_{}_{}/".format(DATASET.name,
                                          MODEL_NAME,
                                          EXPERIMENT_VERSION)
os.system("mkdir -p " + SAVE_DIR)
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Clean previous experiments logs
    if len(os.listdir(SAVE_DIR)) != 0:
        if raw_input("A similar experiment was already saved."
                     " Would you like to delete it ?") is 'y':
            log("Experiment deleted !\n")
            os.system("rm -rf {}/*".format(SAVE_DIR))
        else:
            log("Keeping the records, tensorboard might be screwed up\n")

    p.save(SAVE_DIR + "params.yaml")

    # === SETUP ===============================================================
    with TimeScope("setup", debug_only=True):
        # --- Pre processing function setup -----------------------------------
        feat_compute = get_graph_preprocessing_fn(p)
        # --- Dataset setup ---------------------------------------------------
        dataset = Dataset(batch_size=p.batch_size,
                          val_set_pct=p.val_set_pct)

        # --- Model Setup -----------------------------------------------------
        batch = tf.Variable(0, trainable=False, name="step")
        model = Model()

        # --- Accuracy setup --------------------------------------------------
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(
                    tf.argmax(model.inference, 1),
                    tf.argmax(model.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32))
            tf.summary.scalar('avg', accuracy)

        # --- Optimisation Setup ----------------------------------------------
        learning_rate = tf.train.exponential_decay(p.learning_rate,
                                                   batch,
                                                   p.decay_steps,
                                                   p.decay_rate,
                                                   staircase=True)
        # Clip the learning rate !
        learning_rate = tf.maximum(learning_rate, 0.00001)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(model.loss,
                      global_step=batch,
                      name="optimizer")

        # --- Summaries and saver setup ---------------------------------------
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

    # === GRAPH COMPUTATION ===================================================
    with tf.Session() as sess:
        # Summaries Writer
        train_writer = tf.summary.FileWriter(SAVE_DIR + 'train_tb',
                                             sess.graph)
        val_writer = tf.summary.FileWriter(SAVE_DIR + 'val_tb')

        # Streaming summaries for validation set
        saccuracy, saccuracy_update = streaming_mean(accuracy)
        saccuracy_scalar = tf.summary.scalar('accuracy',
                                             saccuracy)
        sloss, sloss_update = streaming_mean(model.loss)
        sloss_scalar = tf.summary.scalar('total_loss',
                                         sloss)

        # Init
        sess.run(tf.global_variables_initializer())

        # Training
        log("Setup finished, starting training now ... \n\n")
        print "Parameters:"
        print p, "\n"

        for epoch in range(p.max_epochs):
            # --- Training step -----------------------------------------------
            train_iterator = dataset.train_batch(process_fn=feat_compute)
            bar_name = "Epoch {}/{}".format(epoch, p.max_epochs)
            bar = Bar(bar_name, max=dataset.train_batch_no)
            for idx, (xs, ys) in enumerate(train_iterator):
                with TimeScope("optimize", debug_only=True):
                    summary, loss, _ = sess.run(
                        [merged,
                         model.loss,
                         train_op],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=True))
                train_iter = idx + epoch*dataset.train_batch_no
                train_writer.add_summary(summary, train_iter)
                bar.next()
                # epoch_perc = 100. * idx / dataset.train_batch_no
                # log("Epoch {}/{} | {:.2f}%  loss: {}",
                #     epoch,
                #     p.max_epochs,
                #     epoch_perc,
                #     loss)
            bar.finish()

            # --- Validation accuracy -----------------------------------------
            # Re initialize the streaming means
            sess.run(tf.local_variables_initializer())

            for xs, ys in dataset.val_batch(process_fn=feat_compute):
                with TimeScope("validation", debug_only=True):
                    sess.run(
                        [saccuracy_update,
                         sloss_update],
                        feed_dict=model.get_feed_dict(xs, ys,
                                                      is_training=False))
            saccuracy_summ, cur_sacc, sloss_summ, cur_sloss = sess.run(
                [saccuracy_scalar, saccuracy,
                 sloss_scalar, sloss])
            cur_iter = epoch*dataset.train_batch_no
            val_writer.add_summary(saccuracy_summ, cur_iter)
            val_writer.add_summary(sloss_summ, cur_iter)

            log("Epoch {}/{} | Accuracy: {:.1f} (loss: {:.3f})\n",
                epoch,
                p.max_epochs,
                100.*cur_sacc,
                cur_sloss)

            # --- Save the model ----------------------------------------------
            if epoch % 10 == 0 and not epoch == 0:
                # Save the variables to disk.
                save_path = saver.save(
                    sess,
                    "{}model_{}/model.ckpt".format(SAVE_DIR, epoch))
                logd("Model saved in file: {}", save_path)

    dataset.close()
