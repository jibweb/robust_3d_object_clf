from logger import TimeScope
import numpy as np
import tensorflow as tf
from utils.tf import fc, fc_bn, weight_variable, bias_variable,\
                     batch_norm_for_conv1d, batch_norm_for_conv2d,\
                     define_scope
from utils.params import params as p

from graph_attention import attn_head

MODEL_NAME = "VanillaGAT"

p.define("learning_rate", 0.001)
p.define("reg_constant", 0.01)
p.define("decay_steps", 10000)
p.define("decay_rate", 0.96)


class Model(object):
    def __init__(self,
                 bn_decay=None):
        # I/O Tensors
        with TimeScope(MODEL_NAME + "/placeholder_setup", debug_only=True):
            self.x_local = tf.placeholder(tf.float32,
                                          [None,
                                           p.sal_pt_num,
                                           p.neigh_size,
                                           p.local_feat_num],
                                          name="x_local")
            self.x_global_p = tf.placeholder(tf.float32,
                                             [None,
                                              3*p.triplet_num,
                                              p.global_feat_p_num],
                                             name="x_global_p")
            self.x_global_t = tf.placeholder(tf.float32,
                                             [None,
                                              p.triplet_num,
                                              p.global_feat_t_num],
                                             name="x_global_t")
            self.x_valid_pt = tf.placeholder(tf.float32,
                                             [None,
                                              p.sal_pt_num,
                                              p.sal_pt_num],
                                             name="x_valid_pt")
            self.y = tf.placeholder(tf.float32,
                                    [None, p.num_classes],
                                    name="y")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.bn_decay = bn_decay

        # Model properties
        with TimeScope(MODEL_NAME + "/prop_setup", debug_only=True):
            self.inference
            self.loss
            # self.optimize

    def get_feed_dict(self, x_batch, y_batch, is_training):
        xb_local = [x_i[0] for x_i in x_batch]
        xb_global_p = [np.array(x_i[1]) for x_i in x_batch]
        xb_global_t = [np.array(x_i[2]) for x_i in x_batch]
        xb_valid_pt = [np.diag([1.]*x_i[3] + [0.]*(p.sal_pt_num - x_i[3]))
                       for x_i in x_batch]

        keep_prob = p.dropout_prob if is_training else 1.

        return {
            self.x_local: xb_local,
            self.x_global_p: xb_global_p,
            self.x_global_t: xb_global_t,
            self.x_valid_pt: xb_valid_pt,
            self.y: y_batch,
            self.keep_prob: keep_prob,
            self.is_training: is_training
        }

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """
        attns = []
        for _ in range(n_heads[0]):
            attns.append(attn_head(inputs, bias_mat=bias_mat,
                                   out_sz=hid_units[0],
                                   activation=activation,
                                   in_drop=ffd_drop,
                                   coef_drop=attn_drop,
                                   residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i],
                                              activation=activation,
                                              in_drop=ffd_drop,
                                              coef_drop=attn_drop,
                                              residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes,
                                        activation=lambda x: x,
                                        in_drop=ffd_drop,
                                        coef_drop=attn_drop,
                                        residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!! SET POOLING !!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return logits

    @define_scope
    def loss(self):
        # Cross-entropy loss
        with tf.variable_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y,
                    logits=self.inference)

            cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy_avg', cross_entropy)

        reg_loss = tf.losses.get_regularization_loss()
        tf.summary.scalar('regularization_loss_avg', reg_loss)

        total_loss = cross_entropy + reg_loss
        tf.summary.scalar('total_loss', total_loss)
        return total_loss
