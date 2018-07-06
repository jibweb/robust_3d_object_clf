import numpy as np
import tensorflow as tf
from utils.logger import TimeScope
from utils.tf import fc, fc_bn, define_scope
from utils.params import params as p

from layers import attn_head, g_k

MODEL_NAME = "VanillaGAT_PointNetPool"

# Dropout prob params
p.define("attn_drop_prob", 0.0)
p.define("feat_drop_prob", 0.0)
p.define("pool_drop_prob", 0.5)
# Model arch params
p.define("residual", False)
p.define("n_heads", [8, 1])
p.define("attn_hid_units", [16])
p.define("pool_hid_units", [64, 64, 128, 1024])
p.define("red_hid_units", [256, 64])


class Model(object):
    def __init__(self,
                 bn_decay=None):
        # --- I/O Tensors -----------------------------------------------------
        with TimeScope(MODEL_NAME + "/placeholder_setup", debug_only=True):
            self.bias_mat = tf.placeholder(tf.float32,
                                           (None,
                                            p.nodes_nb,
                                            p.nodes_nb),
                                           name="bias_mat")
            self.node_feats = tf.placeholder(tf.float32,
                                             (None,
                                              p.nodes_nb,
                                              p.feat_nb),
                                             name="node_feats")
            self.y = tf.placeholder(tf.float32,
                                    [None, p.num_classes],
                                    name="y")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.attn_drop = tf.placeholder(tf.float32, name="attn_drop_prob")
            self.feat_drop = tf.placeholder(tf.float32, name="feat_drop_prob")
            self.pool_drop = tf.placeholder(tf.float32, name="pool_drop_prob")
            self.bn_decay = bn_decay

        # --- Model properties ------------------------------------------------
        with TimeScope(MODEL_NAME + "/prop_setup", debug_only=True):
            self.inference
            self.loss
            # self.optimize

    def get_feed_dict(self, x_batch, y_batch, is_training):
        xb_node_feats = [np.array(x_i[0]) for x_i in x_batch]
        xb_bias_mat = [np.array(x_i[1]) for x_i in x_batch]

        attn_drop = p.attn_drop_prob if is_training else 0.
        feat_drop = p.feat_drop_prob if is_training else 0.
        pool_drop = p.pool_drop_prob if is_training else 0.

        return {
            self.node_feats: xb_node_feats,
            self.bias_mat: xb_bias_mat,
            self.y: y_batch,
            self.attn_drop: attn_drop,
            self.feat_drop: feat_drop,
            self.pool_drop: pool_drop,
            self.is_training: is_training
        }

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """
        # --- Features dim reduction ------------------------------------------
        feat_red_out = self.node_feats
        if p.red_hid_units != []:
            with tf.variable_scope('feat_dim_red'):
                feat_red = tf.layers.conv1d(self.node_feats,
                                            p.red_hid_units[0], 1,
                                            use_bias=False)
                if len(p.red_hid_units) > 1:
                    for i in range(1, len(p.red_hid_units)):
                        feat_red = tf.layers.conv1d(feat_red,
                                                    p.red_hid_units[i], 1,
                                                    use_bias=False)
                feat_red_out = feat_red
        # --- Graph attention layers ------------------------------------------
        with tf.variable_scope('graph_attn'):
            # attns = []
            # for _ in range(p.n_heads[0]):
            #     attns.append(attn_head(feat_red_out,
            #                            bias_mat=self.bias_mat,
            #                            out_sz=p.attn_hid_units[0],
            #                            activation=tf.nn.elu,
            #                            in_drop=self.feat_drop,
            #                            coef_drop=self.attn_drop,
            #                            residual=False))
            # h_1 = tf.concat(attns, axis=-1)
            h_1 = feat_red_out
            # for i in range(1, len(p.attn_hid_units)):
            for i in range(len(p.attn_hid_units)):
                attns = []
                for _ in range(p.n_heads[i]):
                    attns.append(attn_head(h_1,
                                           bias_mat=self.bias_mat,
                                           out_sz=p.attn_hid_units[i],
                                           activation=tf.nn.elu,
                                           in_drop=self.feat_drop,
                                           coef_drop=self.attn_drop,
                                           residual=p.residual))
                h_1 = tf.concat(attns, axis=-1)
            # out = []
            # for i in range(p.n_heads[-1]):
            #     out.append(attn_head(h_1,
            #                          bias_mat=self.bias_mat,
            #                          out_sz=p.num_classes,
            #                          activation=lambda x: x,
            #                          in_drop=self.feat_drop,
            #                          coef_drop=self.attn_drop,
            #                          residual=False))
            # logits = tf.add_n(out) / p.n_heads[-1]
            attn_out = h_1

        # --- Set Pooling -----------------------------------------------------
        with tf.variable_scope('graph_pool'):
            # g_pool = g_k(attn_out, 'g_' + str(0),
            #              filter_num=p.pool_hid_units[0],
            #              is_training=self.is_training,
            #              bn_decay=self.bn_decay,
            #              reg_constant=p.reg_constant)
            g_pool = attn_out
            for i in range(len(p.pool_hid_units)):
                g_pool = g_k(g_pool, 'g_' + str(i),
                             filter_num=p.pool_hid_units[i],
                             is_training=self.is_training,
                             bn_decay=self.bn_decay,
                             reg_constant=p.reg_constant)

            # Pooling of the pairs
            max_gg = tf.reduce_max(g_pool, axis=1, name='max_g')
            fcg = fc_bn(max_gg, 256,
                        scope='fcg',
                        is_training=self.is_training,
                        bn_decay=self.bn_decay,
                        reg_constant=p.reg_constant)

        # --- Classification --------------------------------------------------
        # fc_1 = fc_bn(fcg, 256, scope='fc_1',
        #              is_training=self.is_training,
        #              bn_decay=self.bn_decay)
        # fc_1d = tf.nn.dropout(fc_1, 1.0 - self.pool_drop)
        fc_2 = fc_bn(fcg, 128, scope='fc_2',
                     is_training=self.is_training,
                     bn_decay=self.bn_decay,
                     reg_constant=p.reg_constant)
        fc_2 = tf.nn.dropout(fc_2, 1.0 - self.pool_drop)

        return fc(fc_2, p.num_classes,
                  activation_fn=None, scope='logits')

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
