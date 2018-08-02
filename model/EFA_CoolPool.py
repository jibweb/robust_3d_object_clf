import numpy as np
import tensorflow as tf
from utils.logger import TimeScope
from utils.tf import fc, fc_bn, define_scope
from utils.params import params as p

from layers import mh_neigh_edge_attn, avg_graph_pool, conv1d_bn, conv3d

MODEL_NAME = "EFA_CoolPool"

# Dropout prob params
p.define("attn_drop_prob", 0.0)
p.define("feat_drop_prob", 0.0)
p.define("pool_drop_prob", 0.5)
# Model arch params
p.define("residual", False)
p.define("graph_hid_units", [16])
p.define("attn_head_nb", [16])
p.define("gcn_dist_thresh", [0])
p.define("red_hid_units", [256, 64])
p.define("graph_pool", [0])


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
            self.edge_feats = tf.placeholder(tf.float32,
                                             (None,
                                              p.nodes_nb,
                                              p.nodes_nb,
                                              p.edge_feat_nb),
                                             name="edge_feats")
            self.valid_pts = tf.placeholder(tf.float32,
                                            [None,
                                             p.nodes_nb,
                                             p.nodes_nb],
                                            name="valid_pts")
            if p.feats_3d:
                self.node_feats = tf.placeholder(tf.float32,
                                                 (None,
                                                  p.nodes_nb,
                                                  p.feat_nb,
                                                  p.feat_nb,
                                                  p.feat_nb, 1),
                                                 name="node_feats")
            else:
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

            self.adj_mask = self.bias_mat < -1.

        # --- Model properties ------------------------------------------------
        with TimeScope(MODEL_NAME + "/prop_setup", debug_only=True):
            self.inference
            self.loss
            # self.optimize

    def get_feed_dict(self, x_batch, y_batch, is_training):
        xb_node_feats = [np.array(x_i[0]) for x_i in x_batch]
        xb_bias_mat = [np.array(x_i[1]) for x_i in x_batch]
        xb_edge_feats = [np.array(x_i[2]) for x_i in x_batch]
        xb_valid_pts = [np.diag(x_i[3]) for x_i in x_batch]

        feat_drop = p.feat_drop_prob if is_training else 0.
        pool_drop = p.pool_drop_prob if is_training else 0.

        return {
            self.node_feats: xb_node_feats,
            self.bias_mat: xb_bias_mat,
            self.edge_feats: xb_edge_feats,
            self.valid_pts: xb_valid_pts,
            self.y: y_batch,
            self.feat_drop: feat_drop,
            self.pool_drop: pool_drop,
            self.is_training: is_training
        }

    @define_scope
    def inference(self):
        """ This is the forward calculation from x to y """
        # --- Features dim reduction ------------------------------------------
        feat_red_out = self.node_feats
        with tf.variable_scope('feat_dim_red'):
            if p.feats_3d:
                feat_red_out = tf.reshape(feat_red_out, [-1, 4, 4, 4, 1])
                feat_red_out = conv3d(feat_red_out,
                                      scope="dimred_3d_1",
                                      out_sz=2,
                                      kernel_sz=2,
                                      reg_constant=p.reg_constant)
                feat_red_out = conv3d(feat_red_out,
                                      scope="dimred_3d_2",
                                      out_sz=4,
                                      kernel_sz=2,
                                      reg_constant=p.reg_constant)
                feat_red_out = tf.reshape(feat_red_out, [-1, p.nodes_nb,
                                                         2**3 * 4])
                feat_red_out = conv1d_bn(feat_red_out,
                                         scope="dimred_flat_1",
                                         out_sz=32,
                                         reg_constant=p.reg_constant,
                                         is_training=self.is_training)
                feat_red_out = conv1d_bn(feat_red_out,
                                         scope="dimred_flat_2",
                                         out_sz=32,
                                         reg_constant=p.reg_constant,
                                         is_training=self.is_training)

            else:
                for i in range(len(p.red_hid_units)):
                    feat_red_out = conv1d_bn(feat_red_out,
                                             scope="dimred_" + str(i),
                                             out_sz=p.red_hid_units[i],
                                             reg_constant=p.reg_constant,
                                             is_training=self.is_training)

        # --- Graph attention layers ------------------------------------------
        with tf.variable_scope('graph_layers'):
            # Pre setup
            feat_gcn = feat_red_out
            edge_feats = self.edge_feats
            adj_mask = self.adj_mask

            # Apply all convolutions
            for i in range(len(p.graph_hid_units)):
                feat_gcn = mh_neigh_edge_attn(
                    feat_gcn, p.graph_hid_units[i], p.gcn_dist_thresh[i],
                    edge_feats, p.attn_head_nb[i], adj_mask, tf.nn.elu,
                    p.reg_constant, self.is_training, self.bn_decay,
                    "attn_heads_" + str(i), in_drop=0.0, coef_drop=0.0,
                    residual=False, use_bias_mat=True)

                if p.graph_pool[i]:
                    with tf.variable_scope("graph_pool_" + str(i)):
                        feat_gcn, edge_feats, adj_mask = avg_graph_pool(
                            feat_gcn, edge_feats,
                            4, adj_mask,
                            p.gcn_dist_thresh[i])

            gcn_out = feat_gcn

        # --- Set Pooling -----------------------------------------------------
        with tf.variable_scope('graph_pool'):
            valid_pts = self.valid_pts
            for i in range(len(p.graph_pool)):
                if p.graph_pool[i]:
                    valid_pts = valid_pts[:, ::4, ::4]

            gcn_filt = tf.matmul(valid_pts, gcn_out)
            max_gg = tf.reduce_max(gcn_filt, axis=1, name='max_g')
            fcg = fc_bn(max_gg, p.graph_hid_units[-1]*p.attn_head_nb[-1],
                        scope='fcg',
                        is_training=self.is_training,
                        bn_decay=self.bn_decay,
                        reg_constant=p.reg_constant)

        # --- Classification --------------------------------------------------
        with tf.variable_scope('classification'):
            fc_2 = fc_bn(fcg, 64, scope='fc_2',
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
