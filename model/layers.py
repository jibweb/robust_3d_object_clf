import tensorflow as tf
from utils.tf import conv1d, conv2d, conv1d_bn, conv3d,\
                     weight_variable, bias_variable, batch_norm_for_conv1d,\
                     batch_norm_for_conv2d


def edge_attn(seq, out_sz, bias_mat, edge_feats, activation,
              reg_constant, is_training, bn_decay, scope,
              in_drop=0.0, coef_drop=0.0, residual=False, use_bias_mat=True):
    with tf.variable_scope(scope):
        # if in_drop != 0.0:
        #     seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        # seq_fts = conv1d(seq, out_sz, reg_constant, "feat_conv",
        #                  activation=None, use_bias=False)
        seq_fts = conv1d_bn(seq, out_sz, reg_constant, is_training,
                            "feat_conv", activation=None, use_bias=False)

        # edge feature attention
        logits = conv2d(edge_feats, 1, 1, reg_constant, "edge_feats_attn")
        # Reshaped to be the same size as bias_mat
        # The last dimension is 1, the nb of filters
        nodes_nb = bias_mat.get_shape()[-1].value
        logits = tf.reshape(logits, [-1, nodes_nb, nodes_nb])

        # bias_mat to reintroduce graph structure
        pre_coefs = tf.nn.leaky_relu(logits)
        if use_bias_mat:
            pre_coefs += bias_mat
        coefs = tf.nn.softmax(pre_coefs)

        vals = tf.matmul(coefs, seq_fts)
        biases = bias_variable([out_sz], reg_constant)
        ret = vals + biases

        # ret_bn = batch_norm_for_conv1d(ret,
        #                                is_training=is_training,
        #                                bn_decay=bn_decay,
        #                                scope='bn')

        return activation(ret)  # activation


def neighboring_edge_attn(seq, out_sz, dist_thresh, edge_feats, activation,
                          reg_constant, is_training, bn_decay, scope,
                          in_drop=0.0, coef_drop=0.0, residual=False,
                          use_bias_mat=True):
    with tf.variable_scope(scope):
        neigh_adj = tf.cast(edge_feats[:, :, :, 0] < dist_thresh, tf.float32)
        neigh_bias = -1e9 * (1.0 - neigh_adj)
        return edge_attn(seq, out_sz, neigh_bias, edge_feats, activation,
                         reg_constant, is_training, bn_decay, "edge_attn",
                         in_drop=in_drop, coef_drop=coef_drop,
                         residual=residual)


def mh_neigh_edge_attn(seq, out_sz, dist_thresh, edge_feats, head_nb, adj_mask,
                       activation, reg_constant, is_training, bn_decay,
                       scope, in_drop=0.0, coef_drop=0.0, residual=False,
                       use_bias_mat=True):
    with tf.variable_scope(scope):
        neigh_mask = edge_feats[:, :, :, 0] < dist_thresh
        neigh_mask = tf.logical_and(neigh_mask, adj_mask)
        neigh_bias = -1e9 * (1.0 - tf.cast(neigh_mask, tf.float32))

        gcn_heads = []
        # edge feature attention
        logits = conv2d(edge_feats, head_nb, 1, reg_constant, "edge_feat_attn")

        for head_idx in range(head_nb):
            with tf.variable_scope("head_" + str(head_idx)):
                seq_fts = conv1d_bn(seq, out_sz, reg_constant, is_training,
                                    "feat_conv", activation=None,
                                    use_bias=False)

                pre_coefs = tf.nn.leaky_relu(logits[:, :, :, head_idx])

                # bias_mat to reintroduce graph structure
                if use_bias_mat:
                    pre_coefs += neigh_bias
                coefs = tf.nn.softmax(pre_coefs)

                vals = tf.matmul(coefs, seq_fts)
                biases = bias_variable([out_sz], reg_constant)
                ret = vals + biases

                gcn_heads.append(activation(ret))

        return tf.concat(gcn_heads, axis=-1)


def avg_graph_pool(seq, edge_feats, kernel_sz, adj_mask, dist_thresh):
    dist_mask = edge_feats[:, :, :, 0] < dist_thresh
    dist_mask = tf.logical_and(dist_mask, adj_mask)
    dist_neigh = tf.cast(dist_mask, tf.float32)
    dist_neigh = tf.nn.softmax(dist_neigh)
    pooled_seq = tf.matmul(dist_neigh, seq)
    pooled_seq = seq[:, ::kernel_sz, :]
    pooled_edge_feats = edge_feats[:, ::kernel_sz, ::kernel_sz, :]
    pooled_adj_mask = adj_mask[:, ::kernel_sz, ::kernel_sz]

    return pooled_seq, pooled_edge_feats, pooled_adj_mask


def attn_head(seq, out_sz, bias_mat, activation,
              reg_constant, is_training, bn_decay, scope,
              in_drop=0.0, coef_drop=0.0, residual=False):
    """
    Layer originally from github.com/PetarV-/GAT
    """
    with tf.variable_scope(scope):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        seq_fts = conv1d(seq, out_sz, reg_constant, "feat_conv",
                         activation=None, use_bias=False)

        # simplest self-attention possible
        # a_1.(W.h_i)
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        # a_2.(W.h_j)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        # a_1.(W.h_i) + a_2.(W.h_j) = e_ij
        # /!\ tf broadcasting, result is [nb_nodes, nb_nodes]
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # a_ij = softmax(leakyReLU(e_ij))  bias_mat reintroduce graph structure
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        biases = bias_variable([out_sz], reg_constant)
        ret = vals + biases

        ret_bn = batch_norm_for_conv1d(ret,
                                       is_training=is_training,
                                       bn_decay=bn_decay,
                                       scope='bn')

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)
            else:
                seq_fts = ret + seq

        return activation(ret_bn)  # activation


def graph_conv(seq, out_sz, bias_mat, activation,
               reg_constant, is_training, bn_decay, scope,
               in_drop=0.0, residual=False):
    """
    Layer originally from github.com/PetarV-/GAT
    """
    with tf.variable_scope(scope):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        seq_fts = conv1d(seq, out_sz, reg_constant, "conv",
                         activation=None, use_bias=False)
        coefs = tf.nn.softmax(bias_mat)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)

        # ret = tf.contrib.layers.bias_add(vals)
        biases = bias_variable([out_sz], reg_constant)
        ret = vals + biases

        ret_bn = batch_norm_for_conv1d(ret,
                                       is_training=is_training,
                                       bn_decay=bn_decay,
                                       scope='bn')

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)
            else:
                seq_fts = ret + seq

        return activation(ret_bn)  # activation


def g_k(tens_in, scope, out_sz, is_training, bn_decay, reg_constant):
    with tf.variable_scope(scope):
        g_k = conv1d(tens_in, out_sz, reg_constant, "conv",
                     activation=None)
        with tf.variable_scope("max_var"):
            max_var = weight_variable([out_sz], reg_constant)
        g_k = g_k - tf.expand_dims(max_var*tf.reduce_max(g_k, axis=1,
                                                         name='max_g'),
                                   1)
        g_k_norm = batch_norm_for_conv1d(
                g_k,
                is_training=is_training,
                bn_decay=bn_decay,
                scope='bn')
        return tf.nn.relu(g_k_norm)


def g_2d_k(tens_in, scope, out_sz, is_training, bn_decay, reg_constant):
    with tf.variable_scope(scope):
        g_k = conv2d(tens_in, out_sz, 1, reg_constant, "conv",
                     activation=None)
        with tf.variable_scope("max_var"):
            max_var = weight_variable([out_sz], reg_constant)
            g_k = g_k - tf.expand_dims(max_var*tf.reduce_max(g_k, axis=2,
                                                             name='max_g'),
                                       2)
        g_k_norm = batch_norm_for_conv2d(
                        g_k,
                        is_training=is_training,
                        bn_decay=bn_decay,
                        scope='bn')
        return tf.nn.relu(g_k_norm)
