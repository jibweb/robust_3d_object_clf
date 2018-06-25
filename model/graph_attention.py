import tensorflow as tf


# Code from github.com/PetarV-/GAT
def attn_head(seq, out_sz, bias_mat, activation,
              in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

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
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation
