import copy

import tensorflow as tf


def init_list(init, dims):
    def empty_list(dims):
        if not dims:
            return None
        else:
            return [copy.deepcopy(empty_list(dims[1:])) for i in range(dims[0])]

    def fill_list(dims, l):
        if len(dims) == 1:
            for i in range(dims[0]):
                if callable(init):
                    l[i] = init()
                else:
                    l[i] = init
        else:
            for i in range(dims[0]):
                fill_list(dims[1:], l[i])

    l = empty_list(dims)
    fill_list(dims, l)

    return l


def matmul_br(a, b, transpose_a=False, transpose_b=False):
    """Broadcasting matmul.

    The shape of a must be a subset of b in the sense that for example b has shape (j, k, l, m) and a has shape
    (k, n, l) or (n, l) (or (j, k, n, l) but then you can use the regular matmul).

    Not all combinations are supported right now.
    """
    a_sh = a.shape.as_list()
    b_sh = b.shape.as_list()
    if len(b_sh) == 2 and len(a_sh) >= 2:
        # this is by far the easiest case and the only one where things are relatively computationally efficient
        # first we merge all dimensions except the last
        a_merged = tf.reshape(a, [-1, a_sh[-1]])
        # then we do the multiplication
        product = tf.matmul(a_merged, b, transpose_a=transpose_a, transpose_b=transpose_b)
        # finally we separate the dimensions again
        return tf.reshape(product, a_sh[0:-1] + [-1])

    if len(b_sh) == 3 and len(a.shape) == 2:
        perm_move_to_end = [1, 2, 0]
        shape_merged = [-1, b_sh[2] * b_sh[0]]
        shape_separated = [-1, b_sh[2], b_sh[0]]
        perm_move_to_front = [2, 0, 1]
    elif len(b_sh) == 4 and len(a.shape) == 2:
        perm_move_to_end = [2, 3, 0, 1]
        shape_merged = [-1, b_sh[3] * b_sh[0] * b_sh[1]]
        shape_separated = [-1, b_sh[3], b_sh[0], b_sh[1]]
        perm_move_to_front = [2, 3, 0, 1]
    elif len(b_sh) == 4 and len(a.shape) == 3:
        perm_move_to_end = [1, 2, 3, 0]
        shape_merged = [b_sh[1], -1, b_sh[3] * b_sh[0]]
        shape_separated = [b_sh[1], -1, b_sh[3], b_sh[0]]
        perm_move_to_front = [3, 0, 1, 2]
    else:
        raise ValueError("Combination of ranks not supported")
    # move the first dimension to the end and then merge it with the last dimension
    b_merged = tf.reshape(tf.transpose(b, perm_move_to_end), shape_merged)
    # do multiplication
    product = tf.matmul(a, b_merged, transpose_a=transpose_a, transpose_b=transpose_b)
    # separate out the last dimension into what it was before the merging, then move the dimension from the back to the
    # front again
    return tf.transpose(tf.reshape(product, shape_separated), perm_move_to_front)


def ceil_divide(dividend, divisor):
    return (dividend + divisor - 1) / divisor


def log_cholesky_det2(chol):
    return 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(chol)), axis=-1)


def log_cholesky_det(chol):
    return 2 * tf.reduce_sum(tf.log(tf.diag_part(chol)))


def diag_mul2(mat1, mat2):
    """
    """
    # TODO(thomas): this seems wrong but nowhere it says what the function is supposed to do!!! so I don't know
    return tf.reduce_sum(mat1 * tf.matrix_transpose(mat2), -1)


def diag_mul(mat1, mat2):
    return tf.reduce_sum(mat1 * tf.transpose(mat2), 1)


def logsumexp(vals, dim=None):
    m = tf.reduce_max(vals, dim)
    if dim is None:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - m), dim))
    else:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - tf.expand_dims(m, dim)), dim))


def mat_square(mat):
    return tf.matmul(mat, mat, transpose_b=True)


def get_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                           'Must divide evenly into the dataset sizes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('n_epochs', 10000, 'Number of passes through the data')
    flags.DEFINE_integer('n_inducing', 240, 'Number of inducing points')
    flags.DEFINE_integer('display_step', 500, 'Display progress every FLAGS.display_step iterations')
    flags.DEFINE_integer('mc_train', 100, 'Number of Monte Carlo samples used to compute stochastic gradients')
    flags.DEFINE_integer('mc_test', 100, 'Number of Monte Carlo samples for predictions')
    flags.DEFINE_string('optimizer', "adagrad", 'Optimizer')
    flags.DEFINE_boolean('is_ard', True, 'Using ARD kernel or isotropic')
    flags.DEFINE_float('lengthscale', 10, 'Initial lengthscale')
    flags.DEFINE_integer('var_steps', 50, 'Number of times spent optimizing the variational objective.')
    flags.DEFINE_integer('loocv_steps', 50, 'Number of times spent optimizing the LOOCV objective.')
    flags.DEFINE_float('opt_growth', 0.0, 'Percentage to grow the number of each optimizations.')
    flags.DEFINE_integer('num_components', 1, 'Number of mixture components on posterior')
    flags.DEFINE_string('kernel', 'rbf', 'kernel')
    flags.DEFINE_string('device_name', 'gpu0', 'Device name')
    flags.DEFINE_integer('kernel_degree', 0, 'Degree of arccosine kernel')
    flags.DEFINE_integer('kernel_depth', 1, 'Depth of arcosine kernel')
    flags.DEFINE_boolean('hyper_with_elbo', True, 'Optimize hyperparameters with elbo as well')
    flags.DEFINE_boolean('normalize_features', False, 'Normalizes features')
    flags.DEFINE_boolean('optimize_inducing', True, 'Optimize inducing inputs')
    flags.DEFINE_float('latent_noise', 0.001, 'latent noise for Kernel matrices')
    return FLAGS
