import tensorflow as tf
import numpy as np


def _merge_and_separate(a, b, func):
    """
    Helper function to make operations broadcast when they don't support it natively.

    The shape of `a` must be a subset of `b` in the sense that for example `b` has shape (j, k, l, m) and `a` has shape
    (k, n, l) or (n, l) (for (j, k, n, l) you can just use the regular operation). Also supported is `b` with shape
    (j, k, l) and `a` with shape (n, k).

    Args:
        a: Tensor
        b: Tensor
        func: a function that takes two arguments
    Returns:
        broadcasted result
    """
    a_rank = len(a.shape)
    b_rank = len(b.shape)
    if a_rank == b_rank:
        # no need to broadcast; just apply the function
        return func(a, b)

    b_sh = b.shape.as_list()
    if b_rank == 3 and a_rank == 2:
        perm_move_to_end = [1, 2, 0]
        shape_merged = [-1, b_sh[2] * b_sh[0]]
        shape_separated = [-1, b_sh[2], b_sh[0]]
        perm_move_to_front = [2, 0, 1]
    elif b_rank == 4 and a_rank == 2:
        perm_move_to_end = [2, 3, 0, 1]
        shape_merged = [-1, b_sh[3] * b_sh[0] * b_sh[1]]
        shape_separated = [-1, b_sh[3], b_sh[0], b_sh[1]]
        perm_move_to_front = [2, 3, 0, 1]
    elif b_rank == 4 and a_rank == 3:
        perm_move_to_end = [1, 2, 3, 0]
        shape_merged = [b_sh[1], -1, b_sh[3] * b_sh[0]]
        shape_separated = [b_sh[1], -1, b_sh[3], b_sh[0]]
        perm_move_to_front = [3, 0, 1, 2]
    else:
        raise ValueError("Combination of ranks not supported")

    # move the first dimension to the end and then merge it with the last dimension
    b_merged = tf.reshape(tf.transpose(b, perm_move_to_end), shape_merged)
    # apply function
    result = func(a, b_merged)
    # separate out the last dimension into what it was before the merging, then move the dimension from the back to the
    # front again
    return tf.transpose(tf.reshape(result, shape_separated), perm_move_to_front)


def matmul_br(a, b, transpose_a=False, transpose_b=False):
    """Broadcasting matmul.

    Supported only up to 5 dimensional tensors.

    Args:
        a: Tensor
        b: Tensor
        transpose_a: whether or not to transpose a
        transpose_b: whether or not to transpose b
    Returns:
        Broadcasted result of matrix multiplication.
    """
    a_dim = len(a.shape)
    b_dim = len(b.shape)
    # the output always has indices that are the tail of 'ijklm'
    # the dimension to reduce is always 'x'
    # the a indices always end in 'l'
    # the b indices always end in 'm' and skip 'l'
    # easy example when both are 3D and not transposing: 'klx,kxm->klm'
    max_dim = max(a_dim, b_dim)
    if max_dim > 5 or min(a_dim, b_dim) < 2:
        raise ValueError("dimensions over 5 or under 2 are not supported")
    # the index prefix is at most 'ijk' for 5 dimensional tensors and at the least '' (empty) for 2 dimensional tensors
    a_index_prefix = 'ijk'[5-a_dim:]
    b_index_prefix = 'ijk'[5-b_dim:]

    # the last two dimensions are always there. they depend on whether the tensor is transposed or  not
    a_index_suffix = 'xl' if transpose_a else 'lx'
    b_index_suffix = 'mx' if transpose_b else 'xm'

    out_indices = 'ijklm'[5-max_dim:]
    return tf.einsum(a_index_prefix + a_index_suffix + ',' + b_index_prefix + b_index_suffix + '->' + out_indices, a, b)


def cholesky_solve_br(chol, rhs):
    """Broadcasting Cholesky solve.

    This only works if `rhs` has higher rank.

    Args:
        chol: Cholesky factorization.
        rhs: Right-hand side of equation to solve.
    Returns:
        Solution
    """
    return _merge_and_separate(chol, rhs, tf.cholesky_solve)


def ceil_divide(dividend, divisor):
    return (dividend + divisor - 1) / divisor


def log_cholesky_det(chol):
    return 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(chol)), axis=-1)


def diag_mul(mat1, mat2):
    """
    """
    # TODO(thomas): this seems wrong but nowhere it says what the function is supposed to do!!! so I don't know
    return tf.reduce_sum(mat1 * tf.matrix_transpose(mat2), -1)


def mat_square(mat):
    return tf.matmul(mat, mat, transpose_b=True)


def broadcast(tensor, tensor_with_target_shape):
    """Make `tensor` have the same shape as `tensor_with_target_shape` by copying `tensor` over and over.

    The rank of `tensor` has to be smaller than the rank of `tensor_with_target_shape`.
    """
    target_shape = tensor_with_target_shape.shape.as_list()
    target_rank = len(target_shape)
    input_shape = tensor.shape.as_list()
    input_rank = len(input_shape)
    if all(input_shape) and all(target_shape):
        # the shapes are all fully specified. this means we can work with integers
        # first we will pad the shape with 1s until the rank is the same. e.g. [m, n] -> [1, 1, 1, m, n]
        expand_dims_shape = [1] * (target_rank - input_rank) + input_shape
        # then we set the multiples that are necessary to reach the target shape. e.g. [j, k, l, 1, 1]
        tile_multiples = target_shape[0:-input_rank] + [1] * input_rank
    else:
        # the shapes are not fully specified. we have to work with tensors
        target_shape = tf.shape(tensor_with_target_shape)
        expand_dims_shape = tf.concat([[1] * (target_rank - input_rank), tf.shape(tensor)], axis=0)
        tile_multiples = tf.concat([target_shape[0:-input_rank], [1] * input_rank], axis=0)
    input_with_expanded_dims = tf.reshape(tensor, expand_dims_shape)
    return tf.tile(input_with_expanded_dims, tile_multiples)


def tri_vec_shape(N):
    """
    Compute the length of a vector that contains all elements of a triangle with dimensions N.
    """
    return [N * (N + 1) // 2]


def vec_to_tri(vectors):
    """
    Takes a DxM tensor vectors and maps it to a D x matrix_size x matrix_size
    tensor where the lower triangle of each matrix_size x matrix_size matrix
    is constructed by unpacking each M-vector.
    """
    M = vectors.shape[1].value
    N = int(np.floor(0.5 * np.sqrt(M * 8. + 1) - 0.5))
    assert N * (N + 1) == 2 * M  # check M is a valid triangle number.
    indices = tf.constant(np.stack(np.tril_indices(N)).T, dtype=tf.int32)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)


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
