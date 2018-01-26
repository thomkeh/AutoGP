import tensorflow as tf


def _merge_and_separate(a, b, func, transpose_b=False):
    """
    Helper function to make operations broadcast when they don't support it natively.

    The shape of a must be a subset of `b` in the sense that for example `b` has shape (j, k, l, m) and `a` has shape
    (k, n, l) or (n, l) (for (j, k, n, l) you can just use the regular operation). Also supported is `b` with shape
    (j, k, l) and `a` with shape (n, k).

    Args:
        a: Tensor
        b: Tensor
        func: a function that takes two arguments
        transpose_b: whether or not to transpose `b` before applying the function. if this option is used, func has to
            support it as well.
    Returns:
        broadcasted result
    """
    a_rank = len(a.shape)
    b_rank = len(b.shape)
    if a_rank == b_rank:
        # no need to broadcast; just apply the function
        if transpose_b:
            return func(a, b, transpose_b=True)
        else:
            return func(a, b)

    if transpose_b:
        # working with a transposed `b` is very easy if `a` is only 2-dimensional
        if b_rank >= 2 and a_rank == 2:
            b_sh = b.shape.as_list()
            # this is by far the easiest case and the only one where things are relatively computationally efficient
            # first we merge all dimensions except the last
            b_merged = tf.reshape(b, [-1, b_sh[-1]])
            # then we do the multiplication
            product = tf.matmul(a, b_merged, transpose_b=True)
            # finally we separate the dimensions again but we have to be careful because `b_sh` contains the not
            # transposed version. because of `transpose_b` b_sh[-2] should now be at the place of b_sh[-1]
            return tf.reshape(product, b_sh[0:-2] + [-1, b_sh[-2]])

        # otherwise we have to do it the hard way:
        b = tf.matrix_transpose(b)

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


def matmul_br(a, b, transpose_b=False):
    """Broadcasting matmul.

    Not all combinations of ranks are supported right now. It is also not supported to tranpose `a` (but this could be
    added).

    Args:
        a: Tensor
        b: Tensor
        transpose_b: whether or not to transpose b
    Returns:
        Broadcasted result of matrix multiplication.
    """
    a_sh = a.shape.as_list()
    if len(b.shape) == 2 and len(a_sh) >= 2:
        # this is by far the easiest case and the only one where things are relatively computationally efficient
        # first we merge all dimensions except the last
        a_merged = tf.reshape(a, [-1, a_sh[-1]])
        # then we do the multiplication
        product = tf.matmul(a_merged, b, transpose_b=transpose_b)
        # finally we separate the dimensions again
        return tf.reshape(product, a_sh[0:-1] + [-1])

    # if b has higher rank than a, we have to use a different approach
    return _merge_and_separate(a, b, tf.matmul, transpose_b=transpose_b)


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
