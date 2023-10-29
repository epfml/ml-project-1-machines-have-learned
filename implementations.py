import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e.
    Args:
        e: numpy array of shape (N,), N is the number of samples.
    Returns:
        scalar
    """
    return 1 / 2 * np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the mse loss.
    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N,D), D is the number of features
        w: numpy array of shape(D,)
    Returns:
        Scalar
    """
    return calculate_mse(y - tx.dot(w))


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w
    """
    N = y.shape[0]
    e = y - tx.dot(w)
    gradient = ((-1) / N) * ((tx.T).dot(e))

    return gradient, e


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        ws: model parameters as numpy arrays of shape (2, )
        losses: loss value (scalar)
    """
    w = initial_w
    losses = []
    losses.append(compute_loss(y, tx, w))
    for n_iter in range(max_iters):
        gradient, e = compute_gradient(y, tx, w)
        w = w - (gamma * gradient)
        losses.append(compute_loss(y, tx, w))

    return w, losses[-1]


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w
    """
    N = y.shape[0]
    e = y - tx.dot(w)
    gradient = ((-1) / N) * ((tx.T).dot(e))

    return gradient, e


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        ws: the model parameters as numpy arrays of shape (2, )
        losses: the loss value (scalar)
    """
    w = initial_w
    losses = []
    losses.append(compute_loss(y, tx, w))
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, e = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            losses.append(compute_loss(y, tx, w))

    return w, losses[-1]


def least_squares(y, tx):
    """Calculate the least squares solution.
    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N,D), D is the number of features
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features
        mse: scalar
    """
    A = tx.T.dot(tx)
    B = tx.T.dot(y)
    w, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    mse = compute_loss(y, tx, w)

    return (w, mse)


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.
    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N,D), D is the number of features
        lambda_: scalar
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = tx.T.dot(tx) + aI
    B = tx.T.dot(y)
    w = np.linalg.solve(A, B)
    loss = compute_loss(y, tx, w)

    return w, loss


def sigmoid(t):
    """Apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """
    return 1.0 / (1 + np.exp(-t))


def compute_loss_log_reg(y, tx, w):
    """Computes the cost by negative log likelihood.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))

    return np.squeeze(-loss) * (1 / y.shape[0])


def compute_gradient_log_reg(y, tx, w):
    """Computes the gradient of loss.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a vector of shape (D, 1)
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])

    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Computes Logistic Regression using Gradient Descent Algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the stepsize
    Returns:
        ws: the model parameters as numpy arrays of shape (2, )
        losses: the loss value (scalar)
    """
    w = initial_w
    losses = []
    losses.append(compute_loss_log_reg(y, tx, w))
    for n_iter in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w)
        w = w - gamma * gradient
        losses.append(compute_loss_log_reg(y, tx, w))

    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Computes Regularized Logistic Regression using Gradient Descent Algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        lambda_: scalar
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the stepsize
    Returns:
        ws: the model parameters as numpy arrays of shape (2, )
        losses: the loss value (scalar)
    """

    w = initial_w
    losses = []
     losses.append(compute_loss_log_reg(y, tx, w))
    for n_iter in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        losses.append(compute_loss_log_reg(y, tx, w))

    return w, loss
