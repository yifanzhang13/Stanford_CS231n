from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
      (3073,10)
    - X: A numpy array of shape (N, D) containing a minibatch of data.
      (500,3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
      (500,)
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    scores_exp = X.dot(W)
    # 这里给e的scroe次方需要减每一行中最大的数值是为了不让指数过高导致blowup，可以看softmax笔记中有写
    scores_exp = np.exp(scores_exp - np.max(scores_exp, axis=1, keepdims=True))
    # cross-entropy loss 交叉商损失
    loss = np.sum(-np.log(scores_exp[range(N), y]/np.sum(scores_exp, axis=1)))/N
    # regularization 正则化
    loss += reg * np.sum(W**2)

    # 导数公式：https://zhuanlan.zhihu.com/p/30748903
    for i in range(N):
      dW[:,y[i]] -= X[i] # 先求-syi这部分导数
      for j in range(C):
        dW[:,j] += X[i] * scores_exp[i,j] / np.sum(scores_exp[i])
    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    scores = X.dot(W)
    # 这里给e的scroe次方需要减每一行中最大的数值是为了不让指数过高导致blowup，可以看softmax笔记中有写
    scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    # cross-entropy loss 交叉商损失
    loss = np.sum(-np.log(scores[range(N), y]/np.sum(scores, axis=1)))/N
    ds = scores / np.sum(scores, axis=1, keepdims=True)
    ds[range(N), y] -= 1
    dW = X.T.dot(ds)

    dW /= N
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
