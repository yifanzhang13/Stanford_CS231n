from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
        0.000005

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    # W.shape = (3073,10)
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # 10个class
    num_train = X.shape[0] # 500, batch of image data
    loss = 0.0
    for i in range(num_train):
        # image data X的第i行点乘Weight
        scores = X[i].dot(W) # (1,3073) dot (3073,10) = (1,10)
        correct_class_score = scores[y[i]] # y[i]是正确label的index
        for j in range(num_classes):
            if j == y[i]:
                continue # 如果当前label index是正确的，则直接下了一轮
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0: # 相当于max(0,score[j]-correct_class_score+1)
                loss += margin
                # gradient
                dW[:,j] += X[i] # 导数计算，在知乎笔记上可以找到公式
                dW[:,y[i]] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W) # 得到每个图片在每个class的score, shape=(500,10)
    '''
    range(num_train)代表scores的数组长度，list(y)代表每个正确score的index
    所以scores[range(num_train), list(y)]可以从scores中将正确的分数提取出来
    并通过reshape将shape改成(500,1)
    '''
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(500, 1)
    # 通过boardcasting，将每个分数与正确分数相减，并用max来得出输出
    margins = np.maximum(0, scores - correct_class_scores +1)
    # 在上一步中正确分数的情况是：max(0,correct_score-correct_score+1)=1，所以要将1化为0
    # 因为我们在计算loss的时候，正确标签是不用带入计算的，但是在举证中又不能为空，所以填0
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

    dW = (X.T).dot(coeff_mat)
    dW = dW/num_train + reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
