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
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    ## W.shape()=(3072,10), X.shape()=(49000,3072), X[i].shape()=(1,3072)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        ## 减去最大值防止数值溢出
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
    ###相当于：
    ### y = torch.tensor([2,0,1])    真实样本对应种类
    ### y_hat = torch.tensor([[0.1, 0.2, 0.7],[0.8, 0.1, 0.1], [0.3, 0.4, 0.3]])  样本的预测概率
    ### y_hat[[0,1,2],y]  每个样本在真实类别上的预测概率
    ### return - torch.log(y_hat[range(len(y_hat)), y])   每个样本在真实类别上的预测概率
    # normalized hinge loss plus regularization
        
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    p[y[i]]-=1
    for j in range (num_classes): #循环更新梯度
            dW[:,j] +=X[i] * p[j]
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train +2 * reg * W

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
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    #计算分数
    s = X @ W
    s -= np.max(s,axis=1,keepdims=True)
    #计算概率和softmax
    p = np.exp(s)
    p /=np.sum(p,axis=1,keepdims=True)
    #计算损失函数
    correct_p = np.log(p[np.arange(num_train),y])
    loss = -np.sum(correct_p)
    loss = loss / num_train +reg *np.sum(W*W)
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    
    #计算梯度
    y_one_hot = np.zeros_like(p)
    y_one_hot[np.arange(num_train),y]=1 
    dW = X.T @ (p - y_one_hot)
    dW = dW /num_train +2*reg*W

    return loss, dW
