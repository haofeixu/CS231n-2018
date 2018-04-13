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
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
      	# Gradient computing: http://cs231n.github.io/optimization-1/#analytic
        dW[:, j] += X[i, :]
        dW[:, y[i]] += -X[i, :]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


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
  
  # Ref: https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html

  num_train = X.shape[0]
  scores = X.dot(W)
  # y_score' shape is (N,), y_score[i] is the i-th training data's score y_i
  y_scores = scores[np.arange(num_train), y]
  # np.maximum: element-wise max, let y_score be a column vector to do broadcasting
  margins = np.maximum(0, scores - y_scores[:, np.newaxis] + 1)
  # Don't count y_i
  margins[np.arange(num_train), y] = 0
  loss = np.mean(np.sum(margins, axis=1))
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # Code: https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
  # Excellent!

  binary = margins
  binary[margins > 0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T
  dW = np.dot(X.T, binary)

  dW /= num_train
  dW += 2 * reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
