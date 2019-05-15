import numpy as np
import sparse_autoencoder
import scipy.optimize
import display_network
import softmax
import tensorflow as tf
import setting_classifier

RATE = 0.7  # train : validation = 7 : 3
RHO = 0.001  # h layer active probability
BETA = 0.05  # sparsity penalty parameter
LAMBDA = 0.9  # weight decay parameter
BATCH_SIZE = 100
EPOCH = 1000
ATTR = 121  # number of attribute
H = 3 * ATTR  # number of hidden layer unit
LEARNING_RATE = 0.001

OUTPUT = "result/sae6_1tho_h363_mac.csv"
INPUT = "kdd_train_1000.csv"
INPUT1 = "kdd_train_20Percent.csv"
PARAM_DIR = "train/model2.ckpt"


## ======================================================================
#  STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

input_size = ATTR
num_labels = 5
hidden_size = 3 * ATTR

sparsity_param = 0.1  # desired average activation of the hidden units.
lambda_ = 3e-3  # weight decay parameter
beta = 3  # weight of sparsity penalty term

## ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

# images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
# labels = load_MNIST.load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')


def data_set(bs_start, bs_last, ind, all_d, data_array):
    for count in range(bs_start, bs_last):
            data_array.append(all_d[ind[count]])

all_data, label_list = [], []
setting_classifier.data_read(all_data, INPUT, label_list)
LEN_ALL_DATA = len(all_data)
N_train = int(LEN_ALL_DATA * RATE)
N_val = LEN_ALL_DATA - N_train

# randomize index from all_data
index = list(range(0, LEN_ALL_DATA))
train_index = list(range(0, N_train))
test_index = list(range(0, N_val))

# Input: x : 127
train_list, test_list = [], []
np.array(data_set(0, N_train, index, all_data, train_list))
np.array(data_set(N_train, LEN_ALL_DATA, index, all_data, test_list))

train_label, val_label = [], []
np.array(data_set(0, N_train, index, label_list, train_label))
np.array(data_set(N_train, LEN_ALL_DATA, index, label_list, val_label))

# -------------------------------------------------------

# unlabeled_index = np.argwhere(labels >= 5).flatten()
# labeled_index = np.argwhere(labels < 5).flatten()

# num_train = round(labeled_index.shape[0] / 2)
# train_index = labeled_index[0:num_train]
# test_index = labeled_index[num_train:]

unlabeled_data = all_data

train_data = train_list
train_labels = train_label

test_data = test_list
test_labels = val_label

print('# examples in unlabeled set: {0:d}\n'.format(len(unlabeled_data)))
print('# examples in supervised training set: {0:d}\n'.format(len(train_data)))
print('# examples in supervised testing set: {0:d}\n'.format(len(test_data)))

## ======================================================================
#  STEP 2: Train the sparse autoencoder
#  This trains the sparse autoencoder on the unlabeled training
#  images.

#  Randomly initialize the parameters
theta = sparse_autoencoder.initialize(hidden_size, input_size)

J = lambda x: sparse_autoencoder.sparse_autoencoder_cost(x, input_size, hidden_size,
                                                         lambda_, sparsity_param,
                                                         beta, unlabeled_data)

options_ = {'maxiter': 400, 'disp': True}
result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
opt_theta = result.x

print(result)

# Visualize the weights
W1 = opt_theta[0:hidden_size * input_size].reshape(hidden_size, input_size).transpose()
display_network.display_network(W1)

##======================================================================
## STEP 3: Extract Features from the Supervised Dataset
#
#  You need to complete the code in feedForwardAutoencoder.m so that the
#  following command will extract features from the data.

train_features = sparse_autoencoder.sparse_autoencoder(opt_theta, hidden_size,
                                                       input_size, train_data)

test_features = sparse_autoencoder.sparse_autoencoder(opt_theta, hidden_size,
                                                      input_size, test_data)

##======================================================================
## STEP 4: Train the softmax classifier

lambda_ = 1e-4
options_ = {'maxiter': 400, 'disp': True}

opt_theta, input_size, num_classes = softmax.softmax_train(hidden_size, num_labels,
                                                           lambda_, train_features,
                                                           train_labels, options_)

##======================================================================
## STEP 5: Testing

predictions = softmax.softmax_predict((opt_theta, input_size, num_classes), test_features)
print("Accuracy: {0:.2f}%".format(100 * np.sum(predictions == test_labels, dtype=np.float64) / test_labels.shape[0]))
