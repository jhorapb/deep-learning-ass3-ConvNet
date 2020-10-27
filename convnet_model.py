import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib as mb
from pathlib import Path
import time
import random
import math
from tqdm import tqdm

## Dimensionality of the one-hot encoding vector (length of the alphabet)
d_len = 0

## Length of the longest name in the dataset.
n_len = 0

## Total number of names in the dataset.
N = 0

## Number of class labels
K_classes = 0

def read_contents(data_fname):
    """
    Read all the names in input file and returns a tuple of two lists
    with the names and countries of origin respectivelly.
    Returns:
        all_names (list): list of all names in the file.
        ys (list): list of all countries (in numbers) associated with the names.
    """
    name_lines = open(data_fname).read().strip().split('\n')
    all_names = []
    ys_list = []
    
    for i, input_name in enumerate(name_lines):
        data = input_name.split(' ')
        if len(data) > 2:
            name = ' '.join(data[:-1])
        else:
            name = data[0]
        all_names.append(name.lower())
        ys_list.append(int(data[-1]) - 1)
    ys = np.array(ys_list)
    return all_names, ys

def build_validation_set(validation_filename, all_names):
    """
    Returns all names used for validation.
    """
    val_indexes = open(validation_filename).read().strip().split(' ')
    validation_names = []
    ys_list = []
    for index in val_indexes:
        data = all_names[int(index)-1].split(' ')        
        if len(data) > 2:
            name = ' '.join(data[:-1])
        else:
            name = data[0]
        all_names.append(name.lower())
        ys_list.append(int(data[-1]))
    ys = np.array(ys_list)
        
    return validation_names, ys_validation


def encode_names(all_names, chars_dict):
    """
    Encodes all input names in a matrix that contains the flatten (vectorized)
    one-hot encoding of the inputs.
    d_len: length of the alphabet.
    n_len: length of the longest name in the dataset.
    
    Parameters:
        all_names (list): names read from the dataset.
        chars_dict (dict): letters in alphabet with corresponding index value.
    Returns:
        X (NumPy array): matrix of dimension d_len*n_len (flattened rows) x N (cols).
    """
    global d_len, n_len, N
    X = np.zeros((d_len, N, n_len))
    
    for i_name, name in enumerate(all_names):
        name_encoding = np.zeros((d_len, n_len))
        for i_letter, char in enumerate(name):
            name_encoding[chars_dict[char]][i_letter] = 1
        X[:, i_name, :] = name_encoding
    
    X = X.reshape(d_len*n_len, N)
    return X
    
def encode_labels_dataset(ys_labels):
    """
    Parameters:
        ys_labels (list): country-related labels associated with the input names.
    Returns:
        Y_one_hot (NumPy array): matrix of dimension K_classes (rows) x N names (cols) 
        containing the one-hot enconding of each label of the names.
    """
    global K_classes
    print(ys_labels)
    Y_one_hot = np.array(np.eye(K_classes)[ys_labels].T)
    return Y_one_hot

def plot_accuracy(acc_F1, acc_F2, acc_W, n_layer_1, n_layer_2):
    plt.plot([1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_F1), [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_F2),
             [1e-6, 1e-7, 1e-8, 1e-9], np.array(acc_W))
    plt.legend(['F1', 'F2', 'W'])
    plt.xlabel('Threshold level')
    plt.ylabel('Accuracy of the gradient')
    plt.title(
        'Accuracy in the derivative calculations with ' + str(n_layer_1) + ' filters in the first layer and ' + str(
            n_layer_2) + ' in the second layer')
    plt.show()


def plot_loss(cost, cost_val, acc, acc_val):
    fig1, axs1 = plt.subplots(1, 2)
    fig1.suptitle('Cost plot')
    axs1[0].plot(np.array(range(len(cost))) * 500, cost, np.array(range(len(cost))) * 500, cost_val)
    axs1[0].legend(['Cost in training set', 'Cost in validation set'])
    axs1[0].set_title('Cost values')
    axs1[0].set_xlabel('Number of steps')
    axs1[0].set_ylabel('Cost')

    axs1[1].plot(np.array(range(len(cost))) * 500, acc, np.array(range(len(cost))) * 500, acc_val)
    axs1[1].legend(['Accuracy in training set', 'Accuracy in validation set'])
    axs1[1].set_title('Accuracy values')
    axs1[1].set_xlabel('Number of steps')
    axs1[1].set_ylabel('Accuracy')
    fig1.show()
    fig1.savefig('cost.pdf')

class ConvNet():
    """
    Defines all the attributes and behaviour of the 
    Convolutional Neural Network.
    """
    def __init__(self, n_layer_filters, ks_width, d_len, K_classes, n_len, H_PARAMS, X_input, start, bias=False):
        """
        Initializes the CNN.
        
        Parameters:
            H_PARAMS: Hyper-parameters of the model.
            K: Number of classes.
            n_layer_filters: Number of filters in each layer of the NN.
            ks_width: Width of the filters applied at layer (i=1, 2, ...)
        """
        ## Add seed
        # np.random.seed(400)
        self.d_len = d_len
        self.n_len = n_len
        self.H_PARAMS = H_PARAMS
        self.K_classes = K_classes
        self.F = []
        self.n_layer_filters = []
        self.n_layers = 0 # Number of convolutional layers
        self.n_lengths = [] # The list of n_len values: [n_len, n_len1, n_len2]
        self.ks_width = []
        self.sig_vals = []
        self.MXinputs_global = []
        self.bias = bias
        
        ## Initializing CNN (n_lengths, weights, filters, matrices)
        self.initialize_structure(n_layer_filters, ks_width, X_input)
        
        self.build_convolution_matrices()
            
    def initialize_structure(self, n_layer_filters, ks_width, X_input):
        self.F = []
        self.n_layer_filters = n_layer_filters
        self.n_layers = len(self.n_layer_filters) # Number of convolutional layers
        self.n_lengths = [self.n_len] # The list of n_len values: [n_len, n_len1, n_len2]
        self.ks_width = ks_width
        self.sig_vals = []
        self.MXinputs_global = []
        
        ## Initializing self.n_lengths so we can initialize the Weigths (self.W).
        self.initialize_n_lengths()
        
        ## Initializing filters
        self.initialize_filters()
        
        ## Initializing weights
        f_size = self.n_layer_filters[-1] * self.n_lengths[-1]  # Formula f_size = n_2 * n_len2
        self.initialize_weights((self.K_classes, f_size))
        
        ## Initializing bias
        if self.bias:
            self.initialize_bias()

        self.start = start

        self.xin_global = X_input[0].reshape((-1, 1), order = 'F')
        for i in range(len(X_input) - 1):
            self.xin_global = np.concatenate([self.xin_global, X_input[i + 1].reshape((-1, 1), order = 'F')], axis = 1)

        for k in range(len(X_input)):
            self.MXinputs_global.append(self.MXMatrix_Efficient(X_input[k], self.d_len, self.ks_width[0]).copy())

        self.MXinputs = self.MXinputs_global

    def initialize_n_lengths(self):
        """
        Applies the formula n_len(i) = n_len(i-1) - k(i) + 1
        """
        for i in range(1, self.n_layers + 1):
            nl = self.n_lengths[i-1] - self.ks_width[i] + 1
            self.n_lengths.append(nl)
    
    def initialize_filters(self):
        """
        Creates the filters for each convolutional layer in the form:
        F1 = { F1i, ..., F(1,n1) }
        F2 = { F2i, ..., F(2,n2) }
        n1 and n2 here correspond to the number of filters in each layer.
        """
        for it in range(self.n_layers):
            if it == 0:
                self.sig_vals.append(2 / 7.1405)
                self.F.append(np.random.normal(loc = 0.0, scale = np.sqrt(sig_vals[it]), \
                    size = (self.d_len, self.ks_width[it], self.n_layer_filters[it])))
            else:
                self.sig_vals.append(2 / (self.n_lengths[it] * self.n_layer_filters[it - 1]))
                self.F.append(np.random.normal(loc = 0.0, scale = np.sqrt(sig_vals[it]), \
                    size = (self.n_layer_filters[it-1], self.ks_width[it], self.n_layer_filters[it])))
    
    def initialize_weights(self, size):
        self.sig_vals.append(2 / (self.n_lengths[-1] * self.n_layer_filters[-1]))
        self.W = np.random.normal(loc = 0.0, scale = np.sqrt(self.sig_vals[-1]), 
                                  size = (size))
    
    def initialize_bias(self):        
        self.b_vals = []
        self.bW = np.random.normal(loc = 0.0, scale = np.sqrt(self.sig_vals[-1]), 
                                   size = (self.K_classes, 1))
        for it in range(self.n_layers):
            self.b_vals.append(
                np.random.normal(loc = 0.0, scale = np.sqrt(self.sig_vals[it]), 
                                 size = (self.n_lengths[it + 1] * self.n_layer_filters[it], 1)))
    
    def set_H_PARAMS(self, H_PARAMS):
        self.H_PARAMS = H_PARAMS
    
    def build_convolution_matrices(self):
        self.build_MF_matrices()
        self.build_MX_matrices()
    
    def build_MF_matrices(self):
        """
        Returns the filter matrices, based on the entries in 
        the convolutional filter, to perform all the 
        convolutions at a given layer.
        
        Returns:
            Matrix MF of size (nlen-k+1)*nf × nlen*dd
        """
        if state == 'acc':
            self.xin = self.xin_global
        else:
            if isinstance(X, list):
                self.xin = X[0].reshape((-1, 1), order = 'F')
                for i in range(len(X) - 1):
                    self.xin = np.concatenate([self.xin, X[i + 1].reshape((-1, 1), order = 'F')], axis = 1)
            else:
                self.xin = X.reshape((-1, 1), order = 'F')

        self.VFlStored = []

        for k in range(self.n_layers):
            self.VFl = self.F[k].reshape((1, -1), order = 'F')
            self.VFl = self.VFl.reshape((self.n_layer_filters[k], -1))
            self.VFlStored.append(self.VFl)

        ## First filtering matrix
        self.MFMatrixStored = []
        for l in range(self.n_layers):
            self.MFMatrixStored.append(np.zeros(((
                self.nlen[l] - self.F[l].shape[1] + 1) * self.n_layer_filters[l], 
                                                 self.n_lengths[l] * self.F[l].shape[0])))
            for i in range(self.n_lengths[l] - self.F[l].shape[1] + 1):
                self.MFMatrixStored[l][i * self.n_layer_filters[l]: i * \
                                       self.n_layer_filters[l] + self.n_layer_filters[l], 
                                       i * self.F[l].shape[0] : i * self.F[l].shape[0] + \
                                           self.F[l].shape[0]*self.F[l].shape[1]] = self.VFlStored[l]
    
    def build_MX_matrices_efficient(self, x_input, d, k):
        nlenaid = x_input.shape[1]
        self.MX_efficient = np.zeros((nlenaid - k + 1, k*d))
        for i in range(nlenaid - k + 1):
            self.MX_efficient[i] = x_input[:, i : i + k].reshape((1, -1), order = 'F')
        return self.MX_efficient
    
    def build_MX_matrices(self, x_input, d, k, nf):
        self.MX1 = np.array([])
        for i in range(d - k + 1):
            if self.MX1.shape[0] == 0:
                test = x_input[:, i:i + k]
                self.MX1 = np.kron(np.identity(nf), x_input[:, i:i + k].reshape((1, -1), order = 'F'))
            else:
                test = x_input[:, i:i + k]
                aid = np.kron(np.identity(nf), x_input[:, i:i + k].reshape((1, -1), order = 'F'))
                self.MX1 = np.concatenate([self.MX1, aid], axis = 0)

        return self.MX1
    
    def relu_activation(self, input_data):
        output = np.maximum(input_data, 0, input_data)
        # indx = np.where(input_data < 0)
        # output = input_data.copy()
        # output[indx] = 0
        return output
    
    def evaluate_classifier(self, X, W, b = 0):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3])
        out = np.matmul(W, X) + b
        soft = np.exp(out) / (sum(np.exp(out)))
        return soft

    def compute_loss(self, X, lab, state = None):
        self.forward_pass(X, state)
        return sum(sum(lab.T * (-np.log(self.softmax)))) / self.xin.shape[1]
    
    def compute_accuracy(self, X, Y, state = None):
        self.forward_pass(X, state)
        max_perc = np.argmax(self.softmax, axis = 0)
        max_lab = np.argmax(Y, axis = 1)
        error = 0
        for i in range(max_perc.shape[0]):
            if max_perc[i] != max_lab[i]:
                error = error + 1
        return 1 - error / len(X)
    
    def forward_pass(self, X, state = None):
        self.build_MF_matrices(self.nlen[0], X, state)

        self.conv = []
        self.RelU_conv = []

        if self.bias == False:
            for k in range(self.n_layers):
                if k == 0:
                    self.conv.append(self.MFMatrixStored[k] @ self.xin)
                else:
                    self.conv.append(self.MFMatrixStored[k] @ self.RelU_conv[k - 1])

                self.RelU_conv.append(self.relu_activation(self.conv[k]))

            self.softmax = self.evaluate_classifier(self.RelU_conv[-1], self.W)

        else:
            for k in range(self.n_layers):
                if k == 0:
                    self.conv.append(self.MFMatrixStored[k] @ self.xin)
                    self.conv[k] = self.conv[k] + self.bFs[k]
                else:
                    self.conv.append(self.MFMatrixStored[k] @ self.RelU_conv[k - 1])
                    self.conv[k] = self.conv[k] + self.bFs[k]

                self.RelU_conv.append(self.relu_activation(self.conv[k]))

            self.softmax = self.evaluate_classifier(self.RelU_conv[-1], self.W, self.bW)
    
    def backward_pass(self, Y, alfa = 0):
        Gbatch = -(Y.T - self.softmax)
        self.dJw = (Gbatch @ self.RelU_conv[-1].T) / self.xin.shape[1]
        self.dLFstore = []
        self.dLF = []
        for k in range(self.n_layers):
            if k == 0:
                Gbatch = self.W.T @ Gbatch
            else:
                Gbatch = self.MFMatrixStored[-k].T @ Gbatch
            Gbatch = Gbatch * (self.RelU_conv[-1 - k] > 0)

            for i in range(Gbatch.shape[1]):
                #gj = Gbatch[:, i].reshape(-1, 1)
                gj = Gbatch[:, i].reshape(-1, 1)

                if k != self.n_layers - 1:
                    gj = gj.reshape(-1, self.F[-1 - k].shape[2],)
                    MxLayer = self.MXMatrix_Efficient( self.RelU_conv[-2 - k][:, i].reshape((self.n_layer_filters[-2 - k], -1), order = 'F'),
                        self.n_layer_filters[-k - 2], self.F[-1 - k].shape[1])
                else:
                    MxLayer = self.MXinputs[alfa + i]
                    gj = gj.reshape(-1, self.F[-1 - k].shape[2],)

                if len(self.dLF) == 0:
                    self.dLF = (MxLayer.T @ gj).reshape((1, -1), order = 'F')
                else:
                    self.dLF = self.dLF + (MxLayer.T @ gj).reshape((1, -1), order = 'F')

            self.dLF = self.dLF / Gbatch.shape[1]
            self.dLFstore.insert(0, self.dLF)
            self.dLF = []

        self.deriv = self.dLFstore
        self.deriv.append(self.dJw)

    def numerical_gradient(self, X_inputs, Ys, h):

        Fs = [self.F[k].copy() for k in range(len(self.F))]
        grads_F = [np.zeros(self.F[k].shape) for k in range(len(self.F))]

        for k in range(len(Fs)):
            for i in range(Fs[k].shape[0]):
                for j in range(Fs[k].shape[1]):
                    for l in range(Fs[k].shape[2]):
                        F_try = Fs[k].copy()
                        F_try[i, j, l] = F_try[i, j, l] - h
                        self.F[k] = F_try
                        l1 = self.compute_loss(X_inputs, Ys)
                        F_try[i, j, l] = F_try[i, j, l] + 2 * h
                        self.F[k] = F_try
                        l2 = self.compute_loss(X_inputs, Ys)
                        self.F[k] = Fs[k].copy()
                        grads_F[k][i, j, l] = (l2 - l1) / (2 * h)

        # compute the gradient for the fully connected layer
        W_try = self.W.copy()
        grads_W = np.zeros(W_try.shape)
        for i in range(W_try.shape[0]):
            for j in range(W_try.shape[1]):
                W_try1 = W_try.copy()
                W_try1[i, j] = W_try[i, j] - h
                self.W = W_try1
                l1 = self.compute_loss(X_inputs, Ys)
                self.W[i, j] = self.W[i, j] + 2 * h
                l2 = self.compute_loss(X_inputs, Ys)
                grads_W[i, j] = (l2 - l1) / (2 * h)
                self.W = W_try.copy()

        return grads_F, grads_W
    
    def relative_error(self, grad_num, grad_anal, eps):
        err = np.abs(grad_num - grad_anal) / np.maximum(eps * np.ones(grad_num.shape),
                                                        np.abs(grad_num) + np.abs(grad_anal))
        return err
    
    def grad_checking(self, X, Y, e, h):
        acc = [1e-6, 1e-7, 1e-8, 1e-9]
        per_F1 = []
        per_F2 = []
        per_F3 = []
        per_W = []

        self.forward(X)
        self.back_prop(Y)
        dLF_anal = []
        for i in range(self.n_layers):
            dLF_anal.append(self.dLFstore[i].copy())
        dJw_anal = self.dJw.copy()

        GradF, GradW = self.NumericalGradient(X, Y, h)

        for err in acc:
            err_F1 = self.relative_error(dLF_anal[0], GradF[0].reshape((dLF_anal[0].shape), order = 'F'), e)
            err_F2 = self.relative_error(dLF_anal[1], GradF[1].reshape((dLF_anal[1].shape), order = 'F'), e)
            err_W = self.relative_error(dJw_anal, GradW, e)
            per_F1.append(np.sum(err_F1 <= err) / (self.F[0].shape[0] * self.F[0].shape[1] * self.F[0].shape[2]))
            per_F2.append(np.sum(err_F2 <= err) / (self.F[1].shape[0] * self.F[1].shape[1] * self.F[1].shape[2]))
            per_W.append(np.sum(err_W <= err) / (self.W.shape[0] * self.W.shape[1]))

        return per_F1, per_F2, per_W
                
    def build_confussion_matrix(self, Y):
        self.confussion_matrix = np.zeros((self.K_classes, self.K_classes))
        max_perc = np.argmax(self.softmax, axis = 0)
        max_lab = np.argmax(Y, axis = 1)
        for i in range(max_perc.shape[0]):
            if max_perc[i] != max_lab[i]:
                self.confussion_matrix[max_lab[i], max_perc[i]] += 1
    
    def training(self, X, Y, Xval, Yval):
        self.Momentum = [0 for i in range(self.n_layers + 1)]
        accuracy_train = [self.compute_accuracy(X, Y, 'acc')]
        accuracy_val = [self.compute_accuracy(Xval, Yval)]
        costs_train = [self.compute_loss(X, Y, 'acc')]
        costs_val = [self.compute_loss(Xval, Yval)]
        print('--- Accuracy computed (1st set) ---')

        t = 0
        f = 0
        for i in range(self.H_PARAMS['n_epoch']):
            indx = []
            for k in range(len(self.start)-1):
                diff = self.start[k + 1] - self.start[k]
                aid = np.random.permutation(diff)
                aid = aid + self.start[k]
                indx.extend(aid[:50])
                
            X_epoch = [X[v] for v in indx]
            Y_epoch = Y[indx]

            self.MXinputs = [self.MXinputs_glob[l] for l in indx]

            for j in range(len(X_epoch) // self.H_PARAMS['n_batch']):
                j_start = j * self.H_PARAMS['n_batch']
                j_end = (j + 1) * self.H_PARAMS['n_batch']
                Xbatch = X_epoch[j_start:j_end]
                Ybatch = Y_epoch[j_start:j_end]

                self.forward(Xbatch)
                self.back_prop(Ybatch, j_start)

                for m in range(len(self.Momentum)):
                    self.Momentum[m] = self.Momentum[m] * self.H_PARAMS['rho'] + self.deriv[m] * self.H_PARAMS['eta']

                for n in range(self.n_layers):
                    aid = self.Momentum[n].reshape((1, -1))
                    self.F[n] = self.F[n] - aid.reshape((self.F[n].shape), order = 'F')

                self.W = self.W - self.Momentum[-1]

                t = t + 1

                if t % 500 == 0:
                    accuracy_train.append(self.compute_accuracy(X, Y, 'acc'))
                    costs_train.append(self.compute_loss(X, Y, 'acc'))
                    accuracy_val.append(self.compute_compute_accuracy(Xval, Yval))
                    costs_val.append(self.compute_loss(Xval, Yval))
                    print('Accuracy computed:', f)
                    f = f + 1
                    t = 0

        accuracy_train.append(self.compute_accuracy(X, Y, 'acc'))
        costs_train.append(self.compute_loss(X, Y, 'acc'))
        accuracy_val.append(self.compute_accuracy(Xval, Yval))
        costs_val.append(self.compute_loss(Xval, Yval))
        self.build_confussion_matrix(Yval)

        return accuracy_train, accuracy_val, costs_train, costs_val, self.confussion_matrix
    
    
if __name__ == "__main__":
    
    ## Dataset filenames
    # data_fname = 'single_ascii_names.txt'
    # data_fname = '2_ascii_names.txt'
    data_fname = 'ascii_names.txt'
    # validation_filename = '2_Validation_Inds.txt'
    validation_filename = 'Validation_Inds.txt'
    
    ## Read contents of the text file with names
    all_names, ys = read_contents(data_fname)
    validation_names, ys_val = build_validation_set(validation_filename, all_names)
    
    # print('---Validation Names (Total: {0})---\n{1}'.format(len(validation_names), validation_names))
    longest_name = max(all_names, key=len)
    n_len = len(longest_name)
    N = len(all_names)
    K_classes = len(np.unique(ys))
    
    print('\n---All Names---\nMax. Length: {0} ({1})\nTotal: {2}'\
          .format(n_len, longest_name, N))
    # print('\n---All Names---\n{0}\nMax. Length: {1} ({2})\nTotal: {3}'\
    #       .format(all_names, n_len, longest_name, N))
    # print('\n---Num. of Classes: {0}---\nOrigins\n{1}'.format(K_classes, ys))
    
    ## Characters used for the names in all_names
    C = sorted({char for word in all_names for char in word})
    d_len = len(C)
    
    chars_dict = {char: index for index, char in enumerate(C)}
    print('\n---Chars Used---\n{0}\nTotal:{1}'.format(C, d_len))
    print('\n---Chars Dictionary Map---\n', chars_dict)
    
    ## Encode all input names and save them as the input X
    X = encode_names(all_names, chars_dict)
    ## Encode all input names and save them as the input X (Validation)
    X_val = encode_names(validation_names, chars_dict)
    ## Enconde all the labels associated with the names as Y_one_hot
    Y_one_hot = encode_labels_dataset(ys)
    ## Enconde all the labels associated with the names as Y_one_hot (Validation)
    Y_val_one_hot = encode_labels_dataset(ys_val)
    print('\n---Full X Matrix Shape---\n', X.shape)
    # print('\n---Full X Matrix---\n', X)
    print('\n---Y One-hot Labels Shape---\n', Y_one_hot.shape)
    # print('\n---Y One-hot Labels---\n', Y_one_hot)
    testing_names = ['perez', 'isachenko', 'moncada', 'adkins', 'fematt']
    testing_labels = [17, 15, 17, 5, 6]
    X_test = encode_names(testing_names, chars_dict)
    Y_test_one_hot = encode_labels_dataset(testing_labels)
    
    H_PARAMS = {'n_epoch': 2000, 'n_batch': 100, 'eta': 0.001, 'rho': 0.9}
    ## Setup 1:
    # conv_nn = ConvNet([5, 4, 3], [5, 3, 3], d_len, K_classes, n_len, H_PARAMS, X[:2], start)
    ## Setup 2:
    conv_nn = ConvNet([5, 3], [5, 5], d_len, K_classes, n_len, H_PARAMS, X[:2], start)

    conv_nn.forward_pass(X[:2])
    conv_nn.backward_pass(Y_one_hot[:2])
    
    accuracy, accuracy_val, cost_train, cost_val, matrix = conv_nn.training(
        X, Y_one_hot, X_val, Y_val)
    
    accuracy_test = conv_nn.compute_accuracy(X_val, Y_val_one_hot)
    conv_nn.build_confussion_matrix(Y_val_one_hot)
    test_mtx = conv_nn.confussion_matrix
    plot_loss(cost_train, cost_val, accuracy, accuracy_val)

    print("\n Test matrix \n")
    print(test_mtx)
    print("\n Accuracy in the test sample: ", acc_test)

    per_F1, per_F2, per_F3, per_W = conv_nn.grad_checking(ohd_names[:2], Y_ohd[:2], eps, h)
    plot_accuracy(per_F1, per_F2, per_F3, per_W, conv_nn.n_layer_filters[0], conv_nn.n_layer_filters[1])