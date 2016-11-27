from __future__ import print_function
from __future__ import division
import numpy as np
import sys
from IPython import embed

#-------------------------------------------
# Training functions
#-------------------------------------------

# an object used to store training parameters
class train_params(object):
    def __init__(self, nb_epoch, batch_size=128, random_scale=False, arg_max=1, callbacks=[], 
    			 fig_comp=True, fig_show=False, fig_path=None, fig_title='', plt_ion=1,
                 threshold=0.5, verbose=2):
        self.epochs = nb_epoch
        self.batch_size = batch_size
        self.random_scale = random_scale
        self.callbacks = callbacks
        # Fig params
        self.fig_comp = fig_comp
        self.fig_show = fig_show
        self.fig_path = fig_path
        self.fig_title = fig_title
        self.arg_max = arg_max
        self.plt_ion = plt_ion
        self.threshold = threshold
        self.verbose = verbose


def train_model(model, train_params, train_data, val_data=(), data_obj=None):
    # dictionary of elements to return
    training_res = dict()
    
    # empty lists for losses
    val_loss, train_loss, val_acc = [], [], []

    # save net weights of each epoch
    if hasattr(train_params, 'save_all_weights'):
        training_res['model_weights_history'] = []

    X_train, Y_train = train_data[0], train_data[1]

    for e in range(train_params.epochs):
        if data_obj is not None:
        	# using a generator to train
            data_obj.train_data()
            X_train, Y_train = data_obj.X_train, data_obj.Y_train

        # random scaling factor
        if train_params.random_scale: scaling_factor = (0.3 + np.random.rand()*0.7)
        else: scaling_factor = 1

        print("* Epoch ", e, "/", train_params.epochs)

        p = model.fit(X_train * scaling_factor, Y_train, batch_size=train_params.batch_size, nb_epoch=1,
                      verbose=train_params.verbose, validation_data=(val_data), callbacks=train_params.callbacks)
        
        if hasattr(train_params, 'save_all_weights'):
            # save net weights of each epoch
            training_res['model_weights_history'].append(model.get_weights())

        if train_params.fig_comp:
        	# save and plot losses and accuracies
            val_loss.append(p.history.get('val_loss'))
            train_loss.append(p.history.get('loss'))
            val_acc.append(1 * [score_exact(model, val_data[0], val_data[1], train_params.threshold, train_params.arg_max)])
            plot_and_save(val_loss, train_loss, val_acc, train_params)

    training_res['val_loss'], training_res['train_loss'], training_res['val_acc'] = val_loss, train_loss, val_acc
    return training_res


def plot_and_save(val_loss, train_loss, val_acc_list, train_params):
	''' 
	plot and save losses and accuracies
	'''
    import matplotlib.pyplot as plt
    plt.figure(5, figsize=(16, 10))
    if train_params.plt_ion:
        plt.clf()
        plt.ion()
    plt.plot(np.hstack(val_loss), linewidth=2, label='val_loss')
    plt.plot(np.hstack(train_loss), linewidth=2, label='train_loss')
    plt.plot(np.hstack(val_acc_list), linewidth=2, label='valid_acc')
    plt.ylim([0, 1])
    plt.grid()
    plt.legend(loc='upper left', prop={'size': 7})
    plt.title(train_params.fig_title+' best val acc so far: '+str(max(val_acc_list)))
    if train_params.fig_show:
        plt.show()
    if train_params.fig_path is not None:
        plt.savefig(train_params.fig_path + train_params.fig_title + '.png')


#-------------------------------------------
# Functions to study network activations
#-------------------------------------------

def return_activated_layers_list(model):
	''' 
	returns a list of layers with meaningful activations to be studied
	'''
    my_activ_layers = np.array(['my_act_funcs' in str(x) for x in model.layers])
    activation_layers = np.array(['Act' in str(x) for x in model.layers])
    conv_layers = np.array(['Conv' in str(x) for x in model.layers])
    dense_layers = np.array(['Dense' in str(x) for x in model.layers])
    rnn_layers = np.array(['recur' in str(x) for x in model.layers])

    layers = activation_layers + conv_layers + dense_layers + my_activ_layers + rnn_layers
    layers = [i for i, x in enumerate(layers) if x]
    return layers


def check_activations(model, data, ptc=1, return_acts=False, get_activation_funcs=None, rnn_enc_complete_output=False):
    '''
    computes min and max for the activations of each layer, returns the activations and the functions used to compute them
    '''

    from keras import backend as K
    # from IPython import embed

    layers = return_activated_layers_list(model)

    # compute the backend functions to extract the activations, unless previously computed
    if get_activation_funcs is None:
        get_activation_funcs = {}
        for i in layers:
            print(i)
            get_activation_funcs[i] = K.function([model.layers[0].input, K.learning_phase()], model.layers[i].output)

    acts_list = []
    for lay_num in range(len(model.layers)):

    	# select the percentage of data to be used
        btch = data.shape[0]//ptc

        if lay_num in layers:

            if rnn_enc_complete_output and lay_num == 0:
            	# return outputs for each step of the rnn
                acts = []
                for ind in range(data.shape[1]):
                    acts.append(get_activation_funcs[lay_num]([data[:btch][:,:ind+1,:], 0]))
                acts = np.dstack(acts).swapaxes(1, 2)
            else:
                acts = get_activation_funcs[lay_num]([data[:btch], 0])

            if return_acts:
                acts_list.append(acts)

            print('Max is: ', '{0:.2f}'.format(acts.max()), ', min is:', '{0:.2f}'.format(acts.min()),
                    'Mean is: ', '{0:.2f}'.format(acts.mean()), ', Std is:', '{0:.2f}'.format(acts.std()), '  - Layer: ', model.layers[lay_num])
            # check if the activations exceed pi/2, needed for the paper "Taming the waves"
            if 'Dense' in str(model.layers[lay_num]) or 'Conv' in str(model.layers[lay_num]):
                print('* * * Ptc |x| > pi/2 is: ', '{0:.2f}'.format( (np.abs(acts.ravel()) > np.pi/2).mean() ),)

    return get_activation_funcs, acts_list


def hist_plot_activations(model, X, layer=0):
	'''
	plot histogram of the activations against the sinusoid function, needed for the paper "Taming the waves"
    '''

    layers = return_activated_layers_list(model)
    act_layer = layers.index(layer)
    import matplotlib.pyplot as plt
    plt.figure(32, figsize=(16.0,3.0))
    acts, get_activ_funcs = check_activations(model, X, return_acts=True)
    a = acts[act_layer].ravel()
    xx = np.arange(a.min()-3, a.max()+3, 0.01)
    n, bins, patches = plt.hist(a, 100, normed=1, facecolor='red', alpha=0.65)
    plt.plot(xx, n.max()*np.sin(xx))
    plt.axvspan(a.min()-3, -np.pi/2, facecolor='b', alpha=0.15)
    plt.axvspan(np.pi/2, a.max()+3, facecolor='b', alpha=0.15)
    plt.xlim([max(-7, a.min()-1),min(7, a.max()+1)])


def eval_network_between_configs(model, mod1_w, mod2_w, alpha=0.05, min_alpha=-1, max_alpha=2, train_data=None, test_data=None, mode='linear'):
    '''
    computes the accuracy of the models interpolated between two configurations (as done in the paper: [add ref])
    train data should be a tuple or list like (X, Y). Mode can be 'linear' or 'curved'.
    returns train_acc, test_acc, alphas
    '''

    backup_model_weights = model.get_weights()
    train_acc = []
    test_acc = []
    alphas = []
    if train_data[0].ndim == 3:
        score_func = score_rnn_encode_task
    else:
        score_func = score_exact

    a = min_alpha
    while a <= max_alpha:
        w_temp = []
        for w1, w2 in zip(mod1_w, mod2_w):
            if 'lin' in mode:
                w_temp.append( (1 - a) * w1 + a * w2 )
            elif 'cur' in mode:
                w_temp.append( np.sin(a*np.pi/2) * w1 + np.cos(a*np.pi/2) * w2 )
        model.set_weights(w_temp)
        if train_data is not None:
            train_acc.append(score_func(model, train_data[0], train_data[1]))
        if test_acc is not None:
            test_acc.append(score_func(model, test_data[0], test_data[1]))
        alphas.append(a)
        a += alpha
    model.set_weights(backup_model_weights)
    return train_acc, test_acc, alphas


#-------------------------------------------
# Generic utils 
#-------------------------------------------

def stack_3D_activ_to_list2D(A):
	'''
	converts 3D activations to a list of 2D activations 
	'''
    B = []
    for i in range(A.shape[1]):
        B.append(A[:,i,:])
    return B


def plot_all_elems_in_list(l):
	'''
	plots in separate plots all the elements in a list 
	'''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16.0, 10.0))
    # max number of columns in the figure
    max_cols = 5.0
    for ind, elem in enumerate(l):
        plt.subplot(max(1,np.ceil(len(l)/float(max_cols))), min(len(l),max_cols), ind+1)
        plt.imshow(elem, interpolation='nearest')


def min_max_list(data_list, axis):
    '''
    computes min and max of all matrices in a list
    '''
    import numpy as np
    min_data = np.min([np.min(elem[..., axis]) for elem in data_list])
    max_data = np.max([np.max(elem[..., axis]) for elem in data_list])
    return min_data, max_data


def generate_grid(data=None, x_min=None, x_max=None, y_min=None, y_max=None, spacing=0.02, extra_plt_margin=2):
    '''
    Returns a grid of points xx, yy either from the hull of the list of data or from the coordinates specified.
    '''
    import numpy as np

    # step size in the mesh
    spacing = spacing * (1 + extra_plt_margin)  
    if data is not None:
        if type(data) is list:
            x_min, x_max = min_max_list(data, axis=0)
            y_min, y_max = min_max_list(data, axis=1)
            x_min -= extra_plt_margin
            y_min -= extra_plt_margin
            x_max += extra_plt_margin
            y_max += extra_plt_margin
        else:
            print('Error in the function args')
            sys.exit(1)

    xx, yy = np.meshgrid(np.arange(x_min - .3, x_max + .3, spacing),
                         np.arange(y_min - .3, y_max + .3, spacing))

    return xx, yy


def specific_layers_idxs(model, list_of_names):
    '''
    return the idx of specific layers in a certain model
    '''
    # TODO implement with lists of names
    res = []
    for i in range(len(model.layers)):
        if list_of_names[0] in str(model.layers[i]):
            res.append(i)
    # sort might be needed
    return sorted(res)


def unif_init_func(low=-1, high=1):
    '''
    initialize between low and high. default values -1 and 1
    '''
    from keras import initializations
    return lambda shape: initializations.uniform(shape, low=low, high=high)


def sort_column_by_max(A):
    '''
    sorts the cols of a matrix A by the position of the max
    '''
    max_idx = np.argmax(A, axis=0)
    A_sorted = np.fliplr(A[:, np.argsort(max_idx)])
    return A_sorted


def binarize_array(a, digits):
	'''
	converts a 2D array of integers to a 3D binary array with the binary representation of each entry
	'''
    new_array = []
    for k in range(a.shape[0]):
        new_array.append((((a[k,:, None] & (1 << np.arange(digits)))) > 0).astype(int))
    return np.rollaxis(np.dstack(new_array), -1, 0)


def mk_dir(dir):
	'''
	creates a directory if it doesn't exist
	'''
    import os
    # check if the dir already exists
    try:
        os.mkdir( dir )
    except OSError:
        # dir already exists
        pass


#-------------------------------------------
# Scoring functions
#-------------------------------------------

def score_exact(classif, X, Y, threshold=0.5, arg_max=0):
	'''
	compute the accuracy for perfect recognition for a given classifier
	works with both multiclass and multilabel data
	'''
    O = classif.predict(X)
    if type(O) is list: O = O[0]
    if O.ndim == 3:
        O = O[:,-1,:]
    if arg_max:
        O = (O.T - O.max(axis=1)).T >= 0
    #if not np.all(np.any((w==0,w==1),axis=0)):
    O = (O > threshold)

    # if target values are not 0,1 but -1,1
    if np.min(Y)==-1:
        Y = (Y > threshold)

    diff = O - Y
    score = np.sum(diff * diff, axis=1)
    score = ((score.size)-np.count_nonzero(score))/(len(score)*1.0)
    return score


def score_rnn_encode_task(model, X, Y):
	'''
	scores encoder rnn task
	'''
    p = model.predict_classes(X)
    #p = model.predict(X)
    #p = np.argmax(p, axis=2)
    t = np.argmax(Y, axis=2)
    return np.all(p == t, axis=1).sum() / float(p.shape[0])


