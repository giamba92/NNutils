# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import csv
import os
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.optimizers import SGD, Nadam, Adam, RMSprop
import numpy as np
from six.moves import range, cPickle
from IPython import embed
from my_act_funcs import *
import matplotlib
matplotlib.use('Agg')
from utils_lts import *
import matplotlib.pyplot as plt

#-------------
# Helpers
#-------------

class CharacterTable(object):
    '''
    Taken from Keras examples.
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'



def operator_func(a, b, operator):
    if operator=='sum':
        return a + b
    elif operator=='dif':
        return a - b
    elif operator=='mul':
        return a * b
    elif operator == 'mem':
        return str(a)+str(b)


def pad_ans(ans):
    return ans + ' ' * (MAXLEN_Y - len(ans))


def MAXLEN_Y_calc(operator):
    if operator=='sum':
        return DIGITS + 1
    elif operator=='dif':
        return DIGITS + 1
    elif operator=='mul':
        return DIGITS * 2
    elif operator == 'mem':
        return DIGITS * 2

#-------------
# Main
#-------------

seed = 123456
np.random.seed(seed)  # for reproducibility
print("seed:", seed)

# define a sublist of activation functions, taken from my_act_funcs
relu, tanh, sigmoid, linear = 'relu', 'tanh', 'sigmoid', 'linear'
activs = [linear, sigmoid, tanh, my_relu, sin, cos, sin_mul_x, sin_x_relu, sin_div_x, clipped_sin_x]
parameteric_activs = [mix_sin_lin, mix_sin_relu, sin_learnable_freq]

# read parameters from a CSV file

found_line = False
line_idx = int(sys.argv[1])
with open('recipes_addition_RNN.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0].split(';')[0]==str(line_idx):
            l = row[0].split(';')
            print(l)
            act_idx_enc, act_idx_dec = int(l[1]), int(l[1])
            RNN = recurrent.LSTM if l[2]=='lstm' else recurrent.SimpleRNN
            OPERATOR = l[3]
            DIGITS = int(l[4])
            LAYERS = int(l[5])
            curriculum_learning = True if l[6]=='TRUE' else False
            found_line = True
            break
        else:
            continue
    if not found_line:
        print('line not found =(')
        sys.exit(1)

# More parameters

TRAIN_SYSTEM    = True
NEW_SUM         = True    # sample addends uniformly, instead of number of digits uniformly
TRAINING_SIZE   = 1024 * 16
INVERT          = True
HIDDEN_SIZE     = 64
BATCH_SIZE      = 128
MAXLEN          = DIGITS + 1 + DIGITS
MAXLEN_Y        = MAXLEN_Y_calc(OPERATOR)

DPT_RATE        = 0.0
NB_EPOCHS       = 5000
LR_DECAY        = 0.00001

ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=LR_DECAY, clipnorm=1)
optimiz = ADAM

# collect the activation functions
activ_enc, activ_dec = activs[act_idx_enc], activs[act_idx_dec]
activ_str = [x.__name__ if type(x) is not str else x for x in activs+parameteric_activs]

if activ_enc not in parameteric_activs: activ_layer_enc = None
else: activ_layer_enc = activ_enc

if activ_dec not in parameteric_activs: activ_layer_dec = None
else: activ_layer_dec = activ_dec


# define operators digit sets
if OPERATOR == 'mul':
    OP_SYMBOL = '*'
    chars = '0123456789* '
elif OPERATOR == 'sum':
    OP_SYMBOL = '+'
    chars = '0123456789+ '
elif OPERATOR == 'dif':
    OP_SYMBOL = '-'
    chars = '0123456789- '
elif OPERATOR == 'mix':
    OP_SYMBOL = '-'
    chars = '0123456789+- '
elif OPERATOR == 'mem':
    OP_SYMBOL = ''
    chars = '0123456789 '
else:
    print("unknown operator :/")
    sys.exit(0)


ctable = CharacterTable(chars, MAXLEN)

print('Enc act fun:', activ_str[act_idx_enc], ' - Dec act fun:', activ_str[act_idx_dec])

print('Build model...')
def make_model(activ_enc=activs[act_idx_enc], activ_dec=activs[act_idx_dec]):
    model = Sequential()

    # encoder
    model.add(RNN(HIDDEN_SIZE, activation=activ_enc, dropout_U=DPT_RATE, dropout_W=0, input_shape=(None, len(chars))))

    # decoder
    model.add(RepeatVector(MAXLEN_Y))
    model.add(RNN(HIDDEN_SIZE, activation=activ_dec, dropout_U=DPT_RATE, dropout_W=DPT_RATE, return_sequences=True, input_shape=(None, len(chars))))
    for _ in range(LAYERS-1):
        model.add(RNN(HIDDEN_SIZE, activation=activ_dec, dropout_U=DPT_RATE, dropout_W=DPT_RATE, return_sequences=True,))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars))))
    model.add(Activation('softmax'))
    return model

model = make_model(activ_enc=activs[act_idx_enc], activ_dec=activs[act_idx_dec])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=optimiz,
              metrics=['accuracy'])

fig_path = 'addition_rnn/'
RNN_STR = str(RNN).split("'")[-2].split(".")[-1]

title = ['NEW' if NEW_SUM else '',
         'CURRL' if curriculum_learning else '',
         OPERATOR, RNN_STR, 'DIG', DIGITS, 'LR_DECAY', LR_DECAY,
         'NB_EPCS', NB_EPOCHS, 'DataSiz', TRAINING_SIZE,
         'encAct', activ_str[act_idx_enc], 'decAct', activ_str[act_idx_dec],
         'HidSiz', HIDDEN_SIZE, 'Drpt', DPT_RATE,
         'L', LAYERS, 'Invert', INVERT,
         str(optimiz).split(' ')[0].split('.')[-1]]

title = '_'.join([str(x) for x in title if x is not ''])

fig_path_folder = fig_path + title + '/'

# check if path existed, or create it anew
if not os.path.exists(fig_path_folder):
    print('creating the folder: ', fig_path_folder)
    os.makedirs(fig_path_folder)
    path_existed = False
else:
    path_existed = True

# save the model architecture and weights
file_name = fig_path_folder + title + ".h5"
json_string = model.to_json()
open(file_name + '.json', 'w').write(json_string)
print("WILL SAVE MODEL TO", file_name)
model.save_weights(file_name, overwrite=True)

val_loss, train_loss, val_acc_list, get_activ_funcs = [], [], [], None

# function that generates samples. Partially taken from Keras examples
def generate_data(DIGITS):
    questions = []
    expected = []
    seen = set()
    while len(questions) < TRAINING_SIZE:
        if not NEW_SUM:
            f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
        else:
            f = lambda: np.random.randint(1, 10**(DIGITS)-1)
        a, b = f(), f()

        # Pad the data with spaces such that it is always MAXLEN
        q = '{}{}{}'.format(a, OP_SYMBOL, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(operator_func(a, b, OPERATOR))
        ans = pad_ans(ans, OPERATOR)
        if INVERT:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)

    X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), MAXLEN_Y, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        X[i] = ctable.encode(sentence, maxlen=MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, maxlen=MAXLEN_Y)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y

#----------
# Training phase
#----------

if TRAIN_SYSTEM and not path_existed:

    if curriculum_learning:
        DIGITS_data = 8
    else:
        DIGITS_data = DIGITS

    for iteration in range(1, NB_EPOCHS):

        if curriculum_learning and not iteration % (NB_EPOCHS/5):
            DIGITS_data = min(DIGITS_data+2, DIGITS)
            print("Increasing number of digits to", DIGITS_data)

        X_train, y_train = generate_data(DIGITS_data)

        # Select 10 samples from the validation set at random so we can visualize errors. From Keras examples
        for i in range(10):
            ind = np.random.randint(0, len(X_train))
            rowX, rowy = X_train[np.array([ind])], y_train[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            q = ctable.decode(rowX[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if INVERT else q)
            print('T', correct)
            print(guess, colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close)
            print('---')

        preds_train = model.predict_classes(X_train, batch_size=512)
        t = np.argmax(y_train, axis=2)
        val_acc = np.all(preds_train == t, axis=1).sum() / float(preds_train.shape[0])
        val_acc_list.append(val_acc)
        print('\nAcc: ', val_acc)
        print()
        print('-' * 50)
        print('Iteration', iteration)

        p = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2)
        train_loss.append(p.history.get('loss'))

        if iteration % 5 == 0:
            model.save_weights(file_name, overwrite=True)

            plt.figure(1, figsize=(16.0, 10.0))
            plt.clf()
            plt.plot(np.hstack(train_loss), linewidth=2, label='train_loss')
            plt.plot(np.hstack(val_acc_list), linewidth=2, label='valid_acc_val')
            plt.ylim([0, 1])
            plt.grid()
            plt.legend(loc='upper left', prop={'size': 7})
            plt.title(title)
            plt.savefig(fig_path + title+'.png')

            loss_dictionary = {'loss': np.hstack(train_loss), 'acc': np.hstack(val_acc_list)}
            f = open(file_name + '.save', 'wb')
            cPickle.dump(loss_dictionary, f)
            f.close()

    plt.savefig(fig_path_folder + title + '.png')
    # generate new data unseen to training
    X_train, y_train = generate_data(DIGITS_data)

else:
    X_train, y_train = generate_data(DIGITS)
    file_name = fig_path_folder + title + ".h5"
    model.load_weights(file_name)

    preds_t = model.predict_classes(X_train, batch_size=512)
    t = np.argmax(y_train, axis=2)
    val_acc = np.all(preds_t == t, axis=1).sum() / float(preds_t.shape[0])
    embed()

# save loss and accuracy
f = open(file_name + '.save', 'rb')
loss_dictionary = cPickle.load(f)
f.close()

print('activs = ', activ_str)

# computes the accuracy using the same model but different activation functions
def model_with_other_act(model, act_enc_clip, act_dec_clip):
    e = model.get_weights()
    model2 = make_model(activ_enc=activs[act_enc_clip], activ_dec=activs[act_dec_clip])
    model2.compile(loss='categorical_crossentropy',
                  optimizer=optimiz,
                  metrics=['accuracy'])
    model2.set_weights(e)
    score2 = model2.evaluate(X_train, y_train, batch_size=128, verbose=0)
    p2 = model2.predict_classes(X_train, batch_size=128, verbose=0)
    t2 = np.argmax(y_train, axis=2)
    val_acc2 = np.all(p2 == t2, axis=1).sum() / float(p2.shape[0])
    print('Previous score', val_acc)
    print('New score with', activs[act_enc_clip], '- - -', activs[act_dec_clip], val_acc2)
    return model2

# model with tanh
model_with_other_act(model, act_enc_clip=2, act_dec_clip=2)
model_with_other_act(model, act_enc_clip=2, act_dec_clip=act_idx_dec)
model_with_other_act(model, act_enc_clip=act_idx_enc, act_dec_clip=2)

# model with sin
model_with_other_act(model, act_enc_clip=4, act_dec_clip=4)
model_with_other_act(model, act_enc_clip=4, act_dec_clip=act_idx_dec)
model_with_other_act(model, act_enc_clip=act_idx_enc, act_dec_clip=4)

# model with clipped sin
model_with_other_act(model, act_enc_clip=9, act_dec_clip=9)
model_with_other_act(model, act_enc_clip=9, act_dec_clip=act_idx_dec)
model_with_other_act(model, act_enc_clip=act_idx_enc, act_dec_clip=9)

#-----------
# Function to compute the figures for all experiments
#-----------

def paper_all_figures_plot():

    fig_path = 'addition_rnn/'
    from matplotlib import style

    style.use(plt.style.available[3])
    loss_or_acc = 'acc' # acc loss
    NEW_SUM = True # ''
    curriculum_learning = False
    for OPERATOR in ['sum', 'dif']:
        for DIGITS in [8, 16]:
            if DIGITS == 8: LAYERS=1
            else: LAYERS=3
            for RNN_STR in ['SimpleRNN', 'LSTM']:
                RNN_STR_plot = RNN_STR if RNN_STR=='LSTM' else 'RNN'

                act_idx_enc, act_idx_dec = 4, 4

                title = ['NEW' if NEW_SUM else '',
                         'CURRL' if curriculum_learning else '',
                         OPERATOR, RNN_STR, 'DIG', DIGITS, 'LR_DECAY', LR_DECAY,
                         'NB_EPCS', NB_EPOCHS, 'DataSiz', TRAINING_SIZE,
                         'encAct', activ_str[act_idx_enc], 'decAct', activ_str[act_idx_dec],
                         'HidSiz', HIDDEN_SIZE, 'Drpt', DPT_RATE,
                         'L', LAYERS, 'Invert', INVERT,
                         str(optimiz).split(' ')[0].split('.')[-1]]

                title = '_'.join([str(x) for x in title if x is not ''])


                fig_path_folder = fig_path + title + '/'
                file_name = fig_path_folder + title + ".h5"
                import glob
                file_name = glob.glob(file_name)[0]

                f = open(file_name + '.save', 'rb')
                loss_dictionary = cPickle.load(f)
                f.close()

                act_idx_enc, act_idx_dec = 2, 2

                title = ['NEW' if NEW_SUM else '',
                         'CURRL' if curriculum_learning else '',
                         OPERATOR, RNN_STR, 'DIG', DIGITS, 'LR_DECAY', LR_DECAY,
                         'NB_EPCS', NB_EPOCHS, 'DataSiz', TRAINING_SIZE,
                         'encAct', activ_str[act_idx_enc], 'decAct', activ_str[act_idx_dec],
                         'HidSiz', HIDDEN_SIZE, 'Drpt', DPT_RATE,
                         'L', LAYERS, 'Invert', INVERT,
                         str(optimiz).split(' ')[0].split('.')[-1]]

                title = '_'.join([str(x) for x in title if x is not ''])


                fig_path_folder = fig_path + title + '/'
                file_name = fig_path_folder + title + ".h5"
                import glob
                file_name = glob.glob(file_name)[0]

                f = open(file_name + '.save', 'rb')
                loss_dictionary_tanh = cPickle.load(f)
                f.close()

                plt.figure(76, figsize=(6, 3.5))
                plt.plot(np.arange(len(loss_dictionary[loss_or_acc]))[::10], loss_dictionary[loss_or_acc][::10], label='sin ' + RNN_STR_plot, c='r', linewidth=1, linestyle='dashed' if RNN_STR=='LSTM' else '-')
                plt.plot(np.arange(len(loss_dictionary_tanh[loss_or_acc]))[::10], loss_dictionary_tanh[loss_or_acc][::10], label='tanh ' + RNN_STR_plot, c='b', linewidth=1, linestyle='dashed' if RNN_STR=='LSTM' else '-')
                plt.title('ENC-DEC on '+ OPERATOR + ' with ' + str(DIGITS) +' digits')
                plt.xlabel('iterations')
                if loss_or_acc=='acc':
                    plt.ylabel('accuracy')
                    plt.ylim([-0.01, 1.0])
                else:
                    plt.ylabel('loss')
                    plt.yscale('log')
                plt.legend(loc='lower right', prop={'size': 12})

            plt.tight_layout()
            plt.savefig('Paper_stuff/'+str(OPERATOR)+'_'+str(DIGITS)+'_NewSum_'+str(NEW_SUM)+
                        '_curr_'+str(curriculum_learning)+'_'+str(loss_or_acc)+'_H'+str(HIDDEN_SIZE) + '.pdf')

            plt.clf()
