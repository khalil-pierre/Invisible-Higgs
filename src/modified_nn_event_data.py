"""
This file contains the run code for a simple feedforward neural network to 
classify different event types.
"""

# ---------------------------------- Imports ----------------------------------
#Stops numpy from trying to capture multiple cores on cluster nodes
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Code from other files in the repo
import models.sequential_models as sequential_models
import utilities.plotlib as plotlib

# Python libraries
from mpi4py import MPI
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from math import sqrt

# -------------------------------- Data setup --------------------------------

SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

# Load files
event_data = np.load(DIR+'preprocessed_event_data.npy', allow_pickle=True)
sample_weight = np.load(DIR+'preprocessed_sample_weights.npy', allow_pickle=True)
weight_nominal = np.load(DIR+'weight_nominal.npy', allow_pickle=True)
xs_weight = np.load(DIR+'xs_weight.npy', allow_pickle=True)
encoding_dict = pickle.load(open(DIR+'encoding_dict.pickle', 'rb'))
event_labels = pd.read_hdf(DIR+'preprocessed_event_labels.hdf')
event_labels = event_labels.values

test_fraction = 0.2
data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
    train_test_split(event_data, event_labels, 
                     sample_weight, test_size=test_fraction)

# Take a sample of the data to speed up training
sample_num = 5000
data_train = data_train[:sample_num]
data_test = data_test[:sample_num]
labels_train = labels_train[:sample_num]
labels_test = labels_test[:sample_num]
sw_train = sw_train[:sample_num]
sw_test = sw_test[:sample_num]

# ------------------------------ Model training -------------------------------

INPUT_SHAPE = event_data.shape[1]
model = sequential_models.base2(64, 8, input_shape=INPUT_SHAPE)

START = time.time()
history = model.fit(data_train, labels_train, batch_size = 128,
                    validation_data=(data_test, labels_test), 
                    sample_weight=sw_train, epochs=16, verbose=2)

test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=2)

# --------------------------------- Plotting ----------------------------------

comm = MPI.COMM_WORLD
taskid = comm.Get_rank()
numtasks = comm.Get_size()
MASTER = 0

accuracy = history.history['accuracy']
final_accuracy = accuracy[-1]

label_pred = model.predict(data_test)
fpr, tpr, _ = roc_curve(labels_test, label_pred)
roc_auc = auc(fpr, tpr)

all_accuracies = comm.gather(accuracy, root=0)
final_accuracies = comm.gather(final_accuracy, root=0)
auc_scores = comm.gather(roc_auc, root=0)

if taskid == MASTER:
    print('---------------all accuracies---------------')
    print(all_accuracies)
    print('---------------final accuracies---------------')
    print(final_accuracies)
    
    IMDIR = "\\users\\user\\documents\\fifth year\\invisible-higgs\\Images\\sequential_model"
    
    fig1,ax1 = plt.subplots()
    ax1.boxplot(final_accuracies)
    ax1.set_xlabel('Sequential model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Sequential model accuracy box plot')
    fig1.savefig(IMDIR + '\\accuracy_boxplot')
    
    fig2,ax2 = plt.subplots()
    ax2.boxplot(auc_scores)
    ax2.set_xlabel('Sequential model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Sequential model roc curve auc box plot')
    fig2.savefig(IMDIR + '\\auc_boxplot')
    
    fig3,ax3 = plt.subplots()
    ax3.boxplot(all_accuracies)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Sequential model training curve')
    fig3.savefig(IMDIR + '\\training_curve_boxplot')
    
    
    SAVEDIR = "\\users\\user\\documents\\fifth year\\invisible-higgs\\"
    
    np.save(SAVEDIR + '\\sequential_model_final_accuracies', final_accuracies)
    np.save(SAVEDIR + '\\sequential_model_accuracies', all_accuracies)
    np.save(SAVEDIR + '\\sequential_model_auc_scores', auc_scores)



    
    
    

