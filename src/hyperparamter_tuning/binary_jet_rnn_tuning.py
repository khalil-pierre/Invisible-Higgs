# -*- coding: utf-8 -*-
"""
This file contains hyperparameter tunning work for the rnn binary classifier.
"""

# ---------------------------------- Imports ----------------------------------

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# Code from other files in the repo
import classifier
from utilities.data_analysis import ModelResultsMulti

# Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIR = parentdir + '\\data_binary_classifier\\'
args_model = {'model_type' : 'binary_classifier',
              'model_architecture' : 'RNN',
              'layer_1_neurons' : 16,
              'layer_2_neurons' : 8,
              'output_shape':1,
              'learning_rate' : 0.001,
              'batch_size' : 64,
              'epochs' : 8,
              'model' : 'base'}

num_runs = 3
dataset_sample = 0.25

test_param = 'layer_1_neurons'
param_values = np.linspace(4, 64, num=7, dtype=np.int)

all_results = ModelResultsMulti()
jet_rnn = classifier.JetRNN(args_model)
jet_rnn.load_data(DIR)
jet_rnn.load_jet_data(DIR)

for parameter in param_values:
    print(f'{test_param} = {parameter}')
    args_model[test_param] = parameter
    jet_rnn.args_model = args_model
    
    for i in range(num_runs):
        model_result = classifier.run(i, jet_rnn, args_model, dataset_sample)
        all_results.add_result(model_result, {test_param : parameter})

df_all_results = all_results.return_results()

# ------------------------------- Results plots -------------------------------

all_data = []
for parameter in param_values:
    test = df_all_results[df_all_results[test_param]==parameter]
    all_data.append(test['accuracy_test'].values)
    
fig, ax = plt.subplots(figsize=(6,4), dpi=200)
bplot = ax.boxplot(all_data, labels=param_values, 
                   showfliers=False, showmeans=True)

ax.yaxis.grid(True)
ax.set_xlabel('Number of neurons')
ax.set_ylabel('Model accuracy')
