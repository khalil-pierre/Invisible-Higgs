# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:06:54 2021

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker
import copy

# Python libraries
import os

args = {'dir_root' : 'C:\\Users\\user\\Documents\\Fifth Year\\ml_postproc',
        'input_dataset' : 'new',
        'output_datasets' : ['binary_classifier'],
        'chosen_output' : 'binary_classifier',
        'set_diJet_mass_nan_to_zero' : True,
        'weight_col' : 'xs_weight',
        'timestamp' : datetime.today().strftime('%Y-%m-%d %H:%M:%S')}

args['save_folder'] = 'data_' + args['chosen_output']
args['dir_output'] = args['save_folder'] + '\\'

args['data_to_collect'] = ['ttH125',
                           'TTTo2L2Nu',
                           'TTToHadronic',
                           'TToSemiLeptonic']
 

# -------------------------------- Load in data -------------------------------

loader = DataLoader(args['dir_root'])
loader.find_files()
loader.collect_data(args['data_to_collect'])
data = DataProcessing(loader)

#%%
non_normalised_weights = data.data.xs_weight

args['cols_to_ignore_events'] = ['entry', 'weight_nominal', 
                                 'xs_weight', 'hashed_filename', 
                                 'BiasedDPhi', 'InputMet_InputJet_mindPhi', 
                                 'InputMet_phi', 'InputMet_pt', 'MHT_phi']

cols_events = data.get_event_columns(args['cols_to_ignore_events'])
data.set_nan_to_zero('DiJet_mass')
# Removes all events with less than two jets
data.data = data.data[data.data.ncleanedJet > 1]

signal_list = ['ttH125']
data.label_signal_noise(signal_list)
event_labels, encoding_dict = LabelMaker.label_encoding(data.return_dataset_labels())
data.set_dataset_labels(event_labels, onehot=False)

signal_idx = data.data.dataset == 'signal'

sample_weight = WeightMaker.weight_nominal_sample_weights(data, weight_col=args['weight_col'])

df_event_data = copy.deepcopy(data)
# Select only the event columns from the data
df_event_data.filter_data(cols_events)
df_event_data = df_event_data.data

#%%

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

plt.rc('font', **font)

col = 'HT'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, axes = plt.subplots(ncols=3, figsize=(8,8))
ax1, ax2, ax3 = axes.flatten()

ax1.hist([df_event_data[signal_idx][col], df_event_data[~signal_idx][col]],
         bins=50, stacked=True)

ax1.set_xlabel(col + '(GeV)')
ax1.set_ylabel('Events')
ax1.set_xlim(0, 2000)

ax2.hist([df_event_data[signal_idx][col], df_event_data[~signal_idx][col]],
         bins=50, weights=[non_normalised_weights[signal_idx],
                           non_normalised_weights[~signal_idx]],stacked=True)

ax2.set_xlabel(col + '(GeV)')
ax2.set_ylabel('Expected number of events for 1 $pb^{-1}$ luminosity')
ax2.set_xlim(0, 2000)

ax3.hist([df_event_data[signal_idx][col], df_event_data[~signal_idx][col]],
         bins=50, weights=[sample_weight[signal_idx], sample_weight[~signal_idx]],
         stacked=True)

ax3.set_xlabel(col + '(GeV)')
ax3.set_ylabel('Normalised expected number of events')
ax3.set_xlim(0, 2000)

DIR = "C:\\Users\\user\\Documents\\Fifth Year\\Report\\"
plt.legend(labels = ['ttH', 'tt¯'])
fig.tight_layout()
fig.savefig(DIR + 'event_hist.pdf')








