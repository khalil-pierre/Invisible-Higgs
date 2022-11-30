# -*- coding: utf-8 -*-
"""
This aim of this script is to produce histograms of the different event 
variables.
"""

# Code from other files in the repo
from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker

# Python libraries
import copy
import numpy as np
import matplotlib.pyplot as plt

ROOT = "C:\\Users\\user\\Documents\\Fifth Year\\ml_postproc"
data_to_collect = ['ttH125', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

# -------------------------------- Data setup --------------------------------

loader = DataLoader(ROOT)
loader.find_files()
loader.collect_data(data_to_collect)
data = DataProcessing(loader)

data.data['non_normalised_weights'] = data.data.xs_weight

cols_to_ignore1 = ['entry', 'weight_nominal', 
                                 'xs_weight', 'hashed_filename', 
                                 'BiasedDPhi', 'InputMet_InputJet_mindPhi', 
                                 'InputMet_phi', 'InputMet_pt', 'MHT_phi']

cols_to_ignore2 = ['cleanJetMask']

cols_events = data.get_event_columns(cols_to_ignore1)


signal_list = ['ttH125']
data.label_signal_noise(signal_list)

data.data.xs_weight = WeightMaker.weight_nominal_sample_weights(data, 'xs_weight')

signal_df = data.data[data.data.dataset == 'signal']
background_df = data.data[data.data.dataset != 'signal']

non_norm_sample_weight_signal = signal_df.non_normalised_weights.values
non_norm_sample_weight_background = background_df.non_normalised_weights.values

sample_weight_signal = signal_df.xs_weight.values
sample_weight_background = background_df.xs_weight.values

# sample_weight_signal *= total_len / len(sample_weight_signal)
# sample_weight_background *= total_len / len(sample_weight_background)

'''This produces the weighted hist plots if you only care about the ratio of the
signal to background'''
for col in cols_events:
    
    fig,axes = plt.subplots(nrows=2)
    ax1,ax2 = axes.flatten()
    
    Svalues,Sbins,_ = ax1.hist([signal_df[col],background_df[col]], bins=50, 
                               weights = [non_norm_sample_weight_signal,
                                          non_norm_sample_weight_background],
                               alpha=0.5, label=['Signal','Background'],
                               stacked=True)

    ax1.set_ylabel('Count')
    ax1.set_xlabel(col)
    ax1.legend()
    ax1.set_yscale('log')
    
    Svalues,Sbins,_ = ax2.hist(signal_df[col], bins=50, weights=sample_weight_signal, 
              alpha=0.5, label='Signal')
    
    Bvalues,Bbins,_ = ax2.hist(background_df[col], bins=50, weights=sample_weight_background, 
              alpha=0.5, label='Background')
    
    ax2.set_ylabel('Normalised Count')
    ax2.set_xlabel(col)
    ax2.legend()

    #fig.savefig('\\Users\\user\\Documents\\Fifth Year\\invisible-higgs\\Images\\event_hist\\{}_newdata.png'.format(col))
    
    
    # ax3.hist([signal_df[col],background_df[col]], bins=100, stacked=True,
    #           label=['Signal','Background'])
    # ax3.set_ylabel('Count')
    # ax3.set_xlabel(col)
    # ax3.legend()
    
    # ax4.hist([signal_df[col],background_df[col]], bins=100, 
    #           weights=[sample_weight_signal,sample_weight_background],
    #           stacked=True,label=['Signal','Background'])
    # ax4.set_ylabel('Normalised Count')
    # ax4.set_xlabel(col)
    # ax4.legend()
    


    












