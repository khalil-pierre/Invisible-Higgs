# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 01:12:04 2021

@author: user
"""
import models.sequential_models as sequential_models
import utilities.plotlib as plotlib
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

event_data = np.load('preprocessed_event_data.npy', allow_pickle=True)
event_labels = np.load('preprocessed_event_labels.npy', allow_pickle=True)
sample_weight = np.load('preprocessed_sample_weights.npy', allow_pickle=True)

test_fraction = 0.2
data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
    train_test_split(event_data, event_labels, 
                     sample_weight, test_size=test_fraction)

cols = ['BiasedDPhi', 'DiJet_mass', 'HT', 'MHT_pt', 'MetNoLep_CleanJet_mindPhi',
       'MetNoLep_pt', 'MinChi', 'MinOmegaHat', 'MinOmegaTilde',
       'min_dphi_clean_2j', 'nMediumBJet', 'ncleanedJet']


accuracy = []
for i in range(0,12):
    acc = 0
    for i in range(0,5):
        tf.keras.backend.clear_session()
        data_train_reduced = data_train[:,i]
        data_test_reduced = data_test[:,i]
        
        model = sequential_models.base2(42, 4, input_shape=1)
        
        history = model.fit(data_train_reduced, labels_train, 
                        validation_data=(data_test_reduced, labels_test), 
                        sample_weight=sw_train, epochs=10, verbose=2)
        
        test_loss, test_acc = model.evaluate(data_test_reduced, labels_test, verbose=2)
        
        acc += test_acc
          
    acc = acc / 5
    accuracy += [acc]
    
    
    
    
    
    












