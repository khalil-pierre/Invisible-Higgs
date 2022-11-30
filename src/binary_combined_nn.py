"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import classifier
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResultsMulti

# ------------------------------------ Main -----------------------------------

if __name__ == "__main__":
    DIR = 'data_binary_classifier\\'
    
    args_model = {'model_type' : 'binary_classifier',
                  'model_architecture' : 'CombinedNN',
                  'ffn_layer_1_neurons' : 16,
                  'ffn_layer_2_neurons' : 8,
                  'rnn_layer_1_neurons' : 64,
                  'rnn_layer_2_neurons' : 8,
                  'final_layer_neurons' : 8,
                  'output_shape' : 1,
                  'loss_function' : 'binary_crossentropy',
                  'learning_rate' : 0,
                  'batch_size' : 64,
                  'epochs' : 16,
                  'model' : 'base'}
    
    num_runs = 1
    dataset_sample = 0.05
    
    all_results = ModelResultsMulti()
    neural_net = classifier.CombinedNN(args_model)  
    neural_net.load_data(DIR)
    neural_net.load_event_data(DIR)
    neural_net.load_jet_data(DIR)
    
    for i in range(num_runs):
        model_result = classifier.run(i, neural_net, args_model, dataset_sample)
        all_results.add_result(model_result, args_model)
    
    df_all_results = all_results.return_results()
    # all_results.save('binary_combined_nn.pkl')
    DIR = "\\Users\\user\\Documents\\Fifth Year\\invisible-higgs\\models_trained\\"
    neural_net.model.save(DIR + 'combined_nn.h5')
# -------------------------- Results plots parameters -------------------------

params_history = {'title' : ('Model accuracy of combined neural network '
                             'trained on event and jet data'),
                'x_axis' : 'Epoch number',
                'y_axis' : 'Accuracy',
                'legend' : ['training data', 'test data'],
                'figsize' : (6, 4),
                'dpi' : 200,
                'colors' : ['#662E9B', '#F86624'],
                'full_y' : False}

params_cm = {'title' : ('Confusion matrix of combined neural network '
                              'trained on event and jet data'),
              'x_axis' : 'Predicted label',
              'y_axis' : 'True label',
              'class_names' : ['ttH (signal)', 'tt¯ (background)'],
              'figsize' : (6, 4),
              'dpi' : 200,
              'colourbar' : False}

params_roc = {'title' : ('ROC curve for the combined neural network '
                              'trained on event and jet data'),
              'x_axis' : 'False Positive Rate',
              'y_axis' : 'True Positive Rate',
              'figsize' : (6, 4),
              'dpi' : 200}

params_discrim = {'title' : ('Distribution of discriminator values for the '
                              'combined neural network trained on event and jet data'),
                  'x_axis' : 'Label prediction',
                  'y_axis' : 'Number of events',
                  'num_bins' : 50,
                  'figsize' : (6, 4),
                  'dpi' : 200,
                  'colors' : ['brown', 'teal']}
    
# ---------------------------- Plots of best result ---------------------------

# Get the index of the row with the best accuracy on the test dataset
idx_best = df_all_results['accuracy_training'].argmax()
df_model_result = df_all_results.iloc[idx_best]

# Plot best training history
fig = plotlib.training_history_plot(df_model_result['history_training_data'], 
                                    df_model_result['history_test_data'], 
                                    params_history)

# Plot best confusion matrix
fig = plotlib.confusion_matrix(df_model_result, params_cm)

# Plot best ROC curve
fig = plotlib.plot_roc(df_model_result, params_roc)

# Plot distribution of discriminator values
fig_combined = plotlib.plot_discriminator_vals(*classifier.EventNN.discriminator_values(neural_net), params_discrim)

discrim_vals_combined = classifier.EventNN.discriminator_values(neural_net)
