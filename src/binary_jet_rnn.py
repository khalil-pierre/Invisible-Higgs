"""
This file contains the training on the RNN model on the jet data.
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
                  'model_architecture' : 'JetRNN',
                  'layer_1_neurons' : 64,
                  'layer_2_neurons' : 8,
                  'output_shape' : 1,
                  'learning_rate' : 0.001,
                  'batch_size' : 64,
                  'epochs' : 8,
                  'model' : 'base'}
    
    num_runs = 1
    dataset_sample = 0.05
    
    all_results = ModelResultsMulti()
    jet_rnn = classifier.JetRNN(args_model)
    jet_rnn.load_data(DIR)
    jet_rnn.load_jet_data(DIR)
    
    for i in range(num_runs):
        model_result = classifier.run(i, jet_rnn, args_model, dataset_sample)
        all_results.add_result(model_result, args_model)
    
    df_all_results = all_results.return_results()
    # all_results.save('binary_jet_rnn.pkl')
    DIR = "\\Users\\user\\Documents\\Fifth Year\\invisible-higgs\\models_trained\\"
    jet_rnn.model.save(DIR + 'jet_rnn.h5')

# -------------------------- Results plots parameters -------------------------

params_history = {'title' : ('Model accuracy of recurrent neural network '
                             'trained on jet data'),
                'x_axis' : 'Epoch number',
                'y_axis' : 'Accuracy',
                'legend' : ['training data', 'test data'],
                'figsize' : (6, 4),
                'dpi' : 200,
                'colors' : ['#662E9B', '#F86624'],
                'full_y' : False}

params_cm = {'title' : ('Confusion matrix of recurrent neural network '
                              'trained on jet data'),
              'x_axis' : 'Predicted label',
              'y_axis' : 'True label',
              'class_names' : ['ttH (signal)', 'tt¯ (background)'],
              'figsize' : (6, 4),
              'dpi' : 200,
              'colourbar' : False}

params_roc = {'title' : ('ROC curve for the recurrent neural network '
                              'trained on jet data'),
              'x_axis' : 'False Positive Rate',
              'y_axis' : 'True Positive Rate',
              'figsize' : (6, 4),
              'dpi' : 200}

params_discrim = {'title' : ('Distribution of discriminator values for the '
                              'recurrent neural network trained on jet data'),
                  'x_axis' : 'Label prediction',
                  'y_axis' : 'Number of events',
                  'num_bins' : 50,
                  'figsize' : (6, 4),
                  'dpi' : 200,
                  'colors' : ['brown', 'teal']}

# --------------------------- Averaged results plots --------------------------

# Plot average training history
data_mean1, data_std1 = all_results.average_training_history('history_training_data')
data_mean2, data_std2 = all_results.average_training_history('history_test_data')
fig = plotlib.training_history_plot(data_mean1, data_mean2, params_history, 
                                    error_bars=[data_std1, data_std2])
print(f'average training accuracy: {data_mean1[-1]:0.4f} \u00B1 {data_std1[-1]:0.4f}')
print(f'average test accuracy:     {data_mean2[-1]:0.4f} \u00B1 {data_std2[-1]:0.4f}')

# Plot average confusion matrix
data_mean1, data_std1 = all_results.average_confusion_matrix()
fig = plotlib.confusion_matrix(data_mean1, params_cm)

# Plot average ROC curve
fig = plotlib.plot_roc(all_results.average_roc_curve(), params_roc)

# # ---------------------------- Plots of best result ---------------------------

# # Get the index of the row with the best accuracy on the test dataset
# idx_best = df_all_results['accuracy_training'].argmax()
# df_model_result = df_all_results.iloc[idx_best]

# # Plot best training history
# fig = plotlib.training_history_plot(df_model_result['history_training_data'], 
#                                     df_model_result['history_test_data'], 
#                                     params_history)

# # Plot best confusion matrix
# fig = plotlib.confusion_matrix(df_model_result, params_cm)

# # Plot best ROC curve
# fig = plotlib.plot_roc(df_model_result, params_roc)

# # Plot distribution of discriminator values
# fig_rnn = plotlib.plot_discriminator_vals(*classifier.EventNN.discriminator_values(jet_rnn), params_discrim)

# discrim_vals_rnn = classifier.EventNN.discriminator_values(jet_rnn)



