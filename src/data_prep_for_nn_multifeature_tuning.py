# Code from other files in the repo
import models.sequential_models as sequential_models
from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker
import pickle

def pickle_data(data,path):
	pickle_out = open('{}.pickle'.format(path),'wb')
	pickle.dump(data,pickle_out)
	pickle_out.close()

#-------------------------------- Data setup ----------------------------
ROOT = "C:\\Users\\user\\Documents\\Fifth Year\\ml_postproc"
data_to_collect = ['ttH125_part1-1', 
				   'TTTo2L2Nu', 
				   'TTToHadronic', 
				   'TTToSemiLeptonic']

# Load in data
loader = DataLoader(ROOT)
loader.find_files()
loader.collect_data(data_to_collect,verbose=False)
data = DataProcessing(loader)

cols_to_ignore = ['entry', 'weight_nominal', 'hashed_filename']
cols_events = data.get_event_columns(cols_to_ignore,verbose=False)

data.remove_nan('DiJet_mass')

signal_list = ['ttH125']
data.label_signal_noise(signal_list)

event_labels = LabelMaker.label_encoding(data.return_dataset_labels())
data.set_dataset_labels(event_labels)


sample_weight = WeightMaker.weight_nominal_sample_weights(data)

# Select only the filtered columns from the data
data.filter_data(cols_events)

cols_to_log = ['HT', 'MHT_pt', 'MetNoLep_pt']
data.nat_log_columns(cols_to_log)

columns = data.data.columns.tolist()

min_max_scale_range = (0, 1)
data.normalise_columns(min_max_scale_range)
feature_data = data.data

pickle_data(feature_data,'feature_data')
pickle_data(columns,'columns')
pickle_data(sample_weight,'sample_weight')
pickle_data(event_labels,'event_labels')






