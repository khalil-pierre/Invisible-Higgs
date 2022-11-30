import pickle

def pickle_in(path):
    pickle_in = open(path,'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    return data

data = pickle_in('feature_data.pickle')
event_lab = pickle_in('event_labels.pickle')
weight = pickle_in('sample_weight.pickle')
cols = pickle_in('columns.pickle')



