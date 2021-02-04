import pickle
import numpy as np

class ProcessingMissingData:
    def __init__(self):
        ## Load the dictionary that we need for predicting values
        with open('Data\\dict_of_cc_dicts_list_of_timestep_values.pkl', 'rb') as handle:
            self.dict_of_cc_dicts_modified = pickle.load(handle)
            
        
    def get_processed_dataset(self, list_of_data, type='mean'):
        max_len = 26

        list_of_processed_trains = list()
        for count, item in enumerate(list_of_data):
            list_of_current_sample_preproccessed = list()
            for feature_idx, feature_vector in enumerate(item.T, 1):
                    len_of_timeseries = len(feature_vector)
                    list_of_additions = list()
                    for i in range(len_of_timeseries+1, max_len+1):
                        if type == 'mean':
                            predicted_value_for_curr_timestep = np.mean(self.dict_of_cc_dicts_modified[feature_idx][i])
                        elif type == 'median':
                            predicted_value_for_curr_timestep = np.median(self.dict_of_cc_dicts_modified[feature_idx][i])
                        elif type == 'zero_padding':
                            predicted_value_for_curr_timestep = 0
                            
                        list_of_additions.append(predicted_value_for_curr_timestep)
                    updated_feature_vector = np.append(feature_vector, np.array(list_of_additions))
                    list_of_current_sample_preproccessed.append(updated_feature_vector)
            list_of_processed_trains.append(np.array(list_of_current_sample_preproccessed).T)
            
        return list_of_processed_trains
    
    def get_processed_dataset_as_list_of_vectors(self, list_of_data, type='mean'):
        list_of_processed_trains = self.get_processed_dataset(list_of_data, type=type)
        list_of_results = list()
        for sample in list_of_processed_trains:
            list_of_sample_results = list()
            for cc_channel in sample.T:
                list_of_sample_results.append(cc_channel)
            list_of_results.append(np.concatenate(list_of_sample_results))
            
        return list_of_results