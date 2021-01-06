import numpy as np
import csv
import pickle
import os

# Train feature vectors 
train_reader = csv.reader(open("Data"+os.sep+"ae.train"), delimiter=" ")

list_of_train = []
curr_np_array = np.empty(shape=(0,12))
for row in train_reader:
    # remove last element as it is empty space
    while row[-1] == '':
        row.pop()
    if float(row[0]) == 1:
        list_of_train.append(curr_np_array)
        curr_np_array = np.empty(shape=(0,12))
    else:
        curr_np_array = np.append(curr_np_array, [list(map(float, row))], axis=0)
        
# Test feature vectors
test_reader = csv.reader(open("Data"+os.sep+"ae.test"), delimiter=" ")

list_of_test = []
curr_np_array = np.empty(shape=(0,12))
for row in test_reader:
    if len(row) == 0:
        continue
    # remove last element as it is empty space
    while row[-1] == '':
        row.pop()
    if float(row[0]) == 1:
        list_of_test.append(curr_np_array)
        curr_np_array = np.empty(shape=(0,12))
    else: 
        curr_np_array = np.append(curr_np_array, [list(map(float, row))], axis=0)
    
# Train teacher signals
list_of_train_labels = []
for i in range(1,271):
    signal_len = np.shape(list_of_train[i-1])[0]
    speaker_label = int(np.ceil(i/30))
    curr_array = np.zeros(shape=(signal_len,9))
    curr_array[:,speaker_label-1] = np.ones(shape=(signal_len))
    list_of_train_labels.append(curr_array)
    
# Test teacher signals
list_of_test_labels=[]
block_counter = 0
speaker_label = 0
block_lengthes = [31,35,88,44,29,24,40,50,29]
for i in range(370):
    block_counter += 1
    if block_counter == block_lengthes[speaker_label]+1:
        speaker_label += 1
        block_counter = 1
    signal_len = np.shape(list_of_test[i-1])[0]
    curr_array = np.zeros(shape=(signal_len,9))
    curr_array[:,speaker_label] = np.ones(shape=(signal_len))
    list_of_test_labels.append(curr_array)
    
# Save as pickles
with open('Data'+os.sep+'train_inputs.pkl', 'wb') as handle:
    pickle.dump(list_of_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('Data'+os.sep+'train_outputs.pkl', 'wb') as handle:
    pickle.dump(list_of_train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('Data'+os.sep+'test_inputs.pkl', 'wb') as handle:
    pickle.dump(list_of_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('Data'+os.sep+'test_outputs.pkl', 'wb') as handle:
    pickle.dump(list_of_test_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)