from operator import concat
import numpy as np
import pickle

class PersonProcessing:
    def __init__(self, person_array):
        self.person_array = person_array
        
    def _segmentation(self, array, chunks=2):
        splitted = np.array_split(array, chunks)
        list_of_results = []
        for n_array in splitted:
            list_of_results.append(np.mean(n_array))
        
        return np.array(list_of_results)
    
    def _sampling(self, array, samples=2):
        result = array[np.round(np.linspace(0, len(array)-1, samples)).astype(int)]
        return result
    
    def get_segmentation(self, k=2):
        """[Segments the mean value of each chunk from each feature vector (each cepstrum coefficient vector (12-vectors per person)). 
        Then concatanates the result vectors. This way the different length of the time series doesn't matter.]

        Args:
            k (int, optional): [How many k samples do you want to separate the feature vector]. Defaults to 2.

        Returns:
            [ndarray]: [Numpy array resulted from the concatanation of all means from the k samples.]
        """
        list_of_results = []
        for n_array in self.person_array.T:
            list_of_results.append(self._segmentation(n_array, k))
            
        concat_results = np.concatenate(list_of_results)
        return concat_results

    def get_sampling(self, k=2):
        """[Samples data points from each feature vector (each cepstrum coefficient vector (12-vectors per person)). Then concatanates the samples]

        Args:
            k (int, optional): [Number of k samples that we want to obtain for each feature vector. Defaults to 2.

        Returns:
            [ndarray]: [Numpy array resulted from the concatanation of all k samples obtained from the cepstrum coefficients.]
        """
        list_of_results = []
        for n_array in self.person_array.T:
            list_of_results.append(self._sampling(n_array, k))
        
        concat_results = np.concatenate(list_of_results)
        return concat_results
    
    def _peak_detection(self, n_array, lag=5, threshold=5, influence=0):
        signals = np.zeros(len(n_array))
        filteredY = np.array(n_array)
        avgFilter = [0]*len(n_array)
        stdFilter = [0]*len(n_array)
        avgFilter[lag - 1] = np.mean(n_array[0:lag])
        stdFilter[lag - 1] = np.std(n_array[0:lag])
        for i in range(lag, len(n_array)):
            if abs(n_array[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
                if n_array[i] > avgFilter[i-1]:
                    signals[i] = 1
                else:
                    signals[i] = -1

                filteredY[i] = influence * n_array[i] + (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
            else:
                signals[i] = 0
                filteredY[i] = n_array[i]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

        return dict(signals = np.asarray(signals),
                    avgFilter = np.asarray(avgFilter),
                    stdFilter = np.asarray(stdFilter))
        
    def get_peaks(self, lag=5, threshold=5, influence=0):
        list_of_results = []
        for n_array in self.person_array.T:
            list_of_results.append(self._peak_detection(n_array, lag, threshold, influence)['signals'])
        
        concat_results = np.concatenate(list_of_results)
        return concat_results