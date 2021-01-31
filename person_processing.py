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