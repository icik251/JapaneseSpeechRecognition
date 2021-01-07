import pandas as pd
import numpy as np
import os


class FeatureSelection:

    def __init__(self, filepath, U=4, V=20, K=3):
        self.df = pd.read_pickle(filepath)
        self.U = U
        self.V = V
        self.n = len(self.df[0][0])
        self.K = K
        self.bins = self.init_bins()

    def init_bins(self, U=4, V=20):
        '''
        initializes the bins that through which the signals should cross. It will evenly divide the length of the
        longest time-serie into U equally spaced threshold values. The range of possible values on y-axis is
        divided into V equally spaced thresholds. Once these thresholds values are determined, it can be seen as a grid
        which overlays the time-series. Finally, all possible rectangles within the gris are determined.
        :param U: number of thresholds values on the x-axis
        :type U: int
        :param V: number of threshold values on the y-axis
        :type V: int
        :return: List of all initialized bins
        :rtype: list
        '''
        max_length, y_min, y_max = self.find_limits()

        # x_theta is cast to an int-array for computational convenience
        x_theta = np.arange(0, max_length+1, (max_length/(self.U-1))).astype(int)
        y_theta = np.arange(y_min, y_max+0.01, ((y_max-y_min)/(self.V-1)))
        idx = 0
        bins = np.empty(self.number_of_rectangles(), dtype=object)
        for x1 in range(0, (self.U-1)):
            for x2 in range((x1+1), self.U):
                for y1 in range(0, (self.V-1)):
                    for y2 in range((y1+1), self.V):
                        bins[idx] = Bin(idx, x_theta[x1], x_theta[x2], y_theta[y1], y_theta[y2])
                        idx += 1
        return bins

    def get_features(self):
        '''
        apply the feature extraction to each sample within the data-frame
        :return: dataframe containing the extracted features from the original dataframe
        :rtype: list[list]
        '''
        feature_matrix = np.empty((len(self.df), self.length_feature_vector()))
        for sample_idx in range(len(self.df)):
            sample_features = self.determine_feature_vector(self.df[sample_idx])
            feature_matrix[sample_idx] = sample_features
        return feature_matrix

    def determine_feature_vector(self, signal):
        '''
        uses the number of passes though each bin to determine the extracted features from the signal
        :param signal: multi-dimensional input signal
        :type signal: list[list]
        :return: feature-vector of the input signal
        :rtype: list[]
        '''
        feature_vector = np.empty(self.length_feature_vector())
        idx = 0
        for bin in self.bins:
            for k in range(1, self.K+1):
                passes = bin.count_passes_for_each_dimension(signal, self.n)
                for entry in passes:
                    feature_vector[idx] = 1 if entry >= k else 0
                    idx += 1
                    feature_vector[idx] = 1 if entry < k else 0
        return feature_vector


    def length_feature_vector(self):
        '''
        Returns the length of the feature vector by multiplying the total amount of possible rectangles
         with:
        - how often the signal should pass through the bin (k).
        - the dimensions of the signal (n).
        This is multiplied by 2 to account for the inverted bins as well
        :return: the length of the feature vector
        :rtype: int
        '''
        return 2*self.n*self.K*self.number_of_rectangles()

    def number_of_rectangles(self):
        '''
        Formula used to calculate the possible rectangles within a grid. Based on formula found in:
        https://math.stackexchange.com/questions/1656686/how-many-rectangles-can-be-observed-in-the-grid
        :return: number of possible rectangles
        :rtype: int
        '''
        U = self.U
        V = self.V
        return int((((U-1)*U)/2)*(((V-1)*V)/2))

    def find_limits(self):
        '''
        Finds the limits within the data sample

        :return:    1. the max length of the sample
                    2. the minimum value encountered
                    3. the maximum value encountered
        :rtype: ??? (I have no idea how its called... some kind of tuple?)
        '''
        y_min = float('inf')
        y_max = float('-inf')

        max_length = 0

        for sample in self.df:
            if len(sample) > max_length:
                max_length = len(sample)
            for entry in sample:
                for value in entry:
                    if value > y_max:
                        y_max = value
                    if value < y_min:
                        y_min = value
        return max_length, y_min, y_max

class Bin:
    def __init__(self, idx, x1, x2, y1, y2):
        '''
        The bins are used as a separate class to make it easier to determine how often a timeseries segment passes
        though the bin.
        :param idx: Identifier of the bin
        :type: int
        :param x1: The x-value of the left-most boundary
        :type x1: int
        :param x2: The x-value of the right-most boundary
        :type x2: int
        :param y1: The y-value of the bottom boundary
        :type y1: float
        :param y2: The y-value of the top boundary
        :type y2: float
        '''
        self.idx = idx
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def count_passes_for_each_dimension(self, signal, n=12):
        '''
        Selects the relevant part of the signal and initiates the recursive method used to
        determine the number of passes trough the bins
        :param signal: The signal that requires processing
        :type signal: list
        :param n: The number of dimensions
        :type n: int
        :return: matrix of the passes though all initialized bins separated by dimension
        :rtype: list[list]
        '''
        relevant_signal = signal[self.x1:self.x2+1]
        return self.count_passes_recursive(relevant_signal, n)

    def count_passes_recursive(self, remaining_signal, n=12, inside=np.full(12, False)):
        '''
        Determines how often the signal enters the bin. This is done by checking each time-step whether the signal is
        going from outside of the bin to inside of the bin for each dimension.
        :param remaining_signal: The signal that still need processing
        :type remaining_signal: list[list]
        :param n: number of dimensions
        :type n: int
        :param inside: a boolean representing whether the signal was in- or outside the bin in the previous time-step
        :type inside: bool
        :return: Multi dimensional array showing how often the signal entered the bin over each dimension
        :rtype: list[list]
        '''
        if len(remaining_signal) == 0:  # the end of the relevant signal has been reached
            return np.zeros(n)
        passes = np.zeros(n)
        next_iteration_inside = np.full((n), False)
        for dim in range(n):
            if inside[dim]:
                if self.is_within_box(remaining_signal[0][dim]):
                    next_iteration_inside[dim] = True
            else:
                if self.is_within_box(remaining_signal[0][dim]):
                    next_iteration_inside[dim] = True
                    passes[dim] += 1
        passes_total = np.add(passes, self.count_passes_recursive(remaining_signal[1:], n, next_iteration_inside))
        return passes_total

    def is_within_box(self, y):
        return self.y1 <= y <= self.y2

    def __str__(self):
        return f"idx: {self.idx} - x:({self.x1},{self.x2}) - y:{self.y1},{self.y2})"


# features = FeatureSelection("Data" + os.sep + "test_inputs.pkl")
# max_l, y_m, y_m = features.find_limits()
# print(f"max_length: {max_l}, y_min: {y_m}, y_max: {y_m}")
# result = features.get_features()
