import numpy as np
from scipy.interpolate import splrep, BSpline
from sklearn.preprocessing import MinMaxScaler
from numpy import polyfit, poly1d
from numpy.fft import fft, ifft
import pywt


class FunctionalClustering:
    def __init__(self, data, k, smooth_factor=None):

        self.data = data
        self.k = k
        self.smooth_factor = smooth_factor
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.normalized_data = None
        self.functions = []
        self.cluster_labels = None
        self.distances_to_centroid = None

    def normalize_data(self):
        """

        """
        scaler = MinMaxScaler()
        self.normalized_data = scaler.fit_transform(self.data)

    def fit_b_spline(self, x, y, k=3, num_knots=10):

        t = np.linspace(x.min(), x.max(), num_knots)
        tck = splrep(x, y, k=k, t=t[1:-1], s=self.smooth_factor)
        # tck = splrep(x, y, k=k, t=t[1:-1])
        spline = BSpline(tck[0], tck[1], tck[2])
        return spline



    def fit_all_functions(self):
        """
        """
        pram = 1
        x = np.linspace(0, 1, self.dim)
        for i in range(self.n):
            if pram == 1:
                y = self.normalized_data[i]
                spl = self.fit_b_spline(x, y)
                self.functions.append(spl)

            if pram == 2:
                y = self.normalized_data[i]
                coeffs = polyfit(x, y, deg=4)
                poly_func = poly1d(coeffs)
                self.functions.append(poly_func)

    def find_extrema(self, f, num_points=10):

        x = np.linspace(0, 1, num_points)
        y = f(x)
        dy = np.gradient(y)
        extrema_indices = np.where(np.diff(np.sign(dy)))[0]

        if extrema_indices.size == 0:

            extrema = [x[0], x[-1]]
        else:

            extrema = x[extrema_indices]

        return extrema

    def calculate_similarity(self, f1, f2):

        x = np.linspace(0, 1, self.dim)
        y1 = f1(x)
        y2 = f2(x)

        numeric_distance = np.linalg.norm(y1 - y2)

        extrema1 = self.find_extrema(f1)
        extrema2 = self.find_extrema(f2)

        if len(extrema1) != len(extrema2):
            common_length = max(len(extrema1), len(extrema2))
            extrema1 = np.interp(np.linspace(0, 1, common_length), np.linspace(0, 1, len(extrema1)), extrema1)
            extrema2 = np.interp(np.linspace(0, 1, common_length), np.linspace(0, 1, len(extrema2)), extrema2)

        extrema_distance = np.linalg.norm(extrema1 - extrema2)


        total_distance = numeric_distance + extrema_distance

        return total_distance

    def calculate_Minkowski(self, f1, f2, p=2):

        x = np.linspace(0, 1, self.dim)
        y1 = f1(x)
        y2 = f2(x)
        distance = np.sum(np.abs(y1 - y2) ** p) ** (1 / p)
        return distance

    def calculate_Expansion_coefficient(self, f1, f2):


        coeffs1 = f1.c

        coeffs2 = f2.c
        return np.linalg.norm(coeffs1 - coeffs2)

    def kmeans_pp(self):

        distances = np.full(self.n, np.inf)
        centers = [np.random.randint(0, self.n)]

        for _ in range(1, self.k):
            for i in range(self.n):
                distances[i] = min(distances[i], self.calculate_similarity(self.functions[i], self.functions[centers[-1]]))
            centers.append(np.argmax(distances))

        print(centers)
        labels = np.zeros(self.n)
        distances_to_centroid = np.zeros(self.n)
        for i in range(self.n):
            min_distance = np.inf
            for j in range(self.k):
                dist = self.calculate_similarity(self.functions[i], self.functions[centers[j]])
                if dist < min_distance:
                    min_distance = dist
                    labels[i] = j
            distances_to_centroid[i] = min_distance

        self.cluster_labels = labels
        self.distances_to_centroid = distances_to_centroid

    def run_clustering(self):

        self.normalize_data()
        self.fit_all_functions()
        self.kmeans_pp()
        results = np.vstack((np.arange(self.n), self.cluster_labels, self.distances_to_centroid)).T
        return results




def ExtremeBias_cluster(sampling_strategy, X, y, k):
    # 返回索引
    ori_indices = []

    x_resampled, y_resampled = [], []
    indices = np.where(y == 1)[0]
    X1 = X[indices]
    fc = FunctionalClustering(X1, k)
    result = fc.run_clustering()

    indices = result[:, 2].argsort()
    sorted_data = X1[indices]
    result1 = result[indices]

    for i in range(sampling_strategy[1]):
        ori_indices.append(result1[i][0])
        x_resampled.append(sorted_data[i])
        y_resampled.append(1)

    indices = np.where(y != 1)[0]
    for i in indices:
        ori_indices.append(i)
        x_resampled.append(X[i])
        y_resampled.append(y[i])

    ori_indices = [int(x) for x in ori_indices]
    x_resampled = np.array(x_resampled, dtype=float)
    y_resampled = np.array(y_resampled, dtype=float)
    return x_resampled, y_resampled, ori_indices
