import os
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from functions.utils import (
    feature_processing,
    univariate_features,
    logistic_regression,
    decision_tree,
    support_vector_machine,
    save_features_plot,
    correlation_matrix,
    tsne2D
)


pathology = 'Diabetes'
path_data = r".\data\diabetes.csv"
output_folder = r'.\functions\images'
os.makedirs(output_folder, exist_ok=True)


raw_data = pd.read_csv(path_data)


save_features_plot(raw_data, output_folder, pathology)


data = feature_processing(raw_data)  # Assicurati che questa funzione accetti pandas DF


data = univariate_features(data, True, False)


dataset_size = len(data)
numPositives = len(data[data['Outcome'] == 1])
per_ones = (numPositives / dataset_size) * 100
numNegatives = dataset_size - numPositives
BalancingRatio = numNegatives / dataset_size
print(f'The number of ones are {numPositives}')
print(f'Percentage of ones are {per_ones:.2f}%')


correlation_matrix(data)
tsne2D(data)


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)


metrics_lr = logistic_regression(train, test)
metrics_dt = decision_tree(train, test, True)
metrics_svm = support_vector_machine(train, test, True)
