import pandas as pd

def load_datasets():
    '''This function will upload the necessary datasets
    to perform the project.'''
    data_1 = pd.read_csv('./files/datasets/input/geo_data_0.csv')
    data_2 = pd.read_csv('./files/datasets/input/geo_data_1.csv')
    data_3 = pd.read_csv('./files/datasets/input/geo_data_2.csv')
    return data_1,data_2,data_3