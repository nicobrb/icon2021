import numpy as np
import pandas as pd
from pgmpy.readwrite import XMLBIFReader
from pgmpy.inference import VariableElimination


class BeliefNet:

    xmlbif_path = './datasets/bn_ratingradice.xml'
    data_path = './datasets/bayesian_with_clusters.csv'
    df = None
    model = None
    inference = None

    def __init__(self, ids, favourite_features):
        self.ids = ids
        self.features = favourite_features
        self.build_bn()
        self.read_data()

    def build_bn(self):
        reader = XMLBIFReader(self.xmlbif_path)
        self.model = reader.get_model()
        self.inference = VariableElimination(self.model)

    def read_data(self):
        self.df = pd.read_csv(self.data_path)

    def compute_probabilities(self):  # attenzione: tutti i 1.0 e 0.0 in bayesian_with_cluster devono diventare 1 e 0
        df_reduced = self.df.loc[self.ids, self.features]
        dict_features_value = df_reduced.T.to_dict()
        dict_features_value = {n: {str(key): str(value) for key, value in dict_features_value[n].items()}
                               for n in dict_features_value.keys()}
        results = [self.inference.query(['avg_rating'],dict_features_value[id]).values[1] for id in self.ids]
        results = (100*np.round(results, 2)).astype(int)
        results = [str(result)+'%' for result in results]
        return results


def main():
    ids = [x for x in range(10,19)]
    features = ['wifi','oven']
    b = BeliefNetwork(ids, features)
    results = b.compute_probabilities()
    print(results)

