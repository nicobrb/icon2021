from data_cleaning import cleaning
from sys import argv
from os import path
import pandas as pd
import numpy as np
from tdm_matrix import term_document_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def discretize(dframe, describer, colonna):
    first_quartile = int(describer[colonna].loc['25%'])
    fourth_quartile = int(describer[colonna].loc['75%'])

    conditions = [
        (dframe[colonna] <= first_quartile),
        (dframe[colonna] > first_quartile) & (dframe[colonna] <= fourth_quartile),
        (dframe[colonna] > fourth_quartile)
    ]

    cond_range = ['<' + str(first_quartile),
                  '[' + str(first_quartile) + '-' + str(fourth_quartile) + ']', '>' + str(fourth_quartile)]

    first_column = np.select(conditions, cond_range)
    column_name = str(colonna + '_range')

    dframe.insert(0, column_name, first_column)
    dframe = dframe.astype({column_name: 'category'})
    dframe.drop(str(colonna), axis=1, inplace=True)

    return dframe


def num_of_outliers(series, up_lier, low_lier):
    return sum((series > up_lier.loc[series.name,]) | (series < low_lier.loc[series.name,]))


def outlier_values(dataframe):
    """
    Sfrutteremo la formula che considera come outliers tutti quei valori che si trovano 1.5*IQR (interquartile range)
    sopra il terzo quartile o sotto il primo quartile.
    In altre parole:
    low outliers = Q1 - 1.5*IQR dove Q1 è il primo quartile
    high outliers = Q3 + 1.5*IQR dove Q3 è il terzo quartile
    :param dataframe: dataframe per cui calcolare gli outliers
    :return: descrizione numerica delle statistiche degli outliers.
    """

    numeric_exibit = dataframe.describe(include='all').T.round(decimals=3)

    numeric_exibit['IQR'] = numeric_exibit['75%'] - numeric_exibit['25%']
    numeric_exibit['outliers'] = (numeric_exibit['max'] > (numeric_exibit['75%'] + (1.5 * numeric_exibit['IQR']))) | \
                                 (numeric_exibit['min'] < (numeric_exibit['25%'] - (1.5 * numeric_exibit['IQR'])))

    IQR = dataframe.quantile(.75) - dataframe.quantile(.25)

    # si calcolano i low e high outliers
    up_lier = dataframe.quantile(.75) + (1.5 * IQR)
    low_lier = dataframe.quantile(.25) - (1.5 * IQR)

    numeric_exibit['numeric_outl'] = dataframe.apply(num_of_outliers, args=(up_lier, low_lier))
    numeric_exibit.sort_values('numeric_outl', ascending=False, inplace=True)
    columns_order = ['count', 'outliers', 'numeric_outl', 'IQR', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    numeric_exibit = numeric_exibit.reindex(columns=columns_order)
    return numeric_exibit.T


def princ_component_analysis(dataframe, n_comps=30):

    pca = PCA(n_components=n_comps)
    pca.fit(dataframe)
    pca_featurettes = pca.transform(dataframe)
    pca_dframe = pd.DataFrame(pca_featurettes)

    return pca_dframe


def reset_and_drop(dataframe):
    dataframe = dataframe.reset_index()
    dataframe.drop('index', axis=1, inplace=True)

    return dataframe


def main():

    print("Preprocessing starting...")

    if not path.isfile(argv[1]):
        print("file not found or wrong directory, returning")
        return

    dframe = pd.read_csv(argv[1])

    dframe = cleaning(dframe)
    newframe = dframe.copy()

    print("Term-Document Matrix creation starting")
    amenities_reducted = term_document_matrix(dframe, np.sqrt(len(dframe)))
    amenities_for_dframe = term_document_matrix(newframe, 0)

    bayesian_dataframe = pd.read_csv("../datasets/prolog_ready.csv")

    # vanno identificati gli outliers: prima isoliamo le features numeriche da quelle categoriche + le feature senza
    # valore semantico (come 'id'). Queste ultime verranno droppate.

    cols_to_be_dropped = list(dframe.select_dtypes(['O']).columns) + ['id', 'maximum_nights']
    dframe.drop(cols_to_be_dropped, axis=1, inplace=True)
    bayesian_dataframe.drop(cols_to_be_dropped, axis=1, inplace=True)
    describer = outlier_values(dframe)

    price_conditions = [
        (bayesian_dataframe['price'] <= 30),
        (bayesian_dataframe['price'] > 30) & (bayesian_dataframe['price'] <= 50),
        (bayesian_dataframe['price'] > 50) & (bayesian_dataframe['price'] <= 200),
        (bayesian_dataframe['price'] > 200) & (bayesian_dataframe['price'] <= 700),
        (bayesian_dataframe['price'] > 700)
    ]
    prices_range = ['cheap', 'low_cost', 'medium', 'expensive', 'luxury']

    price_column = np.select(price_conditions, prices_range)

    bayesian_dataframe.insert(0, 'price_range', price_column)
    bayesian_dataframe = bayesian_dataframe.astype({'price_range': 'category'})
    bayesian_dataframe.drop('price', axis=1, inplace=True)

    bayesian_dataframe = discretize(bayesian_dataframe, describer, 'minimum_nights')
    bayesian_dataframe = discretize(bayesian_dataframe, describer, 'number_of_reviews')

    # per quanto questa procedura non modifichi nulla del dataframe, è estremamente utile
    # ai fini dell'analisi degli outliers

    # print(outlier_values(dframe))

    drop_cols = ['host_listings_count', 'host_total_listings_count']
    dframe.drop(drop_cols, axis=1, inplace=True)
    bayesian_dataframe.drop(drop_cols, axis=1, inplace=True)

    dframe = reset_and_drop(dframe)
    amenities_for_dframe = reset_and_drop(amenities_for_dframe)
    amenities_reducted = reset_and_drop(amenities_reducted)
    bayesian_dataframe = reset_and_drop(bayesian_dataframe)

    # scaling dei dati attraverso il minmaxscaler
    # print(outlier_values(dframe))

    dframe = pd.concat([dframe, amenities_for_dframe], axis=1)
    dframe = princ_component_analysis(dframe, 20)

    mm_scaler = MinMaxScaler()
    print("Scaling the dataset with MinMax...")
    dframe = pd.DataFrame(mm_scaler.fit_transform(dframe), columns=dframe.columns)
    dframe.round(10)
    print("MinMax Done")

    bayesian_dataframe = pd.concat([bayesian_dataframe, amenities_reducted], axis=1)
    dframe.to_csv('../datasets/preprocessed.csv', index=False)
    bayesian_dataframe.to_csv('../datasets/bayesian_ready.csv', index=False)

    print("Preprocessing done")


main()

