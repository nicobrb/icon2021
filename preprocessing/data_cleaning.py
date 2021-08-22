import re
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def missing_values(dataframe):
    """
    Considera e calcola la percentuale di celle vuote nelle colonne del dataframe
    :param dataframe: dataframe per cui calcolare la percentuale di valori mancanti per colonna
    :return restituisce un dataframe contenente il nome di ogni colonna e la percentuale di valori mancanti per ognuna
    """

    empty_vals = dataframe.isnull().sum().to_frame()
    empty_vals.columns = ['mancanti']
    empty_vals['percentuale'] = np.round(100 * (empty_vals['mancanti'] / dataframe.shape[0]))
    empty_vals.sort_values(by='mancanti', ascending=False, inplace=True)

    return empty_vals


def replace_nan_vals(columns, dataframe):
    """
    Considera una lista di colonne e un dataframe e si comporta in questo modo:
    - colonna categorica (tipo oggetto) -> celle vuote sostituite con il valore most_frequent
    - colonna di float/int -> celle vuote sostituite con la mediana
    :param columns: lista di colonne da analizzare
    :param dataframe: dataframe da modificare
    :return il dataframe modificato
    """
    for col in columns:
        if str(dataframe[col].dtype) == 'category':
            # print("replacing empty cells in {} column with the most frequent value".format(col))
            most_freq_subs = SimpleImputer(strategy='most_frequent')
            dataframe.loc[:, col] = most_freq_subs.fit_transform(dataframe[[col]])
        elif dataframe[col].dtype == 'float64' or dataframe[col].dtype == 'int64':
            # print("replacing empty cells in {} column with median value".format(col))
            median_subs = SimpleImputer(strategy='median')
            dataframe.loc[:, col] = median_subs.fit_transform(dataframe[[col]])
        else:
            raise ValueError("Column type is invalid")

    return dataframe


def cleaning(dframe):
    print("dropping useless columns...")
    dframe = dframe.drop(["listing_url", "scrape_id", "last_scraped", "neighborhood_overview",
                          "picture_url", "host_id", "host_url", "host_name", "host_since", "host_location",
                          "host_about", "host_thumbnail_url", "host_picture_url", "host_neighbourhood",
                          "neighbourhood", "latitude", "longitude", "bathrooms",
                          "minimum_minimum_nights", "maximum_minimum_nights",
                          "minimum_maximum_nights", "maximum_maximum_nights", "minimum_nights_avg_ntm",
                          "maximum_nights_avg_ntm", "calendar_updated", "availability_30",
                          "availability_60", "availability_90", "availability_365", "calendar_last_scraped",
                          "number_of_reviews_ltm", "number_of_reviews_l30d", "first_review", "last_review", "license",
                          "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
                          "calculated_host_listings_count_private_rooms",
                          "calculated_host_listings_count_shared_rooms", "reviews_per_month"
                          ], axis=1)

    dframe = dframe[dframe.property_type.isin(['Entire apartment', 'Private room in apartment',
                                               'Private room in house', 'Private room in townhouse',
                                               'Entire condominium', 'Entire house', 'Entire loft',
                                               'Entire townhouse'])]

    dframe['property_type'] = dframe.property_type.apply(lambda x: re.sub(' ', '_', x))
    dframe['room_type'] = dframe.room_type.apply(lambda x: re.sub(' ', '_', x))
    dframe["neighbourhood_cleansed"] = dframe["neighbourhood_cleansed"].apply(lambda x: re.sub(' ', '_', x))
    dframe["neighbourhood_cleansed"] = dframe["neighbourhood_cleansed"].apply(lambda x: re.sub(',', '', x))

    # priviamo la colonna del prezzo del simbolo $ e la convertiamo in variabile numerica

    dframe['price'] = dframe.price.apply(lambda x: re.sub(r'[$,]', '', x)).astype('float')

    mean_col = dframe.loc[:, "review_scores_rating":"review_scores_value"]

    mean_col = mean_col.fillna(0)
    dframe['avg_rating'] = np.round(mean_col.mean(axis=1), 2)
    dframe["avg_rating"] = dframe.apply(lambda x: 0 if x["avg_rating"] < 3 else 1, axis=1)

    empty_vals = missing_values(dframe)
    dropped_list = list(empty_vals[empty_vals.percentuale > 20].index)
    dframe = dframe.drop(dropped_list, axis=1)
    dframe = dframe.rename(columns={'bathrooms_text': 'bathrooms'})
    dframe['bathrooms'] = dframe['bathrooms'].fillna(0)
    dframe["bathrooms"] = dframe.apply(lambda x: 0 if x["bathrooms"] == 0 else x["bathrooms"].split(" ")[0], axis=1)
    dframe["bathrooms"] = dframe["bathrooms"].replace('Half-bath', '0.5')
    dframe["bathrooms"] = dframe["bathrooms"].replace('Private', '1')
    dframe["bathrooms"] = dframe["bathrooms"].replace('Shared', '1')
    dframe["bathrooms"] = dframe["bathrooms"].astype('float')

    categorical = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'property_type',
                   'room_type', 'has_availability', 'instant_bookable', 'neighbourhood_cleansed']

    dframe[categorical] = dframe[categorical].apply(lambda x: x.astype('category'), axis=0)

    """
    conviene droppare ora le righe che non hanno la stanza da letto inserita: sostituirle con la moda non
    avrebbe molto senso visto che corrispondono al 10% del dataset totale (troppo influenzato)
    """

    dframe = dframe[dframe['bedrooms'].notna()]
    dframe.reset_index()
    dframe['id'] = range(1, len(dframe) + 1)

    # colonne di testo vuote -> stringhe vuote
    # colonne categoriche vuote -> moda
    # colonne continue vuote -> mediana

    dframe.loc[dframe.description.isna().copy(), 'description'] = ''
    dframe.loc[dframe.name.isna().copy(), 'name'] = ''

    continue_missing_vals = ['host_listings_count', 'host_total_listings_count', 'beds']
    category_missing_vals = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']
    dframe = replace_nan_vals(continue_missing_vals, dframe)
    dframe = replace_nan_vals(category_missing_vals, dframe)

    # a questo punto il file è pronto per essere elaborato dalla parte di codice gestita da prolog

    print("Prolog preprocessing is ready.")
    dframe.to_csv('../datasets/prolog_ready.csv', index=False)

    # eseguire one-hot encoding delle variabili categoriche: si prendono i dummies delle variabili categoriche,
    # segue un merge con il dataframe originario con conseguente drop delle variabili categoriche iniziali

    category_one_hot_encoding = pd.get_dummies(dframe[categorical])
    dframe = pd.concat([dframe, category_one_hot_encoding], axis=1).drop(categorical, axis=1)

    print("Data Cleaning done.")

    return dframe
