import re
import pandas as pd
import numpy as np
from collections import Counter
from itertools import dropwhile


def amenity_activator(row, cols, idx_amenities):
    x = np.zeros(cols)
    for word in row:
        idx = idx_amenities[word]
        x[idx] = 1
    return x


def neat_columns(text):
    # every space is converted in underscore
    text = re.sub(r'[\s+]]*', '_', text)

    # remove every quotation marks
    text = re.sub(r'[\"]', '', text)

    return text


def term_document_matrix(tdm_df, treshold):

    # togliamo le parentesi ad inizio e fine righe, cambio ogni virgola con spazio a seguire con una singola virgola
    # e modifico ogni virgola tra parentesi con un &

    tdm_df['amenities'] = tdm_df['amenities'].apply(lambda x: x[1:-1])
    tdm_df['amenities'] = tdm_df['amenities'].apply(lambda x: x.replace(', ', ','))
    tdm_df['amenities'] = tdm_df['amenities'].apply(lambda x: x.replace(' , ', ','))
    tdm_df['amenities'] = tdm_df['amenities'].apply(lambda x: re.sub(r'(?<=[a-zA-Z0-9])[,](?=[a-zA-Z0-9])', ' ', x))

    # creiamo una bag of words per la colonna amenities
    words_counter = Counter()
    tdm_df['amenities'].str.lower().str.split(',').apply(words_counter.update)

    if treshold > 0:
        for key, count in dropwhile(lambda key_count: key_count[1] >= treshold, words_counter.most_common()):
            del words_counter[key]

    bag_of_words = list(words_counter.keys())

    # creo un dizionario che mappa ad ogni termine un indice
    index_of_amenities = {}
    indx = 0
    stripped_row = []
    k = tdm_df.columns.get_loc('amenities')

    for i in range(len(tdm_df.amenities)):
        items = tdm_df.iloc[i, k]
        tokenized = items.split(',')
        tokenized = [word.lower() for word in tokenized]
        tokenized = list(set(tokenized) & set(bag_of_words))
        stripped_row.append(tokenized)
        for word in tokenized:
            if word not in index_of_amenities:
                index_of_amenities[word] = indx
                indx += 1

    M = len(tdm_df.amenities)
    N = len(index_of_amenities)

    tdmatrix = np.zeros((M, N))
    i = 0
    for row in stripped_row:
        tdmatrix[i, :] = amenity_activator(row, N, index_of_amenities)
        i += 1

    amenities_df = pd.DataFrame(tdmatrix, columns=list(index_of_amenities.keys())).astype(int)
    amenities_df.columns = [neat_columns(text) for text in list(amenities_df.columns)]

    return amenities_df
