import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
import string
import itertools

import seaborn as sns
import matplotlib.pyplot as plt


def remove_outliers(df):

    # boxplots for item_price and item_cnt_day
    plt.figure(figsize=(10, 4))
    plt.xlim(-100, 3000)
    sns.boxplot(x=df.item_cnt_day)

    plt.figure(figsize=(10, 4))
    plt.xlim(df.item_price.min(), df.item_price.max() * 1.1)
    sns.boxplot(x=df.item_price)

    # drop outliers
    df = df[df.item_price < 100000]
    df = df[df.item_cnt_day < 1001]

    return df


def negative_prices_to_itemmeanprice(df, item_id_key = "item_id", shop_id_key = "shop_id", search_key = "item_price"):

    # find items with negative price (2973)
    median = df[(df[shop_id_key] == 32) & (df[item_id_key] == 2973) & (df.date_block_num == 4) & (df[search_key] > 0)][search_key].median()
    df.loc[df.item_price < 0, 'item_price'] = median

    # df = train
    # item_id_key = "item_id"
    # shop_id_key = "shop_id"
    # search_key = "item_price"
    #
    # # ids = df[df[search_key] > 0][item_id_key].values[0]
    #
    # objs = df[df[search_key] > 0][[item_id_key, shop_id_key, search_key]]
    #
    # print('end')
    # for obj in objs:
    #     print(id)
    #     df[df[item_id_key] == obj[item_id_key]].shape  # there are 780 transactions fo that item
    #     # fill negative values with median
    #     median = df[
    #         (df.shop_id_key == obj[shop_id_key]) & (df.item_id == obj[item_id_key]) & (df.date_block_num == 4) & (
    #                 df.search_key > 0)].search_key.median()
    #     df.loc[df.item_price < 0, 'item_price'] = median
    #
    # df[df[search_key] > 0][item_id_key].values[0]

    return df


def get_levenshtein_distance(word1, word2):

    word2 = word2.lower()
    word1 = word1.lower()
    matrix = [[0 for x in range(len(word2) + 1)] for x in range(len(word1) + 1)]


    for x in range(len(word1) + 1):
        matrix[x][0] = x
    for y in range(len(word2) + 1):
        matrix[0][y] = y

    for x in range(1, len(word1) + 1):
        for y in range(1, len(word2) + 1):
            if word1[x - 1] == word2[y - 1]:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1],
                    matrix[x][y - 1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1] + 1,
                    matrix[x][y - 1] + 1
                )

    return matrix[len(word1)][len(word2)]


def merge_shop_duplicate_references(shops, train, test):

    shops["shop_name"] = pd.DataFrame([s.translate(str.maketrans('', '', string.punctuation)) for s in shops["shop_name"]])
    shop_name_similarities = pd.DataFrame({'shop_name_combs': list(itertools.combinations(shops["shop_name"], 2))})
    shop_name_similarities['sim'] = [get_levenshtein_distance(x[0], x[1]) for x in list(itertools.combinations(shops["shop_name"], 2))]
    shop_name_similarities = shop_name_similarities[shop_name_similarities['sim'] < 10]
    shop_name_similarities = shop_name_similarities.sort_values(by=['sim'])

    print(shop_name_similarities)

    # After observing manually the high-similarity shop-names list "shop_name_similarities",
    # we decided that the shops with te following pairs of ids are duplicates:
    # 0 -> 57
    # 1 -> 58
    # 10 -> 11
    # and hence, we decided to keep only one shop_id of each pair and update accordiingly all the occurences of 'shop_id' in train & test sets.

    train.loc[train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57

    train.loc[train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58

    train.loc[train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11

    return train, test


def asd(df):

    return df