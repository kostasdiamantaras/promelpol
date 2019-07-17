import pandas as pd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    items = pd.read_csv('datasets/items.csv')
    shops = pd.read_csv('datasets/shops.csv')
    cats = pd.read_csv('datasets/item_categories.csv')
    train = pd.read_csv('datasets/sales_train.csv')
    test = pd.read_csv('datasets/test.csv').set_index('ID')
    return items, cats, shops, train, test


# Function for label encoding to city_imp, type and subtype imp
def importance_score(df, group_key):

    aggregate_key = 'item_cnt_day'

    # Build city dataframe based on
    group_key_instances = df.groupby([group_key]).agg({aggregate_key: ['sum']})

    # cities = train2.groupby(group_key).agg({aggregate_key: ['sum']})
    group_key_instances.reset_index(level=0, inplace=True)
    group_key_instances.columns = group_key_instances.columns.droplevel(1)

    # Sort items inside
    group_key_instances = group_key_instances.sort_values(by=[aggregate_key], ascending=True)
    group_key_instances.reset_index(drop=True, inplace=True)
    group_key_instances['importance_score'] = group_key_instances.reset_index().index
    return pd.Series(group_key_instances.importance_score.values,index=group_key_instances[group_key]).to_dict()


# Generate a matrix with all possible combinations of month, shop & item ids
def get_comb_matrix(train):

    number_of_months = 34
    matrix = []
    cols = ['date_block_num', 'shop_id', 'item_id']

    for i in range(number_of_months):
        sales = train[train.date_block_num == i]
        matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
    matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
    matrix['item_id'] = matrix['item_id'].astype(np.int16)
    matrix.sort_values(cols, inplace=True)

    return matrix

# Check for time patterns
def check_time_patterns(train):
    # check for patterns in time
    group = train.groupby(['date_block_num']).agg({'item_cnt_day': ['sum']})
    group.columns = ['date_avg_item_cnt']
    group.reset_index(inplace=True)

    plt.plot(group.date_block_num, group.date_avg_item_cnt)
    plt.xlabel('Months')
    plt.ylabel('Total Sales')
    plt.title('Total sales by month')
    plt.show()


# Fill na values created from lags
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)
    return df


def fill_null(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df['ROC_13_12'].fillna(0, inplace=True)
    return df