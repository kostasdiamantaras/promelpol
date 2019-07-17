import pandas as pd
import utils
import numpy as np

def add_shop_city_attr(shops):# shop name correction
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    # create city attribute
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
    shops.drop(['shop_name'], axis=1, inplace=True)

    return shops

def add_category_type_subtype_attr(cats):
    # create type and subtype attribute
    cats['split'] = cats['item_category_name'].str.split('-')
    cats['type'] = cats['split'].map(lambda x: x[0].strip())
    # if subtype is nan then type
    cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    cats.drop(['split','item_category_name'], axis=1, inplace=True)

    return cats

# Replace city, type, subtype with importancy weights
def add_city_type_subtype_imp_attr(train, shops, items, cats):

    # merge datasets to get insights
    train = pd.merge(train, shops, on=['shop_id'], how='left')
    train = pd.merge(train, items, on=['item_id'], how='left')
    train = pd.merge(train, cats, on=['item_category_id'], how='left')

    # map textual data to numbers
    city_importance_score_dict = utils.importance_score(train, 'city')
    type_importance_score_dict = utils.importance_score(train, 'type')
    subtype_importance_score_dict = utils.importance_score(train, 'subtype')

    shops['city_code_imp'] = shops['city'].map(lambda x: city_importance_score_dict[x])
    cats['type_code_imp'] = cats['type'].map(lambda x: type_importance_score_dict[x])
    cats['subtype_code_imp'] = cats['subtype'].map(lambda x: subtype_importance_score_dict[x])

    # drop redundant information
    shops.drop(['city'], axis=1, inplace=True)
    items.drop(['item_name'], axis=1, inplace=True)
    cats.drop(['type' ,'subtype'], axis=1, inplace=True)

    return shops, items, cats


def add_monthly_sales_attr(matrix, train, test):

    cols = ['date_block_num', 'shop_id', 'item_id']
    group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
    group.columns = ['item_cnt_month']
    group.reset_index(inplace=True)

    # merge matrix with monthly sales
    matrix = pd.merge(matrix, group, on=cols, how='left')
    matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0, 20)  # NB clip target here
                                .astype(np.float16))

    # set test set as the 34-th month and merge with matrix
    test['date_block_num'] = 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)

    matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
    matrix.fillna(0, inplace=True)

    return matrix


def merge_matrix_shops_items_cats(matrix, shops, items, cats):
    # merge matrix with rest of files
    matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
    matrix = pd.merge(matrix, items, on=['item_id'], how='left')
    matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
    matrix['city_code_imp'] = matrix['city_code_imp'].astype(np.int8)
    matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
    matrix['type_code_imp'] = matrix['type_code_imp'].astype(np.int8)
    matrix['subtype_code_imp'] = matrix['subtype_code_imp'].astype(np.int8)

    return matrix

# Lag features
def generate_lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df


def add_windows_features(matrix):
    # Rolling windows features

    # average item sold for each month
    group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
    matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
    matrix = generate_lag_feature(matrix, [1], 'date_avg_item_cnt')
    matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

    # average items sold by item for each month
    group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_item_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
    matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
    matrix = generate_lag_feature(matrix, [1, 2, 3, 12], 'date_item_avg_item_cnt')
    matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

    # average items sold by shop for each month
    group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id'], how='left')
    matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
    matrix = generate_lag_feature(matrix, [1, 2, 3, 12], 'date_shop_avg_item_cnt')
    matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

    # average items sold by item category id for each month
    group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_cat_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_category_id'], how='left')
    matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
    matrix = generate_lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
    matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

    # average items sold by shop and by category id for each month
    group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_shop_cat_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
    matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
    matrix = generate_lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
    matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

    # average items sold by city for every month
    group = matrix.groupby(['date_block_num', 'city_code_imp']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_city_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code_imp'], how='left')
    matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
    matrix = generate_lag_feature(matrix, [1], 'date_city_avg_item_cnt')
    matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

    # average items sold by item and by city for every month
    group = matrix.groupby(['date_block_num', 'item_id', 'city_code_imp']).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_item_city_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code_imp'], how='left')
    matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
    matrix = generate_lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
    matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

    # distance of 12 and 13 previous months
    group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
    group.columns = ['distance_avg_item_cnt']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
    matrix['distance_avg_item_cnt'] = matrix['distance_avg_item_cnt'].astype(np.float16)

    matrix = generate_lag_feature(matrix, [12, 13], 'distance_avg_item_cnt')
    matrix.drop(['distance_avg_item_cnt'], axis=1, inplace=True)
    matrix['ROC_13_12'] = (matrix['distance_avg_item_cnt_lag_12'] - matrix['distance_avg_item_cnt_lag_13']) / matrix[
        'distance_avg_item_cnt_lag_13']
    matrix.drop(['distance_avg_item_cnt_lag_12', 'distance_avg_item_cnt_lag_13'], axis=1, inplace=True)

    return matrix

def add_trend_features(matrix, train):
    # Trend features

    group = train.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['item_id'], how='left')
    matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

    group = train.groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_item_avg_item_price']
    group.reset_index(inplace=True)

    matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id'], how='left')
    matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

    lags = [1, 2, 3, 4, 5, 6]
    matrix = generate_lag_feature(matrix, lags, 'date_item_avg_item_price')

    for i in lags:
        matrix['delta_price_lag_' + str(i)] = \
            (matrix['date_item_avg_item_price_lag_' + str(i)] - matrix['item_avg_item_price']) / matrix[
                'item_avg_item_price']

    def select_trend(row):
        for i in lags:
            if row['delta_price_lag_' + str(i)]:
                return row['delta_price_lag_' + str(i)]
        return 0

    matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
    matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
    matrix['delta_price_lag'].fillna(0, inplace=True)

    fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for i in lags:
        fetures_to_drop += ['date_item_avg_item_price_lag_' + str(i)]
        fetures_to_drop += ['delta_price_lag_' + str(i)]

    matrix.drop(fetures_to_drop, axis=1, inplace=True)

    return matrix

def add_month_days(matrix):
    # create feature month
    matrix['month'] = matrix['date_block_num'] % 12
    # map days of each month
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    matrix['days'] = matrix['month'].map(days).astype(np.int8)
    return matrix

def add_seasons(matrix):
    # add seasons to the dataset
    matrix['season'] = matrix["month"].map(lambda x: 'winter' if x in [11, 0, 1] else x)
    matrix['season'] = matrix["season"].map(lambda x: 'spring' if x in [2, 3, 4] else x)
    matrix['season'] = matrix["season"].map(lambda x: 'summer' if x in [5, 6, 7] else x)
    matrix['season'] = matrix["season"].map(lambda x: 'autumn' if x in [8, 9, 10] else x)
    df_dummies = pd.get_dummies(matrix.season)
    matrix = pd.concat([matrix, df_dummies], axis=1)
    matrix.drop(['season'], axis=1, inplace=True)
    return matrix

def add_december_distance(matrix):
    # difference from december
    matrix['diff_from_dec'] = matrix['month'] - 11

    return matrix


def add_first_last_sale(matrix):
    # Months, since the first and the last sale, for each item and for each pair of item and shop
    cache = {}
    matrix['item_shop_last_sale'] = -1
    matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():
        key = str(row.item_id) + ' ' + str(row.shop_id)
        if key not in cache:
            if row.item_cnt_month != 0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num

    cache = {}
    matrix['item_last_sale'] = -1
    matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
    for idx, row in matrix.iterrows():
        key = row.item_id
        if key not in cache:
            if row.item_cnt_month != 0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            if row.date_block_num > last_date_block_num:
                matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
                cache[key] = row.date_block_num

    return matrix