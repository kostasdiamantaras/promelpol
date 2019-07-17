import utils
import preprocessing
import feature_engineering
import modelling
import pandas as pd

# Load data
items, cats, shops, train, test = utils.load_data()


###### Preproccessing
train = preprocessing.remove_outliers(train)
train = preprocessing.negative_prices_to_itemmeanprice(train)
train, test = preprocessing.merge_shop_duplicate_references(shops, train, test)


###### Shop name correction
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'


###### Feature engineering
# Add new 'city' feature
shops = feature_engineering.add_shop_city_attr(shops)

# Add type and suptype feature
cats = feature_engineering.add_category_type_subtype_attr(cats)

# Replace city, type, subtype with importancy weights
shops, items, cats = feature_engineering.add_city_type_subtype_imp_attr(train, shops, items, cats)

# Generate a matrix with all possible combinations of month, shop & item ids
matrix = utils.get_comb_matrix(train)

matrix = feature_engineering.add_monthly_sales_attr(matrix, train, test)

matrix = feature_engineering.merge_matrix_shops_items_cats(matrix, shops, items, cats)

# Number of items sold per month
matrix = feature_engineering.generate_lag_feature(matrix, [1,2,3,12], 'item_cnt_month')

matrix = feature_engineering.add_windows_features(matrix)

matrix = feature_engineering.add_trend_features(matrix, train)

matrix = feature_engineering.add_month_days(matrix)

matrix = feature_engineering.add_seasons(matrix)

matrix = feature_engineering.add_december_distance(matrix)

matrix = feature_engineering.add_first_last_sale(matrix)

# Check for time patterns
utils.check_time_patterns(train)

# Drop first 12 months
matrix = matrix[matrix.date_block_num > 11]

# Fill na values created from lags
matrix = utils.fill_na(matrix)

matrix.to_pickle('datasets/data.pkl')
data = pd.read_pickle('datasets/data.pkl')

# fill null values for LinearRegression to work
data = utils.fill_null(data)

### Modelling
X_train, Y_train, X_valid, Y_valid, X_test = modelling.split_tests(data)

# Run Light GBM
light_val_pred, light_test_pred = modelling.run_light_gbm(X_train, Y_train, X_valid, X_test)

# Run Cat Boost
cat_val_pred, cat_test_pred = modelling.run_catboost(X_train, Y_train, X_valid, X_test)

# Run Linear Regression
lr_val_pred, lr_test_pred = modelling.run_linear_regression(X_train, Y_train, X_valid, X_test)

# Run Model Ensemble
modelling.stack_models(cat_val_pred, cat_test_pred, lr_val_pred, lr_test_pred, light_val_pred, light_test_pred, Y_valid)