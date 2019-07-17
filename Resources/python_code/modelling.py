import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression



# split train/valid/test set
def split_tests(df):
    X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = df[df.date_block_num < 33]['item_cnt_month']
    X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = df[df.date_block_num == 33]['item_cnt_month']
    X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)

    return X_train, Y_train, X_valid, Y_valid, X_test


def run_light_gbm(X_train, Y_train, X_valid, X_test):

    light_model=lightgbm.LGBMRegressor()
    light_model.fit(X_train, Y_train)

    # save predictions on valid set
    light_val_pred = light_model.predict(X_valid).clip(0., 20.)

    # save predictions on test set
    light_test_pred = light_model.predict(X_test).clip(0., 20.)

    # Create the submission file and submit
    preds = pd.DataFrame(light_test_pred, columns=['item_cnt_month'])
    preds.to_csv('submissionLightGBM.csv',index_label='ID')

    # feature importancies for LightGBM
    feature_imp = pd.DataFrame(sorted(zip(light_model.feature_importances_,X_train.columns)), columns=['Value','Feature'])

    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()

    return light_val_pred, light_test_pred


def run_catboost(X_train, Y_train, X_valid, X_test):

    cat_features = [28,29,30,31]
    cat_model = CatBoostRegressor()
    cat_model.fit(X_train, Y_train, cat_features=cat_features)

    # save predictions on valid set
    cat_val_pred = cat_model.predict(X_valid).clip(0., 20.)
    # save predictions on test set
    cat_test_pred = cat_model.predict(X_test).clip(0., 20.)

    # Create the submission file and submit!
    preds = pd.DataFrame(cat_test_pred, columns=['item_cnt_month'])
    preds.to_csv('submissionCat',index_label='ID')

    return cat_val_pred, cat_test_pred


def run_linear_regression(X_train, Y_train, X_valid, X_test):

    lr_model = LinearRegression()
    lr_model.fit(X_train, Y_train)

    # save predictions on valid set
    lr_val_pred = lr_model.predict(X_valid).clip(0., 20.)
    # save predictions on test set
    lr_test_pred = lr_model.predict(X_test).clip(0., 20.)
    # Create the submission file and submit!
    preds = pd.DataFrame(lr_test_pred, columns=['item_cnt_month'])
    preds.to_csv('submissionLR.csv',index_label='ID')

    return lr_val_pred, lr_test_pred


def stack_models(cat_val_pred, cat_test_pred, lr_val_pred, lr_test_pred, light_val_pred, light_test_pred, Y_valid):

    # create dataframe with predicted values on valid set and actual values
    first_level = pd.DataFrame(light_val_pred, columns=['lightGBM'])
    first_level['catboost'] = cat_val_pred
    first_level['linear_regression'] = lr_val_pred
    first_level['label'] = Y_valid.values

    # create dataframe with predicted values on test set
    first_level_test = pd.DataFrame(light_test_pred, columns=['lightGBM'])
    first_level_test['catboost'] = cat_test_pred
    first_level_test['linear_regression'] = lr_test_pred

    # fit model to learn 'weights' of each regressor
    meta_model = LinearRegression()
    first_level.drop('label', axis=1, inplace=True)
    meta_model.fit(first_level, Y_valid)

    # save prediction on test set
    final_predictions = meta_model.predict(first_level_test)
    prediction_df = pd.DataFrame(final_predictions.clip(0., 20.), columns=['item_cnt_month'])
    prediction_df.to_csv('submissionEns.csv',index_label='ID')