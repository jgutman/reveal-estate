import pandas as pd
import numpy as np
import final_modeling as fm
import final_data_clean as dc

def prepare_data(X_train_raw, affected_properties, updated_affected_properties):

    '''
    Prepares the dataframe of affected properties to have the model applied.

    Args:
        X_train_raw: original training features used to fit the imputer and
            feature scaler for model training data
        affected_properties: raw features and outcome for affected properties
            without shifted distance to subway features
        updated_affected_properties: raw features and outcome for affected
            properties with shifted distance to subway features

    Returns:
        X_pre_lightrail: processed X features for affected properties prior to
            distance to subway shift
        X_post_lightrail: processed X features for affected properties after
            the distance to subway shift
        y_true: true outcome data for affected properties prior to shift

    '''
    X_pre_lightrail, y_true = fm.create_target_var(affected_properties,
        'price_per_sqft')
    X_post_lightrail, _ = fm.create_target_var(updated_affected_properties,
        'price_per_sqft')

    _, X_pre_lightrail = dc.fill_na(X_train_raw,
        X_pre_lightrail)
    X_train_raw, X_post_lightrail = dc.fill_na(X_train_raw,
        X_post_lightrail)

    _, X_pre_lightrail = dc.normalize(X_train_raw,
        X_pre_lightrail)
    X_train_raw, X_post_lightrail = dc.normalize(X_train_raw,
        X_post_lightrail)

    return X_pre_lightrail, X_post_lightrail, y_true


def make_prediction(X_pre_lightrail, X_post_lightrail, y_true, model,
        output = "price_increase.csv"):
    '''
    Predicts price_per_sqft for the dataframe with original and updated subway
    information, and creates Pandas dataframe with affected BBLs, the original
    predicted values for price per square feet under the true features in the
    data, and the new predictions for price per square feet with subway
    distances reduced to 0.5 mi.

    Compares original predictions, altered predictions, and original ground
    truth of sale price outcomes prior to lightrail introduction.
    '''
    predicted_pre = model.predict(X_pre_lightrail)
    predicted_post = model.predict(X_post_lightrail)
    predictions = pd.DataFrame({'y_true': y_true,
        'y_pred_prelightrail': predicted_pre,
        'y_pred_postlightrail': predicted_post})
    predictions.to_csv(output)
    print("Pre and post-lightrail predictions written to {}".format(output))

def apply_model_to_lightrail(data_with_bbl, X_train_raw, model, model_name,
        output_dir = "data/results",
        bbl_path = "data/subway_bbls/Queens Light Rail BBL.csv"):
    # Apply fitted model to affected properties near the Queens Light Rail
    affected_properties, updated_properties = dc.extract_affected_properties(
        data_with_bbl, bbl_path)

    X_pre_lightrail, X_post_lightrail, y_true = prepare_data(X_train_raw,
        affected_properties, updated_properties)

    output_price_increase = "{}/price_increase_{}.csv".format(
        output_dir, model_name)

    make_prediction(X_pre_lightrail, X_post_lightrail, y_true, model,
        output = output_price_increase)
