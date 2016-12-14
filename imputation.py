from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, MinMaxScaler

def impute_missing_values(X_train, X_test, strategy):
    imputer = Imputer(strategy = strategy)
    imputer.fit(train) # fit the imputer on the training mean/median
    train = pd.DataFrame(imputer.transform(train), # returns a numpy array
        columns = train.columns, index = train.index) # back to dataframe
    test = pd.DataFrame(imputer.transform(test),
        columns = test.columns, index = test.index)
    return train, test

def scale_features(X_train, X_test, strategy):
    scaler = StandardScaler() if strategy == 'standard'
        else RobustScaler()

    num_values_by_column = {x: len(train[x].unique()) for x in train.columns}
    zero_variance_columns = [k for k,v in num_values_by_column.items()
        if v == 1]
    train.drop(zero_variance_columns, axis=1, inplace=True)
    test.drop(zero_variance_columns, axis=1, inplace=True)

    non_binary_columns = [k for k, v in num_values_by_column.items()
        if v > 2]
    train_non_binary = train[non_binary_columns]
    test_non_binary = test[non_binary_columns]
    scaler.fit(train_non_binary)
    train_non_binary = pd.DataFrame(scaler.transform(train_non_binary),
        columns = non_binary_columns, index = train.index)
    train_scaled = train.drop(non_binary_columns, axis=1)
    train_scaled = train_scaled.merge(train_non_binary,
        left_index=True, right_index=True)

    test_non_binary = pd.DataFrame(scaler.transform(test_non_binary),
        columns = non_binary_columns, index = test.index)
    test_scaled = test.drop(non_binary_columns, axis=1)
    test_scaled = test_scaled.merge(test_non_binary,
        left_index=True, right_index=True)
    return train_scaled, test_scaled

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

def main():
    X_train_raw, X_train, X_test, y_train, y_test = fm.preprocess_data(data)
    column_set = list(X_train.columns)

    X_pre_lightrail, X_post_lightrail, y_true = prepare_data(X_train_raw,
        affected_properties, updated_properties)

    X_pre_lightrail = X_pre_lightrail.filter(column_set)
    X_post_lightrail = X_post_lightrail.filter(column_set)
