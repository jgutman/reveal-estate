import pickle, os
import predict_price_increase as ppi
import final_modeling as fm

data_path = "data/merged/bronx_brooklyn_manhattan_queens_statenisland_2003_2016.csv"
pkl_dir = "data/results/pkl_models/"
full_filename = "gb_25.pkl"

data_with_bbl = fm.get_data_for_model(data_path)
data = data_with_bbl.drop('bbl', axis=1)

with open(os.path.join(pkl_dir, full_filename), 'rb') as model:
    model_pkl = pickle.load(model)

X_train_raw, X_train, X_test, y_train, y_test = fm.preprocess_data(data)
train_bbl = data_with_bbl['bbl'].loc[y_train.index]
test_bbl = data_with_bbl['bbl'].loc[y_test.index]

output_dir = "data/results"
model_name = "gb"

ppi.apply_model_to_lightrail(data_with_bbl, X_train_raw, model_pkl,
    model_name, output_dir = output_dir)
