import pickle, os
import predict_price_increase as ppi
import final_modeling as fm
import property_tax_revenue as tax

data_dir = "data/merged"
data_filename = "bronx_brooklyn_manhattan_queens_statenisland_2003_2016.csv"
data_path = os.path.join(data_dir, data_filename)
bbl_path = "data/subway_bbls/QueensLightrail_full1.csv"
pkl_dir = "data/results/pkl_models/"
pkl_filename = "gb_25.pkl"
output_dir = "data/results"
model_name = "gb"

data_with_bbl = fm.get_data_for_model(data_path)
data = data_with_bbl.drop('bbl', axis=1)

with open(os.path.join(pkl_dir, pkl_filename), 'rb') as model:
    model_pkl = pickle.load(model)

X_train_raw, X_train, X_test, y_train, y_test = fm.preprocess_data(data)
train_bbl = data_with_bbl['bbl'].loc[y_train.index]
test_bbl = data_with_bbl['bbl'].loc[y_test.index]

price_increase = ppi.apply_model_to_lightrail(data_with_bbl, X_train_raw,
    model_pkl, model_name, output_dir = output_dir)

data_with_bbl = pd.read_csv(data_path, usecols = ['bbl', 'sale_year',
    'sale_price', 'price_per_sqft', 'subwaydist'])

price_increase = tax.construct_price_increase(price_increase,
    data_with_bbl, bbl_path)
price_increase = tax.calculate_updated_sale(price_increase)

filename = 'affected_properties_tax_calculation_{}.csv'.format(model_name)
price_increase.to_csv(os.path.join(output_dir, filename), index = False)

print(tax.compute_tax_revenue(price_increase))
