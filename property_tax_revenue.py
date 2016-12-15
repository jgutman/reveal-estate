from argparse import ArgumentParser
import pandas as pd
import numpy as np
import final_data_clean as dc
import os


def construct_price_increase(price_increase, data_with_bbl):
    data_with_bbl = dc.drop_cols(data_with_bbl, list())
    affected_properties, _, bbls = dc.extract_affected_properties(
        data_with_bbl, bbl_path)
    affected_properties['bbl'] = bbls
    affected_properties = affected_properties.reset_index(drop=True)
    price_increase = price_increase.merge(affected_properties, on='bbl')
    price_increase = price_increase.drop(['y_true', 'subwaydist'], axis=1)
    return price_increase

def calculate_updated_sale(data):
    data['percent_increase'] = (data.y_pred_postlightrail -
        data.y_pred_prelightrail) / data.y_pred_prelightrail
    data['pred_sale_price'] = data.sale_price * (1 + data.percent_increase)
    return data

def compute_tax_revenue(data, tax_rate = .00796):
    current_tax_income = np.sum(data.sale_price * tax_rate)
    new_tax_income = np.sum(data.pred_sale_price * tax_rate)
    tax_revenue_increase = new_tax_income - current_tax_income
    tax_revenue_increase = round(tax_revenue_increase)
    pct_tax_revenue_increase = 100 * tax_revenue_increase / current_tax_income
    pct_tax_revenue_increase = round(pct_tax_revenue_increase, 2)
    return tax_revenue_increase, pct_tax_revenue_increase

def main():
    parser = ArgumentParser(description =
        "Predict the increase in property tax revenue with proposed lightrail")
    parser.add_argument("--data", dest = "datapath",
        help="Path to data with predicted increase in price per sqft.")
    parser.add_argument("--data", dest = "data_original",
        help="Path to merged data used to train the model.")
    parser.set_defaults(datapath = "data/results/price_increase_rf.csv",
        bbl_path = "data/subway_bbls/QueensLightrail_full1.csv",
        data_original = "{}/{}.csv".format("data/merged",
        "bronx_brooklyn_manhattan_queens_statenisland_2003_2016"))

    args = parser.parse_args()
    datapath, data_original, bbl_path = args.datapath,
        args.data_original, args.bbl_path

    price_increase = pd.read_csv(datapath)
    data_with_bbl = pd.read_csv(data_original, usecols = ['bbl', 'sale_year',
        'sale_price', 'price_per_sqft', 'subwaydist'])
    price_increase = construct_price_increase(price_increase, data_with_bbl)
    price_increase = calculate_updated_sale(price_increase)

    filename = 'affected_properties_tax_calculation.csv'
    price_increase.to_csv(os.path.join('data/results', filename),
        index = False)

    print("Total tax revenue increase and percent increase in tax revenue")
    print(compute_tax_revenue(price_increase))

if __name__ == '__main__':
    main()
