from argparse import ArgumentParser
import pandas as pd
import numpy as np
import final_data_clean as dc


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
    affected_properties, _, bbls = dc.extract_affected_properties(
        data_with_bbl, bbl_path)
    affected_properties['bbl'] = bbls


if __name__ == '__main__':
    main()
