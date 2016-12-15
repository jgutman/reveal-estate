from argparse import ArgumentParser
import pandas as pd
import numpy as np
import final_data_clean as dc
import final_modeling as fm


def main():
    parser = ArgumentParser(description =
        "Predict the increase in property tax revenue with proposed lightrail")
    parser.add_argument("--data", dest = "datapath",
        help="Path to data with predicted increase in price per sqft.")
    parser.add_argument("--data", dest = "data_original",
        help="Path to merged data used to train the model.")
    parser.set_defaults(datapath = "data/results/price_increase_rf.csv",
        bbl_path = "data/subway_bbls/QueensLightrail_full1.csv",
        data_original = \
        "data/merged/bronx_brooklyn_manhattan_queens_statenisland_2003_2016")

    args = parser.parse_args()
    datapath, data_original, bbl_path = args.datapath,
        args.data_original, args.bbl_path

    data = pd.read_csv(datapath)
    data_with_bbl = pd.read_csv(data_path, low_memory = True)
    data_with_bbl = data_with_bbl.loc[['bbl', 'sale_year',
        'sale_price', 'price_per_sqft']]
    affected_properties, _, bbls = dc.extract_affected_properties(
        data_with_bbl, bbl_path)


if __name__ == '__main__':
    main()
