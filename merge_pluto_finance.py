from argparse import ArgumentParser
import pandas as pd
import numpy as np

def read_in_boro_year_data(boro, year):
    """
    Fetches data file for a specified boro and year, and returns the data as a Pandas dataframe.

    Args:
        string boro: name of boro for desired data
        int year: year of desired data
    Returns:
        Pandas dataframe
    """
    # Format input arguments appropriately
    try:
        year = int(year)
    except TypeError:
        print("inappropriate year for data")
    if year < 100:
        year = year + 2000
    assert(year in years), "inappropriate year for data"
    if boro == "si":
        boro = "statenisland"
    assert(boro in boros), "inappropriate boro for data"

    # Reads in Excel file skipping appropriate number of junk rows at the beginning
    filename = 'data/finance_sales/{year}_{boro}.xls'.format(year = year, boro = boro)
    skip_rows = 4 if year > 2010 else 3
    data = pd.read_excel(filename, skiprows = skip_rows)
    # Remove newline characters from column headers
    data.columns = [col.strip().lower() for col in data.columns]
    return data

def add_BBL_and_price_per_ft(data, copy = True):
    """
    Takes a raw dataframe and adds the BBL code (Borough, Block, Lot), and price per square foot.
    Uses same 10-digit BBL format as PLUTO: 1 digit for Borough, 5 digits for Block, 4 digits for Lot.

    Args:
        Pandas data: raw data frame to append BBL and PRICESQFT columns
        boolean copy: whether to make a copy or alter the dataframe in place
    Returns:
        Pandas dataframe
    """
    # copy the data frame to a new object if desired
    if copy:
        processed_data = data.copy()
    else:
        processed_data = data

    # extract the borough, block, and lot, and create a 10-digit code zero-padded code from these
    bbl_columns = data[["borough", "block", "lot"]].itertuples()
    bbl_formatted = ["%01d%05d%04d" % (row.borough, row.block, row.lot) for row in bbl_columns]
    processed_data["bbl"] = bbl_formatted
    processed_data["price per sqft"] = data["sale price"] / data["gross square feet"]
    return processed_data

def read_in_pluto(boros, data_dir = "data/nyc_pluto_16v1"):
    """
    """

def read_in_finance(boros, years, data_dir = "data/finance_sales"):
    """
    """

def merge_pluto_finance(pluto, finance):
    """
    """

def main():
    # set up input option parsing for years and boros to pull data for
    parser = ArgumentParser(description =
        "Subset the PLUTO and Dept of Finance data to be merged")
    parser.add_argument("--year", dest="years", nargs="*",
        help="Adds a year to the list of years to pull sales data for")
    parser.add_argument("--borough", dest="boros", nargs="*",
        help="Adds a borough to the list to pull sales/pluto data for")
    parser.set_defaults(years = [2014,2015],
        boros = ["brooklyn, manhattan"])
    args = parser.parse_args()
    years, boros = args.years, args.boros

if __name__ == '__main__':
    main()
