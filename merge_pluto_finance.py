from argparse import ArgumentParser
import pandas as pd
import numpy as np

def read_in_boro_year_data(boro, year, data_dir = "data/finance_sales"):
    """
    Fetches data file for a specified boro and year, and returns the data as a
    Pandas dataframe. Checks integrity of boro/year arguments.

    Args:
        string boro: name of boro for desired data
        int year: year of desired data
    Returns:
        Pandas DataFrame
    """
    # Acceptable inputs
    boros = ['manhattan', 'bronx', 'brooklyn', 'queens', 'statenisland']
    years = range(2003, 2017)

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

    # Reads in Excel file skipping appropriate number of junk rows at the
    # beginning of file, keeping the header row as a header
    filename = "{data_dir}/{year}_{boro}.xls".format(data_dir = data_dir,
        year = year, boro = boro)
    skip_rows = 4 if year > 2010 else 3
    data = pd.read_excel(filename, skiprows = skip_rows)
    # Remove newline characters from column headers
    data.columns = [col.strip().lower() for col in data.columns]
    return data


def add_BBL_and_price_per_ft(data, copy = True):
    """
    Takes a raw dataframe and adds the BBL code (Borough, Block, Lot), and
    price per square foot. Uses same 10-digit BBL format as PLUTO:
    1 digit for Borough, 5 digits for Block, 4 digits for Lot.

    Args:
        Pandas DataFrame data: raw data frame to append the "bbl" and "price
            per sqft" columns to
        boolean copy: whether to make a copy or alter the dataframe in place
    Returns:
        Pandas DataFrame
    """
    # Copy the data frame to a new object if desired
    if copy:
        processed_data = data.copy()
    else:
        processed_data = data

    # Extract the borough, block, and lot, and create a 10-digit code
    # zero-padded code from these three columns in order
    bbl_columns = data[["borough", "block", "lot"]].itertuples()
    bbl_formatted = pd.Series(["%01d%05d%04d" % (row.borough, row.block,
        row.lot) for row in bbl_columns], dtype='int64')
    processed_data["bbl"] = bbl_formatted
    # Use the larger: gross sqft or land sqft, gross sqft may be zero sometimes
    processed_data["price per sqft"] = data["sale price"].astype('float64') / data[[
        "gross square feet", "land square feet"]].max(axis=1)
    # Replace 0s and infinity with NaN for clarity
    processed_data = processed_data.replace(
        {'price per sqft': {0: np.nan, np.inf: np.nan}})
    return processed_data


def read_in_pluto(boros, data_dir = "data/nyc_pluto_16v1"):
    """
    Takes a list of boroughs and extracts PLUTO data for each borough,
    appending each subset to create a single data frame for all boroughs.

    Args:
        list(string) boros: list of all the boroughs to pull pluto data for
        string data_dir: a relative path as a string to folder containing the
            PLUTO data for all boroughs
    Returns:
        Pandas DataFrame
    """
    # mapping of how boroughs are referred in PLUTO filenames
    initials = {"manhattan" : "MN", "brooklyn" : "BK", "bronx" : "BX",
        "queens" : "QN", "statenisland" : "SI"}

    # Create an empty dataframe to store data as we iterate
    pluto = pd.DataFrame()
    for borough in boros:
        filename = "{data_dir}/{boro}.csv".format(data_dir = data_dir,
            boro = initials.get(borough))
        data = pd.read_csv(filename, low_memory = False)
        data.columns = [col.strip().lower() for col in data.columns]
        # Append new rows to existing dataframe
        pluto = pluto.append(data)
    return pluto


def read_in_finance(boros, years, data_dir = "data/finance_sales"):
    """
    Takes a list of boroughs and years and extracts finance data for each year,
    appending each subset to create a single data frame for all years/boroughs.

    Args:
        list(string) boros: list of all the boroughs to pull finance data for
        list(int) years: list of all the years to pull finance data for
        string data_dir: a relative path as a string to folder containing the
            department of finance sales price data for all boroughs
    Returns:
        Pandas DataFrame
    """
    # Create an empty dataframe to store data as we iterate
    finance = pd.DataFrame()
    for year in years:
        for borough in boros:
            print("Pulling Finance data for {}_{}".format(year, borough))
            boro_year = read_in_boro_year_data(borough, year, data_dir)
            boro_year = add_BBL_and_price_per_ft(boro_year)
            # Append new rows to existing dataframe
            finance = finance.append(boro_year)
    return finance


def merge_pluto_finance(pluto, finance, boros, years,
    output_dir = "data/merged"):
    """
    Performs an outer join on PLUTO and Dept of Finance data using BBL as the
    join key, returning a single dataframe. Also writes merged output to file.

    Args:
        Pandas DataFrame pluto: contains PLUTO data and "bbl" join key
        Pandas DataFrame finance: contains finance data and "bbl" join key
        list(string) boros: list of boroughs to use in filename of merged data
        list(int) years: list of years to use in filename of merged data
        string output_dir: directory to store merged output data
     Returns:
        Pandas DataFrame
    """
    buildings = pd.merge(pluto, finance, how='right', on='bbl',
        suffixes=['_pluto', '_finance'])
    output = "{output_dir}/{boros_joined}_{min_year}_{max_year}.csv".format(
        boros_joined = "_".join(boros), min_year = min(years),
        max_year = max(years), output_dir = output_dir)
    buildings.to_csv(output, index = False)
    return buildings


def main():
    # Set up input option parsing for years and boros to pull data for
    parser = ArgumentParser(description =
        "Subset the PLUTO and Dept of Finance data to be merged")
    parser.add_argument("--year", dest="years", nargs="*",
        help="Adds a year to the list of years to pull sales data for. Possible years include 2003-2016 (2016 is rolling data through September).")
    parser.add_argument("--borough", dest="boros", nargs="*",
        help="Adds a borough to the list to pull sales/pluto data for. Possible boroughs include Brooklyn, Bronx, StatenIsland (as 1 or 2 words, or SI), Queens, and Manhattan. Not case sensitive.")
    parser.set_defaults(years = [2014, 2015],
        boros = ["brooklyn", "manhattan"])
    args = parser.parse_args()
    years, boros = args.years, args.boros

    # Convert to lowercase and remove spaces in borough names
    boros = ["".join(boro.lower().split()) for boro in boros]

    print("Getting PLUTO data for: {}".format(boros))
    pluto = read_in_pluto(boros)
    print("Getting Finance data for: {} and {}".format(boros, years))
    finance = read_in_finance(boros, years)
    print("Merging and outputting data")
    buildings = merge_pluto_finance(pluto, finance, boros, years)


if __name__ == '__main__':
    main()
