from argparse import ArgumentParser
import pandas as pd
import numpy as np
from convert_xy import convert_df
import math
import pandas as pd
from scipy import spatial
import os

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

    string_cols = ['BBL', 'BldgClass', 'Borough', 'BuiltCode', 'Council',
        'Ext', 'HistDist', 'IrrLotCode', 'Landmark', 'LandUse', 'LotType',
        'LtdHeight', 'OwnerType', 'ProxCode', 'SchoolDist', 'SplitZone']
    string_cols = {a: str for a in string_cols}

    float_cols = ['BldgDepth', 'BldgFront', 'BsmtCode', 'ComArea',
        'Easements', 'LotDepth', 'LotFront', 'NumBldgs', 'NumFloors',
        'ResArea', 'UnitsRes', 'UnitsTotal', 'XCoord', 'YCoord',
        'YearAlter1', 'YearAlter2', 'YearBuilt', 'ZipCode']
    float_cols = {a: np.float64 for a in float_cols}

    int_cols = ['Block', 'BoroCode',  'CondoNo', 'CD']
    int_cols = {a: np.int64 for a in int_cols}

    cols_dtype = dict(string_cols, **float_cols)
    cols_dtype.update(int_cols)

    # Create an empty dataframe to store data as we iterate
    pluto = pd.DataFrame()
    for boro in boros:
        if boro == "si":
            boro = "statenisland"
        assert(boro in initials.keys()), "inappropriate boro for data"

        filename = "{data_dir}/{boro}.csv".format(data_dir = data_dir,
            boro = initials.get(boro))
        data = pd.read_csv(filename, usecols = cols_dtype.keys(),
            dtype = cols_dtype, engine = 'c')
        data.columns = [col.strip().lower() for col in data.columns]
        # Append new rows to existing dataframe
        pluto = pluto.append(data)

    pluto = clean_pluto(pluto, initials)
    return pluto


def clean_pluto(pluto, initials):
    pluto = pluto.reset_index(drop=True)
    # Convert xcoord and ycoord columns to latitude and longitudes
    pluto = convert_df(pluto, 'xcoord', 'ycoord')

    pluto['gross_sqft_pluto'] = pluto.resarea + pluto.comarea
    pluto = pluto.loc[ pluto.gross_sqft_pluto != 0]
    pluto = pluto.loc[ pluto.gross_sqft_pluto.notnull()]

    BinaryDict = {'N':0, 'Y':1, np.nan:0}
    LtdHeightDict = {'LH*':1, np.nan:0}
    BuiltCodeDict = {'E':1, np.nan:0}
    NamedDistDict = {'\w':1, np.nan:0}

    pluto =  pluto.replace({
                    "splitzone": BinaryDict,
                    "irrlotcode": BinaryDict,
                    "ltdheight": LtdHeightDict,
                    "builtcode": BuiltCodeDict,
                    "histdist": NamedDistDict,
                    "landmark": NamedDistDict},
                    regex = True)

    # Building Class - Use first character of BldgClass only
    pluto.bldgclass = [x[0].upper() for x in pluto.bldgclass.astype(str)]

    # Ext New Columns
    pluto['garage'] = pluto.ext.replace({'G':1, 'EG':1, 'E':0, np.nan:0})
    pluto['extension'] = pluto.ext.replace({'G':0, 'EG':1, 'E':1, np.nan:0})

    # Count Alterations
    pluto = binarize(pluto, 'yearalter1')
    pluto = binarize(pluto, 'yearalter2')
    pluto['countalter'] = pluto.yearalter1 + pluto.yearalter2
    pluto = pluto.drop(['ext', 'yearalter1', 'yearalter2'], axis=1)

    # Round NumFloors
    pluto.numfloors = pluto.numfloors.round()
    # Easements Binary
    pluto = binarize(pluto, 'easements')
    # ProxCode set NaN, BsmtCode Binary
    pluto = pluto.replace({"proxcode": {0:np.nan},
        "bsmtcode": {2:1, 3:1, 4:1, 5:0}})

    # Limit NumBldgs
    pluto = censor(pluto, 'numbldgs', 10)
    # Limit Front and Depth
    pluto = censor(pluto, 'lotfront', 100)
    pluto = censor(pluto, 'lotdepth', 200)
    pluto = censor(pluto, 'bldgfront', 100)
    pluto = censor(pluto, 'bldgdepth', 200)
    # Fix impossible years
    pluto = censor(pluto, 'yearbuilt', 2017)
    # Limit UnitRes and UnitsTotal
    pluto = censor(pluto, 'unitsres', 100)
    pluto = censor(pluto, 'unitstotal', 100)

    pluto = merge_population_data(pluto, initials)
    return pluto

def merge_population_data(pluto, initials,
        data_dir = "data/open_nyc",
        filename = "l_population_by_community_district.csv"):
    """
    Merge pluto with community district level population info by decade.
    """
    population = pd.read_csv(os.path.join(data_dir, filename))
    population.columns = [col.strip().lower() for col in population.columns]
    population.columns = [col.replace(" ", "_") for col in population.columns]
    pluto["cd_number"] = pd.Series([cd % 100 for cd in pluto.cd],
        index = pluto.index)
    inv_initials = {v: k for k, v in initials.items()}
    pluto = pluto.replace({'borough': inv_initials})
    population.borough = [boro.replace(" ", "").lower() for boro in
        population.borough]
    pluto = pluto.merge(population, how='left', on = ['borough', 'cd_number'])
    return pluto


def censor(data, var, upper_limit, lower_limit = 0):
    check_limits = (data[var] >= upper_limit) | (data[var] <= upper_limit)
    data.loc[check_limits, var] = np.nan
    return data


def binarize(data, var):
    data.loc[data[var] > 0, var] = 1
    data.loc[data[var] < 0, var] = np.nan
    return(data)


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
    # Convert column names to lowercase
    # Replaces spaces in column names with underscores
    data.columns = [col.strip().lower().replace(" ", "_")
            for col in data.columns]
    return data


def add_BBL(processed_data):
    """
    Takes a raw dataframe and adds the BBL code (Borough, Block, Lot)
    Args:
        Pandas DataFrame data: raw data frame to append the "bbl" and
    Returns:
        Pandas DataFrame
    """
    processed_data = processed_data.reset_index(drop = True)
    # Extract the borough, block, and lot, and create a 10-digit code
    # zero-padded code from these three columns in order
    bbl_columns = processed_data[["borough", "block", "lot"]].itertuples()
    bbl_formatted = pd.Series(["%01d%05d%04d" %
        (row.borough, row.block, row.lot) for row in bbl_columns],
        index = processed_data.index, dtype='int64')
    processed_data["bbl"] = bbl_formatted.astype(str)
    return processed_data


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
            boro_year.dropna(how = 'all', inplace = True)
            finance = finance.append(boro_year)
    finance = add_BBL(finance)
    finance = finance.loc[:, ['block', 'bbl', 'sale_price',  'sale_date',
        'tax_class_at_time_of_sale', 'residential_units', 'commercial_units',
        'total_units']]
    return finance


def read_in_dtm(boros, data_dir = 'data/dtm',
        filename = 'DTM_0316_Condo_Units.csv'):
    """
    Reads in the Digital Tax Map dataset and returns a dataframe with mapping
    from borough and condo number to unit BBL for the specified boroughs.

    Args:
        list(string) boros: list of all the boroughs to pull dtm data for
        string data_dir: a relative path as a string to folder containing the
            dtm data in csv format
        string filename: the name of the file containing the dtm condo unit data
    Returns:
        Pandas DataFrame
    """
    columns = ['CONDO_BORO', 'CONDO_NUMB', 'UNIT_BLOCK',
               'UNIT_LOT', 'UNIT_BBL', 'UNIT_DESIG']
    boro_names = ['manhattan', 'bronx', 'brooklyn', 'queens', 'statenisland']
    boro_codes = dict(zip(boro_names, range(1,6)))
    dtm = pd.read_csv(os.path.join(data_dir, filename), usecols=columns)
    dtm.dropna(how = 'all', inplace = True)
    dtm.columns = [col.strip().lower() for col in dtm.columns]
    dtm = dtm.dropna(subset = ['unit_bbl', 'condo_boro', 'condo_numb'])
    dtm.unit_bbl = dtm.unit_bbl.astype(int).astype(str)
    dtm = dtm.loc[dtm.condo_boro.isin(
        [boro_codes.get(boro) for boro in boros])]
    return dtm


def subset_data(data, cols_to_keep):
    data = data[cols_to_keep]
    data = data.drop_duplicates()
    return data


def get_finance_condo_lot(pluto, finance, dtm):
    """
    Takes a finance dataset with unit lot BBL numbers and constructs a non-unit
    BBL column that corresponds to the BBL codes listed in the PLUTO data.

    Args:
        Pandas DataFrame pluto: contains PLUTO data and "bbl" join key
            (lot shared by all condo units in a building)
        Pandas DataFrame finance: contains finance data and "bbl" join key
            (unit-level lot numbers distinct for each condo unit)
        Pandas DataFrame dtm: contains "unit_bbl" and "condo_numb" for
            joining pluto and dept. of finance condo unit data
    Returns:
        Pandas DataFrame
    """
    dtm_cols_to_keep = ['unit_bbl', 'condo_boro', 'condo_numb']
    pluto_cols_to_keep = ['bbl', 'block', 'borocode', 'condono']
    finance_cols_to_keep = ['bbl', 'block']

    finance_condos_only = pd.merge(subset_data(finance, finance_cols_to_keep),
        subset_data(dtm, dtm_cols_to_keep),
        how='inner', left_on=['bbl'], right_on=['unit_bbl'])

    # for condos: finance.bbl == finance_condos_only.unit_bbl
    #             finance_condos_only.bbl_pluto == pluto.bbl
    finance_condos_only = pd.merge(subset_data(pluto, pluto_cols_to_keep),
        finance_condos_only, how='inner',
        left_on=['borocode', 'block', 'condono'],
        right_on=['condo_boro', 'block', 'condo_numb'],
        suffixes=['_pluto', '_finance'])

    # drop all other columns except bbl_pluto to bbl_finance
    # drop any rows where the first 6 digits (borough + block) do not match
    finance_condos_only = subset_data(finance_condos_only,
        ['bbl_pluto', 'bbl_finance'])
    finance_condos_only = finance_condos_only.loc[lambda df:
            # np.floor(df.bbl_pluto / 1e4) == np.floor(df.bbl_finance / 1e4)]
            [x[0:6] == y[0:6] for x,y in zip(df.bbl_finance, df.bbl_pluto)]]

    # get a list of bbls that are not condos (same in pluto and finance)
    standard_bbls = list(set(finance.bbl).difference(
                        set(finance_condos_only.bbl_finance)))
    # combine condo bbls that differ with standard bbls that are the same
    bbl_mappings = finance_condos_only.append(pd.DataFrame.from_dict(
        {'bbl_pluto': standard_bbls, 'bbl_finance': standard_bbls}))
    bbl_mappings = bbl_mappings.reset_index(drop = True)

    finance_condo_updated = pd.merge(finance, bbl_mappings,
        how='left', left_on='bbl', right_on='bbl_finance')
    # finance condo updated: remove bbl/block/bbl_finance
    # retain only bbl_pluto to match with pluto.bbl in merge
    finance_condo_updated = finance_condo_updated.drop(
        ['bbl', 'block', 'bbl_finance'], axis=1) # bbl, block, bbl_finance

    # Remove duplicate bbls by returning only the most recent sales data
    # for each BBL and year
    finance_condo_updated = finance_condo_updated.reset_index(drop = True)
    finance_condo_updated["sale_year"] = [d.year for d in
        finance_condo_updated.sale_date]
    grouped = finance_condo_updated.groupby(['bbl_pluto', 'sale_year'])
    max_idx_by_bbl = grouped['sale_price'].idxmax().values
    finance_condo_updated = finance_condo_updated.loc[max_idx_by_bbl]
    return finance_condo_updated


def merge_pluto_finance(pluto, finance, dtm):
    """
    Performs an outer join on PLUTO and Dept of Finance data using BBL as the
    join key, returning a single dataframe. Also writes merged output to file.

    Args:
        Pandas DataFrame pluto: contains PLUTO data and "bbl" join key
        Pandas DataFrame finance: contains finance data and "bbl" join key
        Pandas DataFrame dtm: contains "unit_bbl" and "condo_numb" for
            joining pluto and dept. of finance condo unit data
    Returns:
        Pandas DataFrame
    """
    # First search for finance sales data that matches dtm condo data
    print("Updating lot numbers for condo units")
    print("Finance:{} PLUTO:{} DTM:{}".format(finance.shape,
        pluto.shape, dtm.shape))
    finance_condo_updated = get_finance_condo_lot(pluto = pluto,
        finance = finance, dtm = dtm)
    print("Finance updated:{}".format(finance_condo_updated.shape))
    print("Merging PLUTO with updated Dept. of Finance data")
    buildings = pd.merge(pluto, finance_condo_updated, how='right',
        left_on='bbl', right_on = 'bbl_pluto',
        suffixes=['_pluto', '_finance'])
    buildings["price_per_sqft"] = (buildings.sale_price.astype('float64') /
        buildings.gross_sqft_pluto)
    buildings = buildings[ buildings.price_per_sqft.notnull()]
    buildings = buildings[ buildings.price_per_sqft > 0.]
    return buildings


def bbl_dist_to_subway(data,
        filepath = "data/open_nyc/subwaydist.csv"):
    subwaydist = pd.read_csv(filepath)
    subwaydist = subwaydist.drop(
        ['latitude','longitude'], axis = 1) #zipcode
    return data.merge(subwaydist, how = 'left', on = ['bbl'])


def bbl_dist_to_open_NYC_data(data,
        filepath = "data/open_nyc/distance_metrics.csv"):
    other_distances = pd.read_csv(filepath)
        #other_distances = other_distances.drop(
        #['latitude', 'longitude'], axis = 1) #zipcode
    return data.merge(other_distances, how = 'left', on = ['bbl'])


def make_dummy_variables(dataframe, feature):
    """
    Creates dummy columns for a categorical variable in given Pandas dataframe.

    Args:
        dataframe: Pandas dataframe.
        feature: a categorical variable.
    Returns:
        dummy variable columns for input feature.
    """
    uniques = dataframe[feature].unique()
    lst = list(uniques)
    for x in lst:
        if isinstance(x,float) and math.isnan(x):
            lst.remove(x)
    if np.nan in lst:
        lst.remove(np.nan)
    uniques = sorted(lst)
    dummies = pd.get_dummies(dataframe[feature])
    colnames = ['{}_{}'.format(str(feature), str(x)) for x in uniques]
    dummies.columns = colnames
    dummies.drop(colnames[-1], axis=1, inplace=True)
    return dummies


def clean_categorical_vars(dataframe, list_of_cat_vars, boros, years):
    """Append the original dataframe with output of make_dummy_variables.
    This function also adds a binary column for each feature that has missing values.
    This binary column is 1 if value is missing.
        Args:
            dataframe: Pandas dataframe.
            list (string) list_of_cat_vars: list of categorical column names
            list (string) boros: list of boros in dataframe.
            list (int) years: list of years in dataframe.
    """
    for column in dataframe.columns:
        if (np.any(dataframe[column].isnull())):
            dataframe[column + "_mv"] = dataframe[column].isnull().astype(int)
    for var in list_of_cat_vars:
        dummies = make_dummy_variables(dataframe, var)
        dataframe = pd.concat([dataframe, dummies], axis=1)
        dataframe = dataframe.drop(var, axis=1)
    write_output(dataframe, boros, years)
    return dataframe


def write_output(data, boros, years, output_dir = 'data/merged'):
    boros.sort()
    output = "{output_dir}/{boros_joined}_{min_year}_{max_year}.csv".format(
        boros_joined = "_".join(boros), min_year = min(years),
        max_year = max(years), output_dir = output_dir)
    print("Writing output to file in {}".format(output))
    data.to_csv(output, index = False, chunksize=1e4)


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
    if years == ["all"]:
        years = list(range(2003, 2017))
    if boros == ["all"]:
        boros = ["manhattan", "brooklyn", "queens", "bronx", "statenisland"]

    # Convert to lowercase and remove spaces in borough names
    boros = ["".join(boro.lower().split()) for boro in boros]

    print("Getting PLUTO data for: {}".format(boros))
    pluto = read_in_pluto(boros)
    print("Getting Finance data for: {} and {}".format(boros, years))
    finance = read_in_finance(boros, years)
    print("Getting DTM Condo Unit data for: {}".format(boros))
    dtm = read_in_dtm(boros)
    print("Beginning merge...")
    buildings = merge_pluto_finance(pluto, finance, dtm)

    print("Merging with subway data")
    buildings = bbl_dist_to_subway(buildings)
    print("Merging with Open NYC distances")
    buildings = bbl_dist_to_open_NYC_data(buildings)

    final_cols_to_remove = ['bbl_pluto', 'borocode', 'block', 'condono',
        'sale_year', 'xcoord', 'ycoord']
    buildings = buildings.drop(final_cols_to_remove, axis=1)
    cat_vars = ['borough', 'schooldist', 'council', 'bldgclass', 'landuse',
        'ownertype', 'proxcode', 'lottype', 'tax_class_at_time_of_sale']
    buildings_with_cats = clean_categorical_vars(
        buildings, cat_vars, boros, years)


if __name__ == '__main__':
    main()
