from argparse import ArgumentParser
import pandas as pd
import numpy as np
from convert_xy import convert_df
import math
import pandas as pd
from scipy import spatial

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


def add_BBL(data, copy = True):
    """
    Takes a raw dataframe and adds the BBL code (Borough, Block, Lot)
    Args:
        Pandas DataFrame data: raw data frame to append the "bbl" and 
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

    # Remove duplicate bbls by returning only the most recent sales data
    # for each BBL and year
    processed_data["sale_year"] = [d.year for d in processed_data.sale_date]
    max_idx_by_bbl = processed_data.groupby([
        'bbl', 'sale_year'])['sale_date'].idxmax()
    processed_data = processed_data.loc[max_idx_by_bbl]
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
        #Need to keep 'borocode', 'block' for dtm merge, but remove the rest of unneeded columns
        columns_to_remove = ['lot','zonedist1','zonedist2', 'zonedist3', 'zonedist4', 'overlay1', 'overlay2',
        'spdist1', 'spdist2', 'allzoning1', 'allzoning2','ownername', 'lotarea', 'bldgarea', 
        'officearea', 'retailarea', 'garagearea', 'strgearea', 'factryarea','otherarea', 'areasource',
        'assessland', 'assesstot', 'exemptland', 'exempttot','builtfar', 'residfar', 'commfar', 'facilfar',
        'zmcode','sanborn', 'taxmap', 'edesignum', 'appbbl', 'appdate', 'plutomapid','address','version',
        'ct2010','cb2010','sanitboro','tract2010', 'cd','firecomp','policeprct','healtharea',
        'sanitdistrict','sanitsub']
    for col in columns_to_remove:
        pluto = pluto.drop(col,axis=1)
    # Convert xcoord and ycoord columns to latitude and longitudes
    pluto = convert_df(pluto, "xcoord", "ycoord")
    columns_to_float =['schooldist', 'council', 'zipcode','comarea', 'resarea',
       'landuse', 'easements','numbldgs', 'numfloors', 'unitsres', 'unitstotal', 'lotfront',
       'lotdepth', 'bldgfront', 'bldgdepth', 'proxcode','lottype', 'bsmtcode', 'yearbuilt',
       'yearalter1', 'yearalter2', 'bbl','condono', 'xcoord', 'ycoord']
    for col in columns_to_float:
        pluto[col] = pluto[col].astype(float)
    pluto['gross_sqft_pluto'] = pluto['resarea'] + pluto['comarea']
    pluto = pluto[pluto['gross_sqft_pluto']!=0]
    pluto = pluto[pluto['gross_sqft_pluto']!= np.nan]
    BinaryDict = {'N': 0, 'Y': 1}
    # Split Zone Binary
    pluto.replace({"splitzone": BinaryDict},inplace=True)
    # IrrLotCode Binary
    pluto.replace({"irrlotcode": BinaryDict},inplace=True)
    # Limited Height Binary
    pluto['ltdheight'] = pluto['ltdheight'].fillna(str(0))
    LtdHeightDict = {'LH-1': 1, 'LH-1A': 1, '0': 0}
    pluto.replace({"ltdheight": LtdHeightDict},inplace=True)
    # BuiltCode Binary
    BuiltCodeDict = {'E': 1}
    pluto.replace({"builtcode": BuiltCodeDict},inplace=True)
    pluto['builtcode'].fillna(value=0,inplace=True)
    # Hist Dist Binary
    pluto['histdist'] = (pluto['histdist'].notnull())*1
    pluto['histdist'].fillna(value=0,inplace=True)
    # Landmark
    pluto['landmark'] = (pluto['landmark'].notnull())*1
    pluto['landmark'].fillna(value=0,inplace=True)
    # Ext New Columns
    pluto['garage'] = (pluto['ext']==('G' or 'EG'))*1
    pluto['extension'] = (pluto['ext']==('E' or 'EG'))*1
    pluto = pluto.drop('ext',axis=1)
    # Count Alterations 
    pluto['yearalter1'] = (pluto['yearalter1'] > 0)*1
    pluto['yearalter2'] = (pluto['yearalter2'] > 0)*1
    pluto['countalter'] = pluto['yearalter1'] + pluto['yearalter2']
    pluto = pluto.drop('yearalter1',axis=1)
    pluto = pluto.drop('yearalter2',axis=1)
    # Round NumFloors and Log
    pluto['numfloors'] = pluto['numfloors'].astype(float).round()
    # Easements Binary
    pluto['easements'] = (pluto['easements']>0)*1
    # ProxCode set NaN
    pluto['proxcode'] = pluto['proxcode'].replace(0,np.nan)
    # BsmtCode Binary
    pluto['bsmtcode'] = pluto['bsmtcode'].replace(5,np.nan)
    pluto['bsmtcode'] = (pluto['bsmtcode'] > 0)*1
    # Limit NumBldgs
    pluto['numbldgs'] = ((pluto['numbldgs']<10)*1).replace(0,np.nan)* pluto['numbldgs']
    # Limit Front and Depth
    pluto['lotfront'] = ((pluto['lotfront']<100)*1).replace(0,np.nan)* pluto['lotfront']
    pluto['lotdepth'] = ((pluto['lotdepth']<200)*1).replace(0,np.nan)* pluto['lotdepth']
    pluto['bldgfront'] = ((pluto['bldgfront']<100)*1).replace(0,np.nan)* pluto['bldgfront']
    pluto['bldgdepth'] = ((pluto['bldgdepth']<200)*1).replace(0,np.nan)* pluto['bldgdepth']
    # Fix impossible years
    pluto['yearbuilt'] = ((pluto['yearbuilt']<2016)*1).replace(0,np.nan)* pluto['yearbuilt']
    # Limit UnitRes and UnitsTotal
    pluto['unitsres'] = ((pluto['unitsres']<100)*1).replace(0,np.nan)* pluto['unitsres']
    pluto['unitstotal'] = ((pluto['unitstotal']<100)*1).replace(0,np.nan)* pluto['unitstotal']
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
            boro_year = add_BBL(boro_year)
            # Append new rows to existing dataframe
            finance = finance.append(boro_year)
            finance = finance.append(boro_year)
            finance = finance[['sale_price','sale_date','tax_class_at_time_of_sale','year_built','residential_units', 'commercial_units', 'total_units','block','bbl']]
    #finance = finance[finance['sale_price'] != 0]
    return finance


def read_in_dtm(boros, data_dir = 'data/',
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
    dtm = pd.read_csv('{}/{}'.format(data_dir, filename), usecols=columns)
    dtm = dtm[dtm.CONDO_BORO.isin([boro_codes.get(boro) for boro in boros])]
    dtm.columns = [col.strip().lower() for col in dtm.columns]
    return dtm


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

    print("Finance:{} DTM:{}".format(finance.shape, dtm.shape))
    finance_condos_only = pd.merge(finance, dtm[dtm_cols_to_keep],
        how='inner', left_on=['bbl'], right_on=['unit_bbl'])

    print("Finance intermediate:{} PLUTO:{}".format(finance_condos_only.shape,
        pluto.shape))
    finance_condos_only = pd.merge(pluto[pluto_cols_to_keep],
        finance_condos_only, how='inner',
        left_on=['borocode', 'block', 'condono'],
        right_on=['condo_boro', 'block', 'condo_numb'],
        suffixes=['_pluto', '_finance'])
    print("Finance intermediate (all condos): {}".format(
        finance_condos_only.shape))

    finance_condo_updated = pd.merge(finance,
        finance_condos_only[['bbl_pluto', 'unit_bbl']],
        how='left', left_on='bbl', right_on='unit_bbl')
    finance_condo_updated = finance_condo_updated.drop(['block','bbl'], axis=1)
    
    return finance_condo_updated


def bbl_dist_to_subway(data):
    subwaydist = pd.read_csv("data/subwaydist.csv")
    subwaydist = subwaydist.drop(['latitude','longitude'], axis = 1)
    return data.merge(subwaydist,how='left',on='bbl_pluto')



def merge_pluto_finance(pluto, finance, dtm):
    """
    Performs an outer join on PLUTO and Dept of Finance data using BBL as the
    join key, returning a single dataframe. Also writes merged output to file.

    Args:
        Pandas DataFrame pluto: contains PLUTO data and "bbl" join key
        Pandas DataFrame finance: contains finance data and "bbl" join key
        Pandas DataFrame dtm: contains "unit_bbl" and "condo_numb" for
            joining pluto and dept. of finance condo unit data
        
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
    buildings["price_per_sqft"] = buildings["sale_price"].astype('float64') / buildings["gross_sqft_pluto"]
    buildings = buildings.dropna(how = 'any',subset = ['price_per_sqft'])
    buildings = buildings[buildings["price_per_sqft"] != 0.0]
    #buildings = buildings[buildings["price_per_sqft"] >= 5]
    #buildings = buildings[buildings["price_per_sqft"] <= 5000]
    return buildings


def make_dummy_variables(dataframe, feature):
    """Creates dummy columns for a categorical variable in given Pandas dataframe.
    
        Args:  
            dataframe: Pandas dataframe.
            feature: a categorical variable.
        Returns: dummy variable columns for input feature.
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

def clean_categorical_vars(dataframe, list_of_cat_vars, boros, years, output_dir='data/merged'):
    '''Append the original dataframe with output of make_dummy_variables. This function also adds a binary column for each feature that has missing values. This binary column is 1 if value is missing.
        Args:
            dataframe: Pandas dataframe.
            list (string) list_of_cat_vars: list of categorical column names
            list (string) boros: list of boros in dataframe.
            list (int) years: list of years in dataframe. 
            
    '''
    for column in dataframe.columns:
        if (np.any(dataframe[column].isnull())):
            dataframe[column + "_mv"] = dataframe[column].isnull().astype(int)
    for var in list_of_cat_vars:
        dummies = make_dummy_variables(dataframe, var)
        dataframe = pd.concat([dataframe, dummies], axis=1)
        dataframe = dataframe.drop(var, axis=1)
    output = "{output_dir}/{boros_joined}_{min_year}_{max_year}.csv".format(
    boros_joined = "_".join(boros), min_year = min(years),
    max_year = max(years), output_dir = output_dir)
    print(dataframe.shape)
    print("Writing output to file in {}".format(output))
    dataframe.to_csv(output, index = False, chunksize=1e4)
    return dataframe

def remove_columns(dataframe, columns):
    """Removes final unnecessary columns that were used previously for merges.
        Args:
            dataframe: Pandas dataframe.
            list (string) columns: list of columns to remove.
        Returns: Pandas dataframe
    """
    return dataframe.drop(columns, axis = 1)


                
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
    #pluto.to_csv("data/merged/pluto_manhattan_brooklyn_2014_2015.csv", index = False)
    print("Getting Finance data for: {} and {}".format(boros, years))
    finance = read_in_finance(boros, years)
    #finance.to_csv("data/merged/finance_manhattan_brooklyn_2014_2015.csv", index = False)
    print("Getting DTM Condo Unit data for: {}".format(boros))
    dtm = read_in_dtm(boros)
    buildings = merge_pluto_finance(pluto, finance, dtm)
    #buildings.to_csv("data/merged/manhattan_brooklyn_2014_2015.csv", index = False)
    print("Merging with subway data")
    buildings = bbl_dist_to_subway(buildings)
    final_cols_to_remove = ['borocode','unit_bbl','block','condono']
    buildings = remove_columns(buildings, final_cols_to_remove)
    cat_vars = ['borough','schooldist','council','bldgclass','landuse','ownertype','proxcode','lottype','tax_class_at_time_of_sale']
    buildings_with_cats = clean_categorical_vars(buildings, cat_vars, boros, years)

if __name__ == '__main__':
    main()


