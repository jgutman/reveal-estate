import pandas as pd

def write_bbl_to_file(path, output_path):
    data_limited = pd.read_csv(path,
        usecols = ['bbl', 'latitude', 'longitude', 'zipcode'])
    data_limited = data_limited.drop_duplicates()
    data_limited.to_csv(output_path,
        index = False, chunksize=1e4)
    print("Data of size {} written to {}".format(data_limited.shape,
        output_path))

def check_nrows(path):
    data_limited = pd.read_csv(path, usecols = ['bbl'])
    total_bbls = data_limited.shape[0]
    unique_bbls = len(data_limited.drop_duplicates())
    print("total bbls in data: {}\nunique bbls in data: {}".format(
        total_bbls, unique_bbls))

def main():
    boros = 'bronx_brooklyn_manhattan_queens_statenisland'
    years = '2003_2016'
    path = 'data/merged/{}_{}.csv'.format(boros, years)
    output_path = 'data/{}_{}_bbls.csv'.format(boros, years)
    write_bbl_to_file(path, output_path)
    check_nrows(path)

if __name__ == '__main__':
    main()
