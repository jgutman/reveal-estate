import pandas as pd
import os
import re


def main():
    years = list(range(2003, 2017))
    boros = ["manhattan", "brooklyn", "queens", "bronx", "statenisland"]
    files = [f for f in os.listdir('data/merged') if f.endswith('.csv')]
    ind_files = [f for f in os.listdir('data/merged/individual')
        if f.endswith('.csv')]

    boros_dsize = {}
    for f in files:
        data = pd.read_csv(os.path.join('data/merged', f), low_memory = False)
        nrows = data.shape[0]
        boro_name = re.split('_\d', f, maxsplit=1)[0]
        boros_dsize[boro_name] = nrows
        print('{} contains {} rows'.format(f, nrows))

    boros_dsize['total'] = sum([boros_dsize[b] for b in boros])

    boro_years_dsize = {b: {} for b in boros}
    for f in ind_files:
        data = pd.read_csv(os.path.join('data/merged/individual', f),
            low_memory = False)
        nrows = data.shape[0]
        boro_name, year = re.split('_', f, maxsplit=2)[0:2]
        boro_years_dsize[boro_name][year] = nrows
        print('{} contains {} rows'.format(f, nrows))

    boro_years_dsize['total'] = {b: sum(boro_years_dsize[b].values())
        for b in boros}
    boro_years_dsize['total']['total'] = sum([boro_years_dsize['total'][b]
        for b in boros])

    print('All years: ', boros_dsize)
    print('Year by year: ', boro_years_dsize['total'])


if __name__ == '__main__':
    main()
