# reveal-estate
Fall 2016 Capstone Project for Nora Barry, Laura Buchanan, Jacqueline Gutman

Python version 3.5.2

* Install necessary packages for analysis by running `pip3 install -r requirements.txt`.

* Download and organize NYC Department of Finance data into data directory by running `source download_finance_data.sh`.

* Download PLUTO data and data dictionary by running `source download_pluto_data.sh`.

* Download PyProj github repository to perform transformations between X/Y coordinates and latitude/longitude coordinates by running `source get_pyproj.sh`.

* Merge PLUTO and Department of Finance data for specified years and boroughs by running `python3 merge_pluto_finance.py` to obtain and merge data for default boroughs and years (Manhattan and Brooklyn, 2014-2015), or by running `python3 merge_pluto_finance.py --borough {BOROUGH} {BOROUGH} --year {YEAR} {YEAR}` for whatever boroughs and years desired. See `python3 merge_pluto_finance.py --help` for details.

* Fit Linear Regression or Random Forest Model to merged data set by running `python3 final_modeling.py --data {path_to_merged_csv} --model {LR or RF}`. See `python3 final_modeling.py --help` for details.
