# reveal-estate
Fall 2016 Capstone Project for Nora Barry, Laura Buchanan, Jacqueline Gutman

Final paper in [RevealEstate.pdf](./RevealEstate.pdf)

Python version 3.5.2

* Set up a Python3 virtual enviroment with `conda` or `virtualenv`. Install necessary packages for analysis by running `pip3 install -r requirements.txt -U` or `conda install --yes --file hpc_requirements.txt`.

* Download and organize NYC Department of Finance data into data directory by running `source download_finance_data.sh`.

* Download PLUTO data and data dictionary by running `source download_pluto_data.sh`.

* Download PyProj github repository to perform transformations between X/Y coordinates and latitude/longitude coordinates by running `source get_pyproj.sh`.

* Review and follow instructions for downloading Digital Tax Map data in the [DTM README](./download_dtm.md)

* Extract distance to subway data and other open NYC data (add details here).

* Merge PLUTO and Department of Finance data for specified years and boroughs by running `python3 merge_pluto_finance_new.py --borough all --year all` to obtain and merge data for all boroughs and all years (5 boroughs, 2003-2016), or by running `python3 merge_pluto_finance_new.py --borough {BOROUGH} {BOROUGH} --year {YEAR} {YEAR}` for whatever boroughs and years desired. See `python3 merge_pluto_finance_new.py --help` for details.

* Fit Linear Regression or Random Forest Model to merged data set by running `python3 regression_loop.py --data {path_to_merged_csv} --model {lr, rf, ada, bag, et, gb, en, hr, br, ll, lasso, ridge, sgd, svr, linsvr} --iters {50}`. See `python3 regression_loop.py --help` for details.

* If using the Mercer HPC environment, check the configuration with `source setup_conda.sh` and then submit a job by running `git pull; qsub jobs/{job}.pbs`
