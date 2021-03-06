# Digital Tax Map data for linking Condo lot numbers

* Download Digital Tax Map dataset and convert from DBF to CSV in order to properly merge DOF and Pluto data with Condo unit numbers
    * Download the Digital Tax Map shapefile data from:  https://data.cityofnewyork.us/Housing-Development/Department-of-Finance-Digital-Tax-Map/smk3-tmxj
    * Move to the `data` directory and unzip the Digital Tax Map directory there.
* Install `dbfpy` to convert data from .dbf to .csv format
    * Download `dbfpy` from: https://sourceforge.net/projects/dbfpy/files/dbfpy/2.3.1/dbfpy-2.3.1.tar.gz/download
    * Move to the `dbfpy-2.3.1` directory and run `python setup.py build` and `python setup.py install` to install the package.
    * Visit https://gist.github.com/bertspaan/8220892 or run `curl -O "https://gist.githubusercontent.com/bertspaan/8220892/raw/a173da9e4276537933b10641ac3f23dd39577225/dbf2csv.py"` to download a script for converting dbf to csv format.
* Convert the condo-related .dbf files to .csv format
    * Run `python dbf2csv.py ../data/Digital_Tax_Map_shapefile_03-16/DTM_0316_Condo_Units.dbf`
    * Move the newly created .csv files to the `data/dtm` directory.
* To compare PLUTO Borough-Block-Lot numbers to Department of Finance Borough-Block-Lot numbers, refer to the columns in `DTM_0316_Condo_Units.csv`:  
    * CONDO_BORO (corresponds to finance["Borough"] and pluto["BoroCode"])
    * CONDO_NUMB (corresponds to pluto["CondoNo"])
    * UNIT_BLOCK (corresponds to finance["Block"])
    * UNIT_LOT (corresponds to finance["Lot"])
    * UNIT_BBL (corresponds to finance["Borough" + "Block" + "Lot"])
    * UNIT_DESIG (corresponds to finance["Apartment Number"])
* Where pluto["CondoNo"] > 0, pluto["Lot"] and pluto["BBL"] will not match finance["Lot"] and finance["BBL"].
    * One PLUTO record (the entire building) may correspond to many Department of Finance sales records (each condominium unit in that building).
    * Pluto Condo Lot numbers nearly always start with 75xx.
