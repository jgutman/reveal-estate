# creates a folder for finance sales data, downloads all files from nyc dept of
# finance website (all boroughs, all years, 2003-rolling), and calls bash script
# to rename files appropriately

mkdir -p data/finance_sales
cd data/finance_sales

# for all five boroughs:
for boro in manhattan bronx brooklyn queens statenisland;
do
  # download rolling sales data
  curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_"$boro".xls"

  # download 2010-2015 data
  for year in {2010..2015};
  do
    curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/annualized-sales/"$year"/"$year"_"$boro".xls"
  done
  # download 2007-2009 data
  curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/annualized-sales/2009_"$boro".xls"
  curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/09pdf/rolling_sales/sales_2008_"$boro".xls"
  curl -O "http://www1.nyc.gov/assets/finance/downloads/excel/rolling_sales/sales_2007_"$boro".xls"
done

for boro in manhattan bronx brooklyn queens si;
do # download 2003-2006 data
  for year in {3..6};
  do
    curl -O "http://www1.nyc.gov/assets/finance/downloads/sales_"$boro"_0"$year".xls"
  done
done

curl -o ./data_dictionary.pdf "https://www1.nyc.gov/assets/finance/downloads/pdf/07pdf/glossary_rsf071607.pdf"

cd ../..
# run script to rename files in a uniform standard
source rename_sales_data.sh
mkdir -p data/merged
