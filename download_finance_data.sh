# creates a folder for finance sales data, downloads all files from nyc dept of
# finance website (all boroughs, all years, 2003-rolling), and calls bash script
# to rename files appropriately

mkdir -p data/finance_sales
cd data/finance_sales

for boro in manhattan bronx brooklyn queens statenisland;
do
  curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/rollingsales_"$boro".xls"

  for year in {2010..2015};
  do
    curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/annualized-sales/"$year"/"$year"_"$boro".xls"
  done

  curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/annualized-sales/2009_"$boro".xls"
  curl -O "http://www1.nyc.gov/assets/finance/downloads/pdf/09pdf/rolling_sales/sales_2008_"$boro".xls"
  curl -O "http://www1.nyc.gov/assets/finance/downloads/excel/rolling_sales/sales_2007_"$boro".xls"
done

for boro in manhattan bronx brooklyn queens si;
do
  for year in {3..6};
  do
    curl -O "http://www1.nyc.gov/assets/finance/downloads/sales_"$boro"_0"$year".xls"
  done
done

cd ../..
source rename_sales_data.sh
