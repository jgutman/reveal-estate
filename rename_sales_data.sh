# bash script to use the rename utility to rename all finance sales data
# to a standard format: 20YY_boro.xls

brew install rename
cd data/finance_sales

# rename [options] [regex search/replace] [in these files]
# -v = verbose, display what you are doing on screen
# -n = "do not, just show what might happen",
# in other words, use '-n' if you want to test the end result

rename -v 's/sales_//' ./*
rename -v 's/si/statenisland/' ./*
rename -v 's/rolling/2016_/' ./*
rename -v 's/(\w+)_(\d+)/20$2_$1/g' ./*

cd ../..
