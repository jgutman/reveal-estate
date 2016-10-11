brew install rename
cd data/finance_sales

// rename [options] [regex search/replace] [in these files]
// -v = verbose, display what you are doing on screen
// -n = "do not, just show what might happen",
// in other words, use '-n' if you want to test the end result

rename 's/sales_//' ./*
rename 's/si/statenisland/' ./*
rename 's/rolling/2016_/' ./*
rename 's/(\w+)_(\d+)/20\2_\1/g' ./*
