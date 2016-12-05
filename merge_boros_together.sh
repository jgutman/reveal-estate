for boro in manhattan bronx brooklyn queens statenisland all;
do
  # run python merge script
  python3 merge_pluto_finance_new.py --borough $boro --year all
done
