mkdir -p data/merged/individual

for boro in manhattan bronx brooklyn queens statenisland;
do
  for year in {2003..2016};
  do
    # run python merge script
    python3 merge_pluto_finance_new.py --borough $boro --year $year
    mv "data/merged/${boro}_${year}_${year}.csv" data/merged/individual
  done
done
