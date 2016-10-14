# creates a folder for pluto data, downloads and unzips files from NYC website

mkdir -p data/nyc_pluto_16v1
cd data
# download and unzip data
curl -O "http://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyc_pluto_16v1.zip"
# unzip directory and remove zip file
unzip nyc_pluto_16v1.zip -d nyc_pluto_16v1
rm *.zip
cd ..
