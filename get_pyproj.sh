# Get PyProj repository to perform cartographic transformations from
# geographic (longitude,latitude) to native map projection (x,y) coordinates
# and vice versa using the Proj class

git clone https://github.com/jswhit/pyproj
cd pyproj
python3 setup.py build
python3 setup.py install
# runs unit tests to check install
# if this doesn't work, run `pip3 install nose2` and retry
nose2

# In a conda environment try the following:
# source activate capstone
# conda install -c jjhelmus pyproj=1.9.3 
