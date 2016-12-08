curl -O "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash Miniconda3-latest-Linux-x86_64.sh
conda update conda
cd $SCRATCH
git clone https://github.com/NYU-CDS-Capstone-Project/reveal-estate.git
cd reveal-estate
conda create --name capstone python=3.5
source activate capstone
pip install -r requirements.txt
