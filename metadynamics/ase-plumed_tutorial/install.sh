# ASE and PLUMED have specific installation instructions that you can
# find in:
#
# PLUMED: https://www.plumed.org/doc-v2.8/user-doc/html/_installation.html#installingpython
# ASE: https://wiki.fysik.dtu.dk/ase/install.html#installation-from-source
#
# However, this file contains the necessary ubuntu commands to install both
# codes USING ANACONDA. If you have conda already installed, ignore
# the first part. 
#
# You can also execute completely this file in your terminal with the lines:
#
# chmod +x install.sh
# ./install.sh

echo -e "\n\n----------------- INSTALL ANACONDA -------------------\n\n"
cd ~/
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sha256sum Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
rm Anaconda3-2021.05-Linux-x86_64.sh
~/anaconda3/bin/conda init &&
eval "$(/home/$USER/anaconda3/bin/conda shell.bash hook)" &&

echo -e "\n\n------------------- INSTALL PLUMED -------------------\n\n"
~/anaconda3/bin/conda install -c conda-forge py-plumed -y &&

echo -e "\n\n-------------------- INSTALL ASE ---------------------\n\n"
pip install ase
