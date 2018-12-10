conda remove -y --name gluoncv --all
conda create -y --name gluoncv python=3.6
source activate gluoncv
echo "Installing things in " $CONDA_DEFAULT_ENV

pip install mxnet-cu90
pip install gluoncv
