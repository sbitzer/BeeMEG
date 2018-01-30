#!/usr/bin/env bash

conda create -n sufer python=2

source activate surfer

# need to specify wxpython=3, because wxpython 4 doesn't work, numpy needs to be at least 1.11
conda install -c conda-forge pandas seaborn numpy matplotlib mayavi vtk qt ipython wxpython=3 scipy pytables mkl imageio nibabel

# need to install pil separately, because otherwise it somehow didn't find image writers
conda install pil

cd ~/PySurfer
python setup.py develop

cd ~/mne-python
python setup.py develop

