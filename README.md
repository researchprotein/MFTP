# MFTP

Download the compressed MFTP.zip file from Releases and execute the following instructions to running the MFTP prediction on the independent testing dataset:

unzip MFTP.zip

conda create -n your_env_name python=3.8

conda install --yes --file requirements.txt

python TestGAT.py
