This is the user guide of running the MFTP prediction model.

First of all, download the Python source code from https://github.com/researchprotein/MFTP

Next, download the datasets, feature files and pretrained model from https://pan.baidu.com/s/1XagNIUpc8cPg2IkvNdRgRg?pwd=agtx with password agtx

Subsequently, execute following commands:

unzip MFTP.zip

mv sequence.zip MFTP/

mv gat.pkl MFTP/

cd MFTP

unzip sequence.zip

conda create --name MFTP python=3.8

pip install -r requirement.txt

python TestGAT.py

And now the program should be running and will output the prediction result for the independent test dataset.
