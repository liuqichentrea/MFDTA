# MGDTA

The code is the implementation for the paper 'MFDTA: A Heterogeneous Drug-Target Affinity Prediction Network Based on Multivariate Features'.

# Requirements

python 3.6.2

pytorch 1.10.1+cu111

scipy 1.3.1

numpy 1.19.5

pandas 0.25.1

rdkit-pypi 2021.9.4

transformers 4.18.0

# Dataset

Datasets in datasets file

Because the Kiba file is too large, we uploaded it in groups

# Run code

run MFDTA.py 

# Notes

The above environment configuration is built based on our server. It is worth noting that you must download the corresponding dependent package version according to your server's configuration. In addition, the path to the dataset must also be changed to your local file path.

Our server configuration：

![4a10a2750eb88a56ad7f4e6dec118c6](https://github.com/liuqichentrea/MFDTA/assets/87010868/1ae845e3-bce6-464c-9298-f05760480f1a)

Next, take our machine configuration as an example to illustrate each step of our code running：

1.Creating a virtual environment：conda create -n <env_name> python=<python_version>

![image](https://github.com/liuqichentrea/MFDTA/assets/87010868/9ed3da0d-22af-4d6c-bb62-204954c21864)

2.Activate the created virtual environment：conda activate MFDTA

![image](https://github.com/liuqichentrea/MFDTA/assets/87010868/81d6d8d4-0e41-4452-9512-7872c4669ca5)

3.Download dependent packages: the required dependent packages are listed above. Installation command:pip install <Name of dependent package> or conda install <Name of dependent package>
If the installation fails, you can manually download and install it on the corresponding official website

4.Download the files and codes on GitHub to the local file folder: cd <local folder path>

![image](https://github.com/liuqichentrea/MFDTA/assets/87010868/a465be03-4bef-44c6-a6da-861e46e63ec3)

5.Run the script MFDTA.py:python MFDTA.py

![image](https://github.com/liuqichentrea/MFDTA/assets/87010868/3e67c30c-3210-452e-85df-9507b5d82e3f)


Screenshot of code running process：

![image](https://github.com/liuqichentrea/MFDTA/assets/87010868/f057951a-f7ac-4ca1-9756-2eeb9b40f9de)


