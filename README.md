# RPCT
A protein classification prediction toolkit based on RAAC-PSSM
## Introduction
The RPCT toolkit is a dedicated toolkit based on the RAAC-PSSM prot-ein classification prediction method. It uses 7 feature extraction methods and SVM for protein classification prediction.
### Quick Start
1. The RPCT package is written in Python. It is recommended to use [conda](https://www.anaconda.com/download/) to manage python packages. Or please make sure all the following packages are installed in their Python environment: ray, sklearn, blast.
2. Please convert the data to FASTA format.
3. The RPCT toolkit supports both windows and linux platforms. Before you run RPCT, you should check your commands.
#### Run RPCT by windows
```
python RPCT_windows.py
```
#### Run RPCT by linux
```
python RPCT_linux.py <Fuctions> <parameters>
```
### Usage For Linux
#### 1.  read.  Load your Fasta datasets and split them into separate fasta files.
Command line
```
python RPCT_linux.py read file_name -o out_folder
```
Example
```
python RPCT_linux.py read test_positive.fasta -o test_p
```
