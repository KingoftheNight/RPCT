# RPCT
A protein classification prediction toolkit based on RAAC-PSSM
## Introduction
The RPCT toolkit is a dedicated toolkit based on the RAAC-PSSM prot-ein classification prediction method. It uses 7 feature extraction methods and SVM for protein classification prediction.
### Quick Start
1. The RPCT package is written in Python. It is recommended to use [conda](https://www.anaconda.com/download/) to manage python packages. Or please make sure all the following packages are installed in their Python environment: ray, sklearn, blast.
```
# install packages by conda
conda install package_name
# install blast+ by conda
conda install -c bioconda blast
```
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
#### 1. Read
Load your Fasta datasets and split them into separate fasta files.
#### Command line
```
python RPCT_linux.py read file_name -o out_folder
```
#### Example
```
python RPCT_linux.py read test_positive.fasta -o test_p
```
#### 2. Blast
Get PSSM profiles through _psiblast_ function provided by _BLAST+_ (https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/).
#### Command line
```
python RPCT_linux.py blast input_folder_name -db blast_database_name -n num_iterations -ev expected_value -o out_folder
```
#### Example
```
python RPCT_linux.py blast test_p -db pdbaa -n 3 -ev 0.001 -o pssm-tp
```
#### 3. Extract
Extract feature files through RAAC-PSSM extract method.
#### Command line
```
python RPCT_linux.py extract input_folder_name -raa raac_book_name -o out_folder -l windows_size -r self_raa_code

# optional arguments:
#   -raa  raac book saved in raacDB folder in rpct, you can not use this parameter and -r together.
#   -o    if you choose the parameter -raa, you should input a folder name, and if you choose the parameter -r, you should input a file name
#   -r    self_raa_code format should contain all amino acid types, and be separated by '-', for example: LVIMC-AGST-PHC-FYW-EDNQ-KR
```
#### Example
```
python RPCT_linux.py extract pssm-tp -raa raaCODE -o Train_fs -l 5
python RPCT_linux.py extract pssm-tp -o My_train -l 5 -r LVIMC-AGST-PHC-FYW-EDNQ-KR
```
