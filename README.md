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
#   -raa  raac book saved in raacDB folder in rpct, and you can not use this parameter and -r together.
#   -o    if you choose the parameter -raa, you should input a folder name, and if you choose the parameter -r, you should input a file name.
#   -r    self_raa_code format should contain all amino acid types, and be separated by '-', for example: LVIMC-AGST-PHC-FYW-EDNQ-KR .
```
#### Example
```
python RPCT_linux.py extract pssm-tp -raa raaCODE -o Train_fs -l 5
python RPCT_linux.py extract pssm-tp -o My_train -l 5 -r LVIMC-AGST-PHC-FYW-EDNQ-KR
```
#### 4. Search
Search Hyperparameters of the target feature file through the grid function provided by LIBSVM (https://www.csie.ntu.edu.tw/~cjlin/libsvm/oldfiles/index-1.0.html).
#### Command line
```
python RPCT_linux.py search -d document_name -f folder_name

# optional arguments:
#   -d    input the target feature file and output a single result of it, and you can not use this parameter and -f together.
#   -f    input the target feature folder and output the Hyperparameters file which contains all results of the feature folder, and you can not use this parameter and -d together.
```
#### Example
```
python RPCT_linux.py search -d .\Train_fs\t1s2_rpct.fs
python RPCT_linux.py search -f Train_fs
```
#### 5. Filter
Filter the features of the target feature file through the IFS-RF method (Incremental Feature Selection based on the Relief-Fscore method).
#### Command line
```
python RPCT_linux.py filter document_name -c c_number -g gamma -cv cross_validation_fold -n feature_number -o out_file_name -r random_number

# optional arguments:
#   -c    the penalty coefficient of SVM, you can get it through Search function or define it by your experience.
#   -g    the gamma of RBF-SVM, you can get it through Search function or define it by your experience.
#   -cv   the cross validation fold of SVM, you can choose 5, 10 or -1 or define it by your experience.
#   -r    the random sampling number of Relief method.
```
#### Example
```
python RPCT_linux.py filter .\Train_fs\t1s2_rpct.fs -c 8 -g 0.125 -cv 5 -n 190 -o t1s2 -r 30
```
#### 6. Filter Features File Setting
Create a filtered feature file of the target feature file through the Feature_Sort_File which has been output in Filter function.
#### Command line
```
python RPCT_linux.py fffs document_name -f feature_sort_file -n hole_feature_number -l last_feature_number

# optional arguments:
#   -f    the Feature_Sort_File.
#   -n    the gamma of RBF-SVM, you can get it through Search function or define it by your experience.
#   -l    the cross validation fold of SVM, you can choose 5, 10 or -1 or define it by your experience.
```
#### Example
```
python RPCT_linux.py fffs .\Train_fs\t1s2_rpct.fs -f t1s2-ifs.txt -n 190 -l 72
```
#### 7. Train
Train feature files through the LIBSVM.
#### Command line
```
python RPCT_linux.py train -d input_document_name -f input_folder_name -c c_number -g gamma -o out_folder -cg Hyperparameters_file_name

# optional arguments:
#   -d    input the target feature file, and you can not use this parameter with -f and -cg together.
#   -f    input the feature folder, and you can not use this parameter with -d, -c and -g together.
#   -o    if you choose the parameter -f, you should input a folder name, and if you choose the parameter -d, you should input a file name.
#   -cg   the Hyperparameters file which has been created in Search function, and you can not use this parameter with -d, -c and -g together.
```
#### Example
```
python RPCT_linux.py train -d .\Train_fs\t1s2_rpct.fs -c 8 -g 0.125 -o t1s2
python RPCT_linux.py train -f Train_fs -o Model_fs -cg Hyperparameters.txt
```
