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
python RPCT_linux.py search -d input_document_name -f folder_name

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
Filter the features of the target feature file through the IFS-RF method (Incremental Feature Selection based on the Relief-Fscore method). And output an ACC_Chart and a Feature_sort_file for the target feature file
#### Command line
```
python RPCT_linux.py filter input_document_name -c c_number -g gamma -cv cross_validation_fold -n total_feature_number -o out_file_name -r random_number

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
python RPCT_linux.py fffs input_document_name -f feature_sort_file -n total_feature_number -l stop_feature_number -o out_file_name

# optional arguments:
#   input_document_name    the target feature file which has been chosen in Filter function.
#   -f                     the Feature_Sort_File which has been created in Filter function.
#   -n                     the total feature number of target feature file.
#   -l                     the stop feature number of target feature file, and you can find it in the ACC_Chart which has been created in Filter function.
```
#### Example
```
python RPCT_linux.py fffs .\Train_fs\t1s2_rpct.fs -f t1s2-ifs.txt -n 190 -l 72 -o t1s2-72
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
python RPCT_linux.py train -f Train_fs -o .\Train_fs\t1s2_rpct.fs -cg Hyperparameters.txt
```
#### 8. Eval
Evaluate feature files through the Cross-validation function provided by LIBSVM.
#### Command line
```
python RPCT_linux.py eval -d input_document_name -f input_folder_name -c c_number -g gamma -o out_folder -cg Hyperparameters_file_name -cv cross_validation_fold

# optional arguments:
#   -d    input the target feature file, and you can not use this parameter with -f and -cg together.
#   -f    input the feature folder, and you can not use this parameter with -d, -c and -g together.
#   -o    if you choose the parameter -f, you should input a folder name, and if you choose the parameter -d, you should input a file name.
#   -cg   the Hyperparameters file which has been created in Search function, and you can not use this parameter with -d, -c and -g together.
#   -cv   the cross validation fold of SVM, you can choose 5, 10 or -1 or define it by your experience.
```
#### Example
```
python RPCT_linux.py train -d .\Train_fs\t1s2_rpct.fs -c 8 -g 0.125 -o t1s2
python RPCT_linux.py train -f Train_fs -o Model_fs -cg Hyperparameters.txt
```
#### 9. ROC
Draw the ROC-Cruve by sklearn.
#### Command line
```
python RPCT_linux.py roc input_document_name -c c_number -g gamma -o out_file_name -n total_feature_number
```
#### Example
```
python RPCT_linux.py roc .\Train_fs\t1s2_rpct.fs -c 8 -g 0.125 -o t1s2 -n 190
```
#### 10. Predict
Evaluate the target model with a feature files which from an independent datasets. And output a Evaluation_file for the target model.
#### Command line
```
python RPCT_linux.py predict input_document_name -m target_model_name -o out_file_name

# optional arguments:
#   input_document_name    input the target feature file, and make sure it has the same reduce type with the target model.
#   -m                     input the target model file, and make sure it has the same reduce type with the target feature file.
```
#### Example
```
python RPCT_linux.py predict .\Predict_fs\t1s2_rpct.fs -m t1s2.model -o t1s2
```
#### 11. Res
Reduce amino acids by personal rules. And output a personal RAAC list from size_2 to size_19.
#### Command line
```
python RPCT_linux.py res aaindex_id

# optional arguments:
#   aaindex_id    the ID of physical and chemical characteristics in AAindex Database, and you can check it in aaindexDB folder in rpct folder or view it online.
```
#### Example
```
python RPCT_linux.py res CHAM830102
```
#### 12. IntLen
Choose the top-n classify model to participate in the Integrated-Learning which predict through majority vote mothod.
#### Command line
```
python RPCT_linux.py intlen -tf train_feature_folder -pf predict_feature_folder -ef eval_result_file -cg Hyperparameters_file_name -m member

# optional arguments:
#   -tf    input the train feature folder name.
#   -pf    input the predict feature folder name which has been created by an independent datasets.
#   -ef    input the eval result file name which has been created by Eval funtion and saved in eval_result folder.
#   -m     the number of integrated-learning members.
```
#### Example
```
python RPCT_linux.py intlen -tf Train_fs -pf Predict_fs -ef .\Eval_fs\Features_eval.csv -cg Hyperparameters.txt -m 10
```
