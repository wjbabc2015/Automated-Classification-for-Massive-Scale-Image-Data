CSV and .mat files:
data_index.csv
--- the index of 1000 experiment samples of each diffraction image type in Cell_old.csv, Debris_old.csv and Strip_old.csv


MATLAB/OCTAVE function:
ConfusionMatrix.m
--- build the numerical confusion matrix

read_csv_file.m
--- load CSV file 

label_com.m
--- remove wrongly classified samples from existing experiment samples

svm_func.m
--- simplify the SVM training and predicting procedure 


MATLAB/OCTAVE script:
data_preprocessing_3000_firsttime.m
--- construct 3000 experiment samples directly from Cell_old.csv, Debris_old.csv and Strip_old.csv and scaling them ranging 0 to 1.

data_preprocessing_for3000_firsttime.m
--- construct 3000 experiment samples directly from Cell_old.csv, Debris_old.csv and Strip_old.csv and scaling them ranging 0 to 1 for 10-folder CV using directly.

data_preprocessing_3000
--- load experiment samples directly from training_data.mat and testing_data.mat and scaling them ranging 0 to 1.

data_preprocessing_cs3.m
--- prepare experiment samples for case study 3 when mixing all diffraction images together

backward_propagation.m
--- obtain the feature combination by removing features

combined_feature_single.m
--- examine the accuracy level of the SVM classifier only with single feature

combined_feature_more.m
--- obtain the feature combination by adding features

combined_feature_s_polar.m
--- obtain the feature combination by adding features but only considering s-polarized features

combined_feature_p_polar.m
--- obtain the feature combination by adding features but only considering p-polarized features

confusion_matrix_plot.m
--- convert numerical confusion matrix into graphic

cross_validation_10_folder.m
--- 10-folder cross_validation including dividing training data set

cross_validation_10_folder_cs3.m
--- 10-folder cross_validation including dividing training data set for case study 3 when mixing diffraction image together

cross_validation_10_folder_for3000.m
--- 10 FCV applied for 3000 data matrix. (feature combination settting inside)

forward_propagation_scenario_plot.m
--- plot the scenario of the accuracy trend

parameter_estimate.m
--- examine the parameter pair of the RBF kernel

refine_classifier.m
--- Redo SVM training and testing procedure after removing wrongly classified samples

SVM_training_3000.m
--- simple procedure for training and testing SVM classifier




