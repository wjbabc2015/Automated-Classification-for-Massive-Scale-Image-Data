%{
	The different feature combination based on 2100 training data set 
	and 600 testing data set
%}

accuracy_2 = ones(40,1);

%speicfy_range = [1:14,16:17,21:34,36:37];%total 32 feature parameters Table 3
%speicfy_range = [2,3,4,5,6,10,12,16,21,23,25,27,28,29,30,34,36];% Table 7
%speicfy_range = [36,12,23,6,28,4,25,27,2,29,16,30,34,10,5,3,21,1,31,33];%20 feature scenario
%speicfy_range = [1,2,5,6,10,11,13,16,17,21,23,25,26,27,28,29,33,34,36,37];
%speicfy_range = [36,11,6,23,26,29,17,1,25,16,13,10,27,24,28,21,34,2,33,37,5,32,22,12];
%speicfy_range = [27,32,12,25,4,29,16,5,23,10,24,34,14,28,2,36,26,1,13,37];%Table 20
speicfy_range = [27,5,25,16,23,4,2,6,10,3,30,36,29,34,1]; % Table 16

%speicfy_range = [1:6,9:10,12,14,16:17,21:23,25:28,30,33,34,37];%Table 11
%speicfy_range = [2,4:6,16,23,25,27];
%speicfy_range = [1:6,10,16,22:23,25:27,30,33,37];%table 10
%speicfy_range = [1:4,6:7,9:12,14,16:17,21:29,31,33:34,36:37];

%s-polarized features
%speicfy_range = [1:14,16:17];% accuracy is 53%
%speicfy_range = [1,2,4,6,10,13,14,16];
%speicfy_range = [1,2,4,6,10,12,13,14,16,17,21,22,24,26,30,32,33,34,36,37];% accuracy is 68.83
%speicfy_range = [1,2,3,4,5,6,8,10,12,14,17];
%speicfy_range = [2,4,5, 1,7,10,13,16, 17];
%speicfy_range = [2,4,5, 1,7,10,13,16, 17, 22,24,25, 21,27,30,33,36, 37];

%p-polarized features
%speicfy_range = [21:34,36:37];% accuracy is 61.5
%speicfy_range = [21,23,25,26,27,28,29,30,31,34,36,37];
%speicfy_range = [25,28,31, 36,34,27,29,30, 23,37];
%speicfy_range = [5,8,11, 7,9,16,14,10, 3,17, 25,28,31, 27,29,36,34,30, 23,37];

%mixed diffraction image classification
%speicfy_range = [1:14,16:17];
%speicfy_range = [2,13,16,4,1,6,14];
%speicfy_range = [5,8,11, 7,9,16,14,10, 3,17];
%speicfy_range = [2,4,5, 1,7,10,13,16, 17];

%speicfy_range = [23,16,4,25,27,2,6,30,10,5,3,26,33,1,22,37,14,36,12,28,29,34,4];


train_feature = training_feature(:,speicfy_range);
test_feature = testing_feature(:,speicfy_range);

str = '-c 2 -g 4';

[predict_label, accuracy] = svm_func(train_label, train_feature, test_label, test_feature, str, true);

%{
%model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
%model_post = svmtrain(train_label, train_feature, '-c 1 -g 8');%32 feature paramters 2^15 2^(-5)
%model_post = svmtrain(train_label, train_feature, '-c 32 -g 1');%32 feature paramters 2^5 2^0
model_post = svmtrain(train_label, train_feature, '-c 2 -g 4');%32 feature paramters 2^1 2^2
%model_post = svmtrain(train_label, train_feature, '-c 4096 -g 0.125');
%model_post = svmtrain(train_label, train_feature, '-c 128 -g 2');
%model_post = svmtrain([train_label; val_label], [train_feature; val_feature], '-c 1 -g 0.07');
[predict_label, accuracy_2_final, prob_est] = svmpredict(test_label, test_feature, model_post);
%[predict_label, accuracy_2_final, prob_est] = svmpredict(val_label, val_feature, model_post);
    
cMatrix = ConfusionMatrix(test_label, predict_label);
confusion_matrix_plot;

s-:
CON or DIS 2 or 12
IDM or DVA 5 or 11
VAR or SVA 4 or 8

ASM or MAP 1 or 16
CLS or CLP 13 or 14
SEN or ENT 7 or 9
DEN 10

COR 3
SAV or MEA 6 or 17
%}
