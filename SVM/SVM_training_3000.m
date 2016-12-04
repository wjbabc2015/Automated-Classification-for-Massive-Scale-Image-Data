

accuracy_2 = ones(40,1);

%speicfy_range = [1:14,16:17,21:34,36:37];%total 32 feature parameters
%speicfy_range = [1,3,5,6,11,16,17,29,32,33,34];%forward propagation approach 3000
speicfy_range = [1,3:14,16:17,21:23,25:32,34,37]; %Classifier 1
%speicfy_range = [1,3:14,16:17,21:27,29:32,34,37]; %Classifier 2
%speicfy_range = [1,3:13,16:17,21:27,29:32,34,37]; %Classifier 3
%speicfy_range = [1,4:13,16:17,21:27,29:32,37]; %Classifier 4

train_feature = experiment_feature(1:2000,speicfy_range);
train_label = experiment_label(1:2000,:);
val_feature = experiment_feature(2001:2400,speicfy_range);
val_label = experiment_label(2001:2400,:);
test_feature = experiment_feature(2401:3000,speicfy_range);
test_label = experiment_label(2401:3000,:);

%model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
%model_post = svmtrain(train_label, train_feature, '-c 512 -g 0.5');
%model_post = svmtrain(train_label, train_feature, '-c 4096 -g 0.125');
model_post = svmtrain([train_label; val_label], [train_feature; val_feature], '-c 1 -g 0.07');
[predict_label, accuracy_2_final, prob_est] = svmpredict(test_label, ...
        test_feature, model_post);
    
%cMatrix = confusionmat(test_label, predict_label);
%confusion_matrix_plot();

%{
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
