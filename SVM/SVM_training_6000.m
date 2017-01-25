
speicfy_range = [16,2,3,6,4,13];
%speicfy_range = [1,5,8,16,6,7,12];



train_feature = experiment_feature(1:4000,speicfy_range);
train_label = experiment_label(1:4000,:);
val_feature = experiment_feature(4001:4800,speicfy_range);
val_label = experiment_label(4001:4800,:);
test_feature = experiment_feature(4801:6000,speicfy_range);
test_label = experiment_label(4801:6000,:);

%model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
model_post = svmtrain(train_label, train_feature, '-c 256 -g 4');
%model_post = svmtrain(train_label, train_feature, '-c 4096 -g 0.125');
%model_post = svmtrain([train_label; val_label], [train_feature; val_feature], '-c 1 -g 0.07');
[predict_label, accuracy_2_final, prob_est] = svmpredict(test_label, test_feature, model_post);
%[predict_label, accuracy_2_final, prob_est] = svmpredict(val_label, val_feature, model_post);
    
cMatrix = confusionmat(test_label, predict_label);
confusion_matrix_plot();

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
