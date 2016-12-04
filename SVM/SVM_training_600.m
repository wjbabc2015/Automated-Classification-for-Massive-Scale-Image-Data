

accuracy_2 = ones(40,1);

speicfy_range = [3,13,22,24,27,29,32,34];%forward propagation approach

train_feature = train_feature_200(:,speicfy_range);
val_feature = val_feature_200(:,speicfy_range);
test_feature = test_feature_200(:,speicfy_range);

%model_post = svmtrain(train_label_200, train_feature, '-c 1 -g 0.07');
model_post = svmtrain([train_label_200; val_label_200], [train_feature; val_feature], '-c 1 -g 0.07');
[predict_label, accuracy_2_final, prob_est] = svmpredict(test_label_200, ...
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
