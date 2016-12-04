accuracy_2 = ones(40,1);

train_feature = experiment_feature(1:2000,:);
train_label = experiment_label(1:2000,:);
test_feature = experiment_feature(2401:3000,:);
test_label = experiment_label(2401:3000,:);

for i = 1:17
    model_pre = svmtrain(train_label, train_feature(:,i), '-c 1 -g 0.07');
    [predict_label, accuracy_temp, prob_est] = svmpredict(test_label, ...
        test_feature(:,i), model_pre);
    accuracy_2(i) = accuracy_temp(1);
    %cMatrix = confusionmat(test_label, predict_label);
    %confusion_matrix_plot();
end

for i = 21:37
    model_pre = svmtrain(train_label, train_feature(:,i), '-c 1 -g 0.07');
    [predict_label, accuracy_temp, prob_est] = svmpredict(test_label, ...
        test_feature(:,i), model_pre);
    accuracy_2(i) = accuracy_temp(1);
    %cMatrix = confusionmat(test_label, predict_label);
    %confusion_matrix_plot();
end