accuracy_2 = ones(40,1);

train_feature = experiment_feature(1:4000,:);
train_label = experiment_label(1:4000,:);
test_feature = experiment_feature(4801:6000,:);
test_label = experiment_label(4801:6000,:);

for i = 1:17
    model_pre = svmtrain(train_label, train_feature(:,i), '-c 1 -g 0.07');
    [predict_label, accuracy_temp, prob_est] = svmpredict(test_label, ...
        test_feature(:,i), model_pre);
    accuracy_2(i) = accuracy_temp(1);
    %cMatrix = confusionmat(test_label, predict_label);
    %confusion_matrix_plot();
end