accuracy_2 = ones(40,1);

for i = 1:17
    model_pre = svmtrain(train_label_200, train_feature_200(:,i), '-c 1 -g 0.07');
    [predict_label, accuracy_temp, prob_est] = svmpredict(test_label_200, ...
        test_feature_200(:,i), model_pre);
    accuracy_2(i) = accuracy_temp(1);
    %cMatrix = confusionmat(test_label, predict_label);
    %confusion_matrix_plot();
end

for i = 21:37
    model_pre = svmtrain(train_label_200, train_feature_200(:,i), '-c 1 -g 0.07');
    [predict_label, accuracy_temp, prob_est] = svmpredict(test_label_200, ...
        test_feature_200(:,i), model_pre);
    accuracy_2(i) = accuracy_temp(1);
    %cMatrix = confusionmat(test_label, predict_label);
    %confusion_matrix_plot();
end