function [predict_label, accuracy] = svm_func...
    (train_label, train_feature, test_label, test_feature, condition_str, cm_condition)

model = svmtrain(train_label, train_feature, condition_str);
[predict_label, accuracy_detail, prob_est] = svmpredict(test_label, test_feature, model);

accuracy = accuracy_detail(1);

	if cm_condition
		cMatrix = ConfusionMatrix(test_label, predict_label);
		confusion_matrix_plot;
	end
end