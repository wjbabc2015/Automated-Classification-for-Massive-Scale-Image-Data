training_size_refined = size(train_feature, 1);

refined_data_index = zeros(1,1);

pos = 1;

for i = 1:training_size_refined
	if ~any(wrong_label_index == i)
		refined_data_index(pos,1) = i;
		pos ++;
	end
end

refined_train_label = train_label(refined_data_index, :);
refined_train_feature = train_feature(refined_data_index, :);

[predict_label, accuracy] = svm_func(refined_train_label, refined_train_feature, test_label, test_feature, str, true);

