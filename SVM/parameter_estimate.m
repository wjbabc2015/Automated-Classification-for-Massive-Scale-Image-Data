%speicfy_range = [1,3:14,16:17,21:23,25:32,34,37];
speicfy_range = [36,22,23,26,24,33];
%speicfy_range = [1,5,8,16,6,7,12];

train_feature = experiment_feature(1:2000,speicfy_range);
train_label = experiment_label(1:2000,:);
val_feature = experiment_feature(2001:2400,speicfy_range);
val_label = experiment_label(2001:2400,:);
test_feature = experiment_feature(2401:3000,speicfy_range);
test_label = experiment_label(2401:3000,:);

accuracy_c_v_array = ones(21, 19);
for i = 1:21
	c = 2 ^ (i - 6);
	for j = 1:19
		v = 2 ^ (j - 16);

		str = ['-c ', num2str(c), ' -g ', num2str(v)];

		model_post = svmtrain(train_label, train_feature, str);
		%model_post = svmtrain(train_label, train_feature, '-c 2048 -g 0.25');
		%model_post = svmtrain([train_label; val_label], [train_feature; val_feature], '-c 1 -g 0.07');
		[predict_label, accuracy_temp, prob_est] = svmpredict(test_label, ...
		        test_feature, model_post);

		accuracy_c_v_array(i, j) = accuracy_temp(1);
	end
end



