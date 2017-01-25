%{
	Feature removing approach to get the feature combination
%}
speicfy_range_pre = [23,16,4,25,27,2,30,10,5,3,26,33,1,22,37,14,36,28,29,4];


accuracy_delete_final = ones(2, 37);

for i = 1:37
	if find(speicfy_range_pre == i)
		speicfy_range = speicfy_range_pre(find(speicfy_range_pre~=i));
		
		%Create training and testing data set
        train_feature = training_feature(:,speicfy_range);
        test_feature = testing_feature(:,speicfy_range);

        str = '-c 2 -g 4';

		[predict_label, accuracy] = svm_func(train_label, train_feature, test_label, test_feature, str, false);

		accuracy_delete_final(2,i) = accuracy;
    end
    accuracy_delete_final(1,i) = i;
end

[M_d, I_d] = max(accuracy_delete_final(speicfy_range));

%{
speicfy_range_pre = [1,3,4,5,6,7,8,9,10,11,12,13,14,16,17,...
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,...
                37];
%}