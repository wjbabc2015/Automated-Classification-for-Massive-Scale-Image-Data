%{
This script is used to examine the accuracy by addding new feature to
feature p-ASM, p-ENT, p-MAP
%}
base_range = [21,23,27,29,36];
new_range = [1:14,16:17,22:28,30:34,37];

accuracy_two_com = ones(29, 6);

for i = 1:5
    for j = 1:29
        specify_range = [base_range(i),new_range(j)];
        train_feature = train_feature_200(:,specify_range);

        test_feature = test_feature_200(:,specify_range);

        
        model_post = svmtrain(train_label_200, train_feature, '-c 1 -g 0.07');
		[predict_label, accuracy_2_result, prob_est] = svmpredict(test_label_200, ...
		        test_feature, model_post);
		accuracy_two_com(j, i+1) = accuracy_2_result(1);
        accuracy_two_com(j , 1) = new_range(j);
    end
end

[M_two, I_two] = max(accuracy_two_com);