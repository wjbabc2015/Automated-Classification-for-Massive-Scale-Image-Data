%{
This script is used to examine the accuracy by addding new feature to
feature p-ASM, p-ENT, p-MAP
%}
base_range = [21,29,36];
new_range = [1:14,16:17,22:28,30:34,37];

accuracy_two_com = ones(29, 3);

for i = 1:3
    for j = 1:29
        specify_range = [base_range(i),new_range(j)];
        train_feature = experiment_feature(1:2000,specify_range);
		train_label = experiment_label(1:2000,:);
        test_feature = experiment_feature(2401:3000,specify_range);
		test_label = experiment_label(2401:3000,:);
        
        model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
		[predict_label, accuracy_2_result, prob_est] = svmpredict(test_label, ...
		        test_feature, model_post);
		accuracy_two_com(j, i) = accuracy_2_result(1);
    end
end

[M_two, I_two] = max(accuracy_two_com);