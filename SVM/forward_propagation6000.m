%{
This script is used to examine the accuracy by addding new feature to
feature p-ASM, p-ENT, p-MAP
%}
base_range = [1,5,8,16,6,7,12,4,13,14,17];
new_range = [2:14,16:17];

accuracy_two_com = ones(16, 16);

for i = 1:1
    for j = 1:15
        specify_range = [base_range(i,:),new_range(j)];
        train_feature = experiment_feature(1:4000,specify_range);
		train_label = experiment_label(1:4000,:);
        test_feature = experiment_feature(4801:6000,specify_range);
		test_label = experiment_label(4801:6000,:);
        
        model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
		[predict_label, accuracy_2_result, prob_est] = svmpredict(test_label, ...
		        test_feature, model_post);
		accuracy_two_com(j, i+1) = accuracy_2_result(1);
        accuracy_two_com(j,1) = new_range(j);
    end
end

[M_two, I_two] = max(accuracy_two_com);