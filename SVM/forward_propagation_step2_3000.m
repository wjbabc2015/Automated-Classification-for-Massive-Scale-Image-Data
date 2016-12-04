%{
This script is used to examine the accuracy of classifier trained on data
with three feature parameters
%}
base_range = [22,36;...
                21,22;...
                29,32;...
                22,29;...
                6,21;...
                21,26];
new_range = [1:14,16:17,21:34,36:37];

accuracy_com = ones(29, 6);

for i = 1:6
    for j = 1:32
        specify_range = [base_range(i,:),new_range(j)];
        train_feature = experiment_feature(1:2000,specify_range);
		train_label = experiment_label(1:2000,:);
        test_feature = experiment_feature(2401:3000,specify_range);
		test_label = experiment_label(2401:3000,:);
        
        model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
		[predict_label, accuracy_temp, prob_est] = svmpredict(test_label, ...
		        test_feature, model_post);
		accuracy_com(j, i) = accuracy_temp(1);
    end
end

[M, I] = max(accuracy_com);