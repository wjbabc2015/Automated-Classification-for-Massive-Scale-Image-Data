%{
This script is used to examine the accuracy of classifier trained on data
with three feature parameters
%}
base_range = [29,22;...
                21,14;...
                27,22;...
                21,32;...
                36,22;...
                21,22];
new_range = [1:14,16:17,21:34,36:37];

accuracy_com = ones(29, 7);

for i = 1:6
    for j = 1:32
        specify_range = [base_range(i,:),new_range(j)];
        
        train_feature = train_feature_200(:,specify_range);

        test_feature = test_feature_200(:,specify_range);
        
        model_post = svmtrain(train_label_200, train_feature, '-c 1 -g 0.07');
		[predict_label, accuracy_temp, prob_est] = svmpredict(test_label_200, ...
		        test_feature, model_post);
		accuracy_com(j, i+1) = accuracy_temp(1);
        accuracy_com(j,1) = new_range(j);
    end
end

[M, I] = max(accuracy_com);