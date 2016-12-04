%{
This script is used to examine the accuracy of classifier trained on data
with more than five feature parameters
%}
%{
%six features examination
base_range = [29,22,13,3,32;...
                36,22,4,26,28];
accuracy_com = ones(32, 3);

%seven features examination
base_range = [29,22,13,3,32,27];
accuracy_com = ones(32, 2);
%}

%eight features examination
base_range = [29,22,13,3,32,27,24];
accuracy_com = ones(32, 2);

%{
%nine features examination
base_range = [29,22,13,3,32,27,24,34];
accuracy_com = ones(32, 2);


%ten features examination
base_range = [29,22,13,3,32,27,24,34,28];
accuracy_com = ones(32, 2);



%}

new_range = [1:14,16:17,21:34,36:37];

for i = 1:size(base_range, 1)
    for j = 1:32
        specify_range = [base_range(i,:),new_range(j)];
         
        train_feature = train_feature_200(:,specify_range);

        test_feature = test_feature_200(:,specify_range);
        
        model_post = svmtrain(train_label_200, train_feature, '-c 1 -g 0.07');
		[predict_label, accuracy_temp, prob_est] = svmpredict(test_label_200, ...
		        test_feature, model_post);
		accuracy_com(j, i+1) = accuracy_temp(1);
        accuracy_com(j, 1) = new_range(j);
    end
end

[M, I] = max(accuracy_com);