%{
This script is used to examine the accuracy of classifier trained on data
with more than five feature parameters
%}
%{
%six features examination
base_range = [22,36,23,16,7;...
                29,32,6,17,1;...
                22,29,6,17,16];
accuracy_com = ones(29, 4);

%seven features examination
base_range = [22,36,23,16,7,12;...
                29,32,6,17,1,16;...
                22,29,6,17,16,28];
accuracy_com = ones(29, 4);

%eight features examination
base_range = [22,36,23,16,7,12,2;...
                29,32,6,17,1,16,5;...
                22,29,6,17,16,28,2];
accuracy_com = ones(29, 4);
%}
%{
%nine features examination
base_range = [22,36,23,16,7,12,2,26;...
                29,32,6,17,1,16,5,33;...
                22,29,6,17,16,28,2,7];
accuracy_com = ones(29, 4);


%ten features examination
base_range = [22,36,23,16,7,12,2,26,37;...
                29,32,6,17,1,16,5,33,11];
accuracy_com = ones(29, 3);


%eleven features examination
base_range = [29,32,6,17,1,16,5,33,11,3];
accuracy_com = ones(29, 2);
%}

%twelve features examination
base_range = [29,32,6,17,1,16,5,33,11,3,34];
accuracy_com = ones(29, 2);

new_range = [1:14,16:17,21:34,36:37];

for i = 1:size(base_range, 1)
    for j = 1:32
        specify_range = [base_range(i,:),new_range(j)];
        train_feature = experiment_feature(1:2000,specify_range);
		train_label = experiment_label(1:2000,:);
        test_feature = experiment_feature(2401:3000,specify_range);
		test_label = experiment_label(2401:3000,:);
        
        model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
		[predict_label, accuracy_temp, prob_est] = svmpredict(test_label, ...
		        test_feature, model_post);
		accuracy_com(j, i+1) = accuracy_temp(1);
        accuracy_com(j, 1) = new_range(j);
    end
end

[M, I] = max(accuracy_com);