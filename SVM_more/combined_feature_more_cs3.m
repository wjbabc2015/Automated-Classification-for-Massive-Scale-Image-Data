%{
    It is developed for case study 3. Adding features to obtain the feature
    combination(Testing version, no longer use)
%}

base_range = [2,13,16,4,1,6,14,12,17,10,5,7];
new_range = [1:14,16:17];

accuracy_two_com = ones(2, 16);

    for j = 1:16

        if ~any(base_range == new_range(j))
            specify_range = [base_range,new_range(j)];
            
            %Create training and testing data set
            train_feature = training_feature(:,specify_range);
            test_feature = testing_feature(:,specify_range);
            
            model_post = svmtrain(train_label, train_feature, '-c 2 -g 4');
    		[predict_label, accuracy_2_result, prob_est] = svmpredict(test_label, ...
    		        test_feature, model_post);

    		accuracy_two_com(2, j) = accuracy_2_result(1);
        end

        accuracy_two_com(1,j) = new_range(j);

    end