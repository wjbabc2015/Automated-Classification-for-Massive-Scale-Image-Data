%{
    It is developed for p-polarized features in case study 3. Adding features to 
    obtain the featurecombination
%}

%base_range = [];
base_range = [11,13,4,16,2,5,17,1,7,10,8];
new_range = [1:14,16:17];

accuracy_two_com_s = ones(2, 16);

    for j = 1:16

        if ~any(base_range == new_range(j))
            specify_range = [base_range,new_range(j)];
            
            %Create training and testing data set
            train_feature = training_feature(:,specify_range);
            test_feature = testing_feature(:,specify_range);
            
            str = '-c 2 -g 4';

            [predict_label, accuracy] = svm_func(train_label, train_feature, test_label, test_feature, str, false);

    		accuracy_two_com_s(2, j) = accuracy;
        end

        accuracy_two_com_s(1,j) = new_range(j);

    end