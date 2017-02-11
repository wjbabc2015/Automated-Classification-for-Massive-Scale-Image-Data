%{
    It is developed for case study 2. Adding features to obtain the feature
    combination
%}

%base_range = [36,12,23,6,28,4,25,27,2,29,16,30,34,10,5,3,21,1,31,33,...
%                11,26,37,24,22,14,9,17,13,7,8];

%base_range = [36,12,23,17,28,4,25,27,2,16,11,34,6,10,1,5];
%base_range = [36,11,6,23,26,29,17,1,25,16,13,10,27,33,14];
%base_range = [36,11,6,23,26,29,17,1,25,16,13,10,27,24,28,21,34,2,33,37,5,32,22];
base_range = [27,5,25,16,23,4,2,6,10,3,30,36,29,34,1,12,28,33,21,26,37];
new_range = [1:14,16:17,21:34,36:37];

%base_range = [2];
%new_range = [1:14,16:17];

accuracy_two_com = ones(2, 32);

    for j = 1:32

        if ~any(base_range == new_range(j))
            specify_range = [base_range,new_range(j)];
            
            %Create training and testing data set
            train_feature = training_feature(:,specify_range);
            test_feature = testing_feature(:,specify_range);
            
            str = '-c 2 -g 4';

            [predict_label, accuracy] = svm_func(train_label, train_feature, test_label, test_feature, str, false);


    		accuracy_two_com(2, j) = accuracy;
        end

        accuracy_two_com(1,j) = new_range(j);

    end