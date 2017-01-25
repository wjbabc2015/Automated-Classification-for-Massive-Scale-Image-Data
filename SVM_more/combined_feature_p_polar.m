%{
    It is developed for p-polarized features in case study 3. Adding features to 
    obtain the featurecombination 
%}

%base_range = [36,12,23,6,28,4,25,27,2,29,16,30,34,10,5,3,21,1,31];
base_range = [36];
new_range = [21:34,36:37];

accuracy_two_com_p = ones(2, 16);

    for j = 1:16

        if ~any(base_range == new_range(j))
            specify_range = [base_range,new_range(j)];
            
            %Create training and testing data set
            train_feature = training_feature(:,specify_range);
            test_feature = testing_feature(:,specify_range);
            
            str = '-c 2 -g 4';

            [predict_label, accuracy] = svm_func(train_label, train_feature, test_label, test_feature, str, false);


    		accuracy_two_com_p(2, j) = accuracy;
        end

        accuracy_two_com_p(1,j) = new_range(j);

    end