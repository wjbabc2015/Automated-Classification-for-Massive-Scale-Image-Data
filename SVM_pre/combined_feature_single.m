%{
    Examine the accuracy of applying single feature in SVM
%}

accuracy_com_1 = ones(2,17);

for i = 1:17

    %Create training and testing data set
    train_feature = training_feature(:,i);
    test_feature = testing_feature(:,i); 

    str = '-c 2 -g 4';

    [predict_label, accuracy] = svm_func(train_label, train_feature, test_label, test_feature, str, false);

    accuracy_com_1(1,i) = accuracy;

end

for i = 21:37

    %Create training and testing data set
    train_feature = training_feature(:,i);
    test_feature = testing_feature(:,i);

    str = '-c 2 -g 4';

    [predict_label, accuracy] = svm_func(train_label, train_feature, test_label, test_feature, str, false);

    accuracy_com_1(2,i - 20) = accuracy;
end