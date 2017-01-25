clear;
%Load data from csv file
load('training_data');
load('testing_data');

%Modify training and testing data set
pre_training_data = [experiment_training_data(:,1:21);experiment_training_data(:,[1,22:41])];
pre_testing_data = [experiment_testing_data(:,1:21);experiment_testing_data(:,[1,22:41])];

%Initiate scaled training and testing data sets
training_data = [pre_training_data(:,1),ones(size(pre_training_data, 1),20)];
testing_data = [pre_testing_data(:,1),ones(size(pre_testing_data, 1),20)];

for i=2:21
    I = pre_training_data(:,i);
    if max(I(:)-min(I(:))) ~= 0
        training_data(:,i) = (I-min(I(:))) ./ (max(I(:)-min(I(:))));
    end

    J = pre_testing_data(:,i);
    if max(J(:)-min(J(:))) ~= 0
        testing_data(:,i) = (J-min(J(:))) ./ (max(J(:)-min(J(:))));
    end
end

training_feature = training_data(:,2:21);
train_label = training_data(:,1);

testing_feature = testing_data(:,2:21);
test_label = testing_data(:,1);
