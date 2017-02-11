clear;
%Load data from csv file
load('training_data');
load('testing_data');

%Initiate scaled training and testing data sets
training_data = [experiment_training_data(:,1),ones(size(experiment_training_data, 1),40)];
testing_data = [experiment_testing_data(:,1),ones(size(experiment_testing_data, 1),40)];


for i=2:41
    I = [experiment_training_data(:,i);experiment_testing_data(:,i)];
    if max(I(:)-min(I(:))) ~= 0
        training_data(:,i) = (experiment_training_data(:,i)-min(I(:))) ./ (max(I(:)-min(I(:))));
        testing_data(:,i) = (experiment_testing_data(:,i)-min(I(:))) ./ (max(I(:)-min(I(:))));
    end
end


training_feature = training_data(:,2:41);
train_label = training_data(:,1);

testing_feature = testing_data(:,2:41);
test_label = testing_data(:,1);
