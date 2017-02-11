clear;
%Load data from csv file
load('training_data');
load('testing_data');

%Initiate scaled training and testing data sets
training_data = [experiment_training_data(:,1),ones(size(experiment_training_data, 1),40)];
testing_data = [experiment_testing_data(:,1),ones(size(experiment_testing_data, 1),40)];


for i=2:41
    I = experiment_training_data(:,i);
    if max(I(:)-min(I(:))) ~= 0
        training_data(:,i) = (I-min(I(:))) ./ (max(I(:)-min(I(:))));
    end

    J = experiment_testing_data(:,i);
    if max(J(:)-min(J(:))) ~= 0
        testing_data(:,i) = (J-min(J(:))) ./ (max(J(:)-min(J(:))));
    end
end


training_feature = training_data(:,2:41);
train_label = training_data(:,1);

testing_feature = testing_data(:,2:41);
test_label = testing_data(:,1);
