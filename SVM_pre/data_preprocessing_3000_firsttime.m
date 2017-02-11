clear;
%Load data from csv file
%experiment_data = csvread('3000_data.csv');
[cell_data, debris_data, strip_data] = read_csv_file...
('Cell_old.csv', 1, 'Debris_old.csv', 2, 'Strip_old.csv', 3);

cell_data_index = randperm(size(cell_data, 1), 1000);

cell_vector = cell_data(cell_data_index, :);

debris_data_index = randperm(size(debris_data, 1), 1000);

debris_vector = debris_data(debris_data_index, :);

strip_data_index = randperm(size(strip_data, 1), 1000);

strip_vector = strip_data(strip_data_index, :);

%Divid data into training and testing data sets
experiment_training_data = [cell_vector(1:700,:);debris_vector(1:700,:);strip_vector(1:700,:)];
experiment_testing_data = [cell_vector(801:1000,:);debris_vector(801:1000,:);strip_vector(801:1000,:)];

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
