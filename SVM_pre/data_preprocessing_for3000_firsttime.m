clear;
%Load data from csv file
[cell_data, debris_data, strip_data] = read_csv_file...
('Cell_old.csv', 1, 'Debris_old.csv', 2, 'Strip_old.csv', 3);

cell_data_index = randperm(size(cell_data, 1), 1000);

cell_vector = cell_data(cell_data_index, :);

debris_data_index = randperm(size(debris_data, 1), 1000);

debris_vector = debris_data(debris_data_index, :);

strip_data_index = randperm(size(strip_data, 1), 1000);

strip_vector = strip_data(strip_data_index, :);

experiment_data_pre = [cell_vector;debris_vector;strip_vector];

experiment_data = [experiment_data_pre(:,1),ones(size(experiment_data_pre,1), 40)];


for i=2:41
    I = experiment_data_pre(:,i);
    if max(I(:)-min(I(:))) ~= 0
        experiment_data(:,i) = (I-min(I(:))) ./ (max(I(:)-min(I(:))));
    end
end
