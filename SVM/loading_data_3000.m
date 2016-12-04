clear;
experiment_data = csvread('3000_data.csv');
experiment_data_scaled = [experiment_data(:,1),ones(3000,40)];

for i=2:18
    I = experiment_data(:,i);
    if max(I(:)-min(I(:))) ~= 0
        experiment_data_scaled(:,i) = (I-min(I(:))) ./ (max(I(:)-min(I(:))));
    end
end

for i=22:38
    I = experiment_data(:,i);
    if max(I(:)-min(I(:))) ~= 0
        experiment_data_scaled(:,i) = (I-min(I(:))) ./ (max(I(:)-min(I(:))));
    end
end

experiment_feature = experiment_data_scaled(:,2:41);
experiment_label = experiment_data(:,1);