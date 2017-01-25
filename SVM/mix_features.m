clear;
experiment_data_org = csvread('3000_data.csv');
experiment_data = [experiment_data_org(:,1:21);experiment_data_org(:,[1,22:41])];
experiment_data_scaled = [experiment_data(:,1),ones(6000,20)];

for i=2:18
    I = experiment_data(:,i);
    if max(I(:)-min(I(:))) ~= 0
        experiment_data_scaled(:,i) = (I-min(I(:))) ./ (max(I(:)-min(I(:))));
    end
end

experiment_data_mix_index = randperm(size(experiment_data_scaled,1), 6000);
experiment_data_mix = experiment_data_scaled(experiment_data_mix_index,:);
experiment_feature = experiment_data_mix(:,2:21);
experiment_label = experiment_data_mix(:,1);