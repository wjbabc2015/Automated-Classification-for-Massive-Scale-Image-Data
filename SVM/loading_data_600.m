%clear;
[cell_data, debris_data, strip_data]=read_csv_file(...
        'Cell_237.csv', 1, 'Debris_237.csv', 2, 'Strip_237.csv', 3);
experiment_data = [cell_data(1:200,:);debris_data(1:200,:);strip_data(1:200,:)];
experiment_data_scaled = [experiment_data(:,1),ones(600,40)];

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
experiment_label = experiment_data_scaled(:,1);

train_feature_200 = [experiment_feature(1:120,:);experiment_feature(201:320,:);experiment_feature(401:520,:)];
train_label_200 = [experiment_label(1:120,1);experiment_label(201:320,1);experiment_label(401:520,1)];

val_feature_200 = [experiment_feature(121:140,:);experiment_feature(321:340,:);experiment_feature(521:540,:)];
val_label_200 = [experiment_label(121:140,1);experiment_label(321:340,1);experiment_label(521:540,1)];

test_feature_200 = [experiment_feature(141:200,:);experiment_feature(341:400,:);experiment_feature(541:600,:)];
test_label_200 = [experiment_label(141:200,1);experiment_label(341:400,1);experiment_label(541:600,1)];