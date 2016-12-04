speicfy_range = [3,13,22,24,27,29,32,34];

section_label = ones(72,1,5);
section_feature = ones(72,size(speicfy_range, 2),5);

cell_feature_3 = train_feature_200(1:120,speicfy_range);
debris_feature_3 = train_feature_200(121:240,speicfy_range);
strip_feature_3 = train_feature_200(241:360,speicfy_range);

for i=1:5
    section_label(:,:,i) = ...
        [train_label_200(1:24,1);train_label_200(121:144,1);train_label_200(241:264,1)];
    section_feature(:,:,i) = ...
        [cell_feature_3((24*(i-1)+1):(24*i),:);...
         debris_feature_3((24*(i-1)+1):(24*i),:);...
         strip_feature_3((24*(i-1)+1):(24*i),:)];
end

accuracy_5_folder = ones(5, 1);
train_feature_5 = ones(288,size(speicfy_range, 2),5);
train_label_5 = ones(288, 1, 5);


train_label_5(:,:,1) = [section_label(:,:,2);section_label(:,:,3);section_label(:,:,4);...
    section_label(:,:,5)];


train_feature_5(:,:,1) = [section_feature(:,:,2);section_feature(:,:,3);section_feature(:,:,4);...
    section_feature(:,:,5)];

train_label_5(:,:,2) = [section_label(:,:,1);section_label(:,:,3);section_label(:,:,4);...
    section_label(:,:,5)];


train_feature_5(:,:,2) = [section_feature(:,:,1);section_feature(:,:,3);section_feature(:,:,4);...
    section_feature(:,:,5)];

train_label_5(:,:,3) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,4);...
    section_label(:,:,5)];


train_feature_5(:,:,3) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,4);...
    section_feature(:,:,5)];

train_label_5(:,:,4) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,5)];


train_feature_5(:,:,4) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,5)];

train_label_5(:,:,5) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,4)];


train_feature_5(:,:,5) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,4)];


for j = 1:5

    model_post = svmtrain(train_label_5(:,:,j), train_feature_5(:,:,j), '-c 1 -g 0.07');
    [predict_label, accuracy_5_folder_temp, prob_est] = svmpredict(section_label(:,:,j), ...
            section_feature(:,:,j), model_post);
        
    %cMatrix = confusionmat(test_label_10, predict_label);
    %confusion_matrix_plot();

    accuracy_5_folder(j) = accuracy_5_folder_temp(1);
end


