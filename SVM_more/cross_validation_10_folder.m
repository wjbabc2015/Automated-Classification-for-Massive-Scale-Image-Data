train_data_size = size(train_feature, 1);

section_size = train_data_size / 10;

section_label = ones(section_size,1,10);
section_feature = ones(section_size,size(speicfy_range, 2),10);

group_separate = train_data_size / 3;

CV_cell_label = train_label(1:group_separate,:);
CV_cell_inst = train_feature(1:group_separate,:);

CV_debris_label = train_label(group_separate + 1:group_separate * 2,:);
CV_debris_inst = train_feature(group_separate + 1:group_separate * 2,:);

CV_strip_label = train_label(2 * group_separate + 1:train_data_size,:);
CV_strip_inst = train_feature(2 * group_separate + 1:train_data_size,:);

single_group_size = section_size / 3;

for i=1:10
    sub_part = [(single_group_size*(i-1)+1):(single_group_size*i)];
    section_label(:,:,i) = [CV_cell_label(sub_part);CV_debris_label(sub_part);CV_strip_label(sub_part)];
    section_feature(:,:,i) = [CV_cell_inst(sub_part,:);CV_debris_inst(sub_part,:);CV_strip_inst(sub_part,:)];
end

accuracy_10_folder = ones(10, 1);
train_feature_10 = ones(section_size * 9, size(speicfy_range, 2),10);
train_label_10 = ones(section_size * 9, 1, 10);


train_label_10(:,:,1) = [section_label(:,:,2);section_label(:,:,3);section_label(:,:,4);...
    section_label(:,:,5);section_label(:,:,6);section_label(:,:,7);...
    section_label(:,:,8);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,1) = [section_feature(:,:,2);section_feature(:,:,3);section_feature(:,:,4);...
    section_feature(:,:,5);section_feature(:,:,6);section_feature(:,:,7);...
    section_feature(:,:,8);section_feature(:,:,9);section_feature(:,:,10)];

train_label_10(:,:,2) = [section_label(:,:,1);section_label(:,:,3);section_label(:,:,4);...
    section_label(:,:,5);section_label(:,:,6);section_label(:,:,7);...
    section_label(:,:,8);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,2) = [section_feature(:,:,1);section_feature(:,:,3);section_feature(:,:,4);...
    section_feature(:,:,5);section_feature(:,:,6);section_feature(:,:,7);...
    section_feature(:,:,8);section_feature(:,:,9);section_feature(:,:,10)];

train_label_10(:,:,3) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,4);...
    section_label(:,:,5);section_label(:,:,6);section_label(:,:,7);...
    section_label(:,:,8);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,3) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,4);...
    section_feature(:,:,5);section_feature(:,:,6);section_feature(:,:,7);...
    section_feature(:,:,8);section_feature(:,:,9);section_feature(:,:,10)];

train_label_10(:,:,4) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,5);section_label(:,:,6);section_label(:,:,7);...
    section_label(:,:,8);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,4) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,5);section_feature(:,:,6);section_feature(:,:,7);...
    section_feature(:,:,8);section_feature(:,:,9);section_feature(:,:,10)];

train_label_10(:,:,5) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,4);section_label(:,:,6);section_label(:,:,7);...
    section_label(:,:,8);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,5) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,4);section_feature(:,:,6);section_feature(:,:,7);...
    section_feature(:,:,8);section_feature(:,:,9);section_feature(:,:,10)];

train_label_10(:,:,6) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,4);section_label(:,:,5);section_label(:,:,7);...
    section_label(:,:,8);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,6) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,4);section_feature(:,:,5);section_feature(:,:,7);...
    section_feature(:,:,8);section_feature(:,:,9);section_feature(:,:,10)];

train_label_10(:,:,7) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,4);section_label(:,:,5);section_label(:,:,6);...
    section_label(:,:,8);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,7) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,4);section_feature(:,:,5);section_feature(:,:,6);...
    section_feature(:,:,8);section_feature(:,:,9);section_feature(:,:,10)];


train_label_10(:,:,8) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,4);section_label(:,:,5);section_label(:,:,6);...
    section_label(:,:,7);section_label(:,:,9);section_label(:,:,10)];


train_feature_10(:,:,8) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,4);section_feature(:,:,5);section_feature(:,:,6);...
    section_feature(:,:,7);section_feature(:,:,9);section_feature(:,:,10)];

train_label_10(:,:,9) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,4);section_label(:,:,5);section_label(:,:,6);...
    section_label(:,:,7);section_label(:,:,8);section_label(:,:,10)];


train_feature_10(:,:,9) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,4);section_feature(:,:,5);section_feature(:,:,6);...
    section_feature(:,:,7);section_feature(:,:,8);section_feature(:,:,10)];

train_label_10(:,:,10) = [section_label(:,:,1);section_label(:,:,2);section_label(:,:,3);...
    section_label(:,:,4);section_label(:,:,5);section_label(:,:,6);...
    section_label(:,:,7);section_label(:,:,8);section_label(:,:,9)];


train_feature_10(:,:,10) = [section_feature(:,:,1);section_feature(:,:,2);section_feature(:,:,3);...
    section_feature(:,:,4);section_feature(:,:,5);section_feature(:,:,6);...
    section_feature(:,:,7);section_feature(:,:,8);section_feature(:,:,9)];

label_index_cell = ones(1, 1);
label_index_debris = ones(1, 1) .* 2;
label_index_strip = ones(1, 1) .* 3;

for j = 1:10

    str = '-c 2 -g 4';

    [predict_label, accuracy] = svm_func(train_label_10(:,:,j), train_feature_10(:,:,j), section_label(:,:,j),...
                     section_feature(:,:,j), str, false);

    accuracy_10_folder(j) = accuracy;

    [c, d, s] = label_com(predict_label, section_label(:,:,j));

    label_index_cell = [label_index_cell;c + single_group_size * (j - 1)];
    label_index_debris = [label_index_debris;d + single_group_size * (j - 1)];
    label_index_strip = [label_index_strip;s + single_group_size * (j - 1)];

end

wrong_label_index = [label_index_cell(2:end,:);label_index_debris(2:end,:) + group_separate; label_index_strip(2:end,:) + group_separate * 2];

accuracy_ave_10CV = sum(accuracy_10_folder)/10;


