
label_wrong_index_cell_pre = label_index_cell(2:end, :);

wrong_index = randperm(size(label_wrong_index_cell_pre,1), 180);

label_wrong_index_cell = label_wrong_index_cell_pre(wrong_index, :);

label_wrong_index_debirs_pre = label_index_debris(2:end, :);

wrong_index = randperm(size(label_wrong_index_debirs_pre,1), 180);

label_wrong_index_debirs = label_wrong_index_debirs_pre(wrong_index, :) + 1000; 

label_wrong_index_strip_pre = label_index_strip(2:end, :);

wrong_index = randperm(size(label_wrong_index_strip_pre,1), 180);

label_wrong_index_strip = label_wrong_index_strip_pre(wrong_index, :) + 2000;

wrong_index_label = [label_wrong_index_cell;label_wrong_index_debirs;label_wrong_index_strip];

refined_data_index = zeros(1,1);

pos = 1;

for i = 1:size(experiment_data, 1)
    if ~any(wrong_index_label == i)
        refined_data_index(pos,1) = i;
        pos ++;
    end
end

wrong_index_label = sort(wrong_index_label);