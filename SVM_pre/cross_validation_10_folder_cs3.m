section_label = ones(420,1,10);
section_feature = ones(420,size(speicfy_range, 2),10);

CV_cell_label_s = train_label(1:700,:);
CV_cell_inst_s = train_feature(1:700,:); 

CV_cell_label_p = train_label(2101:2800,:);
CV_cell_inst_p = train_feature(2101:2800,:); 

CV_debris_label_s = train_label(701:1400,:);
CV_debris_inst_s = train_feature(701:1400,:);

CV_debris_label_p = train_label(2801:3500,:);
CV_debris_inst_p = train_feature(2801:3500,:);

CV_strip_label_s = train_label(1401:2100,:);
CV_strip_inst_s = train_feature(1401:2100,:);

CV_strip_label_p = train_label(3501:4200,:);
CV_strip_inst_p = train_feature(3501:4200,:);

for i=1:10
    sub_part = [(70*(i-1)+1):(70*i)];
    
    CV_cell_label = [CV_cell_label_s(sub_part);CV_cell_label_p(sub_part)];
    CV_cell_inst = [CV_cell_inst_s(sub_part,:);CV_cell_inst_p(sub_part,:)];

    CV_debris_label = [CV_debris_label_s(sub_part);CV_debris_label_p(sub_part)];
    CV_debris_inst = [CV_debris_inst_s(sub_part,:);CV_debris_inst_p(sub_part,:)];

    CV_strip_label = [CV_strip_label_s(sub_part);CV_strip_label_p(sub_part)];
    CV_strip_inst = [CV_strip_inst_s(sub_part,:);CV_strip_inst_p(sub_part,:)];

    section_label(:,:,i) = [CV_cell_label;CV_debris_label;CV_strip_label];
    section_feature(:,:,i) = [CV_cell_inst;CV_debris_inst;CV_strip_inst];
end

accuracy_10_folder = ones(10, 1);
train_feature_10 = ones(3780,size(speicfy_range, 2),10);
train_label_10 = ones(3780, 1, 10);


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


for j = 1:10

   str = '-c 2 -g 4';

    [predict_label, accuracy] = svm_func(train_label_10(:,:,j), train_feature_10(:,:,j), section_label(:,:,j),...
                     section_feature(:,:,j), str, false);

    accuracy_10_folder(j) = accuracy;

end

accuracy_ave_10CV = sum(accuracy_10_folder)/10;


