function [cell_data, debris_data, strip_data] = read_csv_file...
    (cell_csv, label_cell, debris_csv, label_debris, strip_csv, label_strip)

cell_feature = csvread(cell_csv, 1, 1);

label_set_cell = ones(size(cell_feature, 1), 1);

cell_label = label_set_cell .* label_cell;

cell_data = [cell_label, cell_feature];

debris_feature = csvread(debris_csv, 1, 1);

label_set_debris = ones(size(debris_feature, 1), 1);

debris_label = label_set_debris .* label_debris;

debris_data = [debris_label, debris_feature];

strip_feature = csvread(strip_csv, 1, 1);

label_set_strip = ones(size(strip_feature, 1), 1);

strip_label = label_set_strip .* label_strip;

strip_data = [strip_label, strip_feature];

end