function [cell_label_index, debris_label_index, strip_label_index] = label_com (predict_label, test_label)

matrix_size = size(test_label, 1);

cell_label_index = zeros(1, 1);
debris_label_index = zeros(1, 1);
strip_label_index = zeros(1, 1);

c = 1; d = 1; s = 1;

	for i = 1:matrix_size

		vale = test_label(i);

		switch vale
			case 1
				if predict_label(i) ~= vale
					cell_label_index (c,1) = i;
					c = c + 1;
				end

			case 2
				if predict_label(i) ~= vale
					debris_label_index (d,1) = i - matrix_size / 3;
					d = d + 1;
				end

			otherwise
				if predict_label(i) ~= vale
					strip_label_index (s,1) = i - (matrix_size / 3) * 2;
					s = s + 1;
				end
		end
	end
end