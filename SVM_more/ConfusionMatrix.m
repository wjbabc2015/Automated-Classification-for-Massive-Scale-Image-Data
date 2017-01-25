%{
	Build confusion matrix based on testing label and predicted label	
%}

function [cMatrix] = ConfusionMatrix(test_label, predict_label)

	%Get the number of classes
	matrix_size = max(test_label);

	%Initiate the cMatrix
	cMatrix = zeros(matrix_size, matrix_size);

	for i = 1:size(test_label,1)
		cMatrix(test_label(i),predict_label(i)) ++;
	end

end
