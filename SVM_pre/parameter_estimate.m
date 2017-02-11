

accuracy_c_v_array = ones(21, 19);
for i = 1:21
	c = 2 ^ (i - 6);
	for j = 1:19
		v = 2 ^ (j - 16);

		str = ['-c ', num2str(c), ' -g ', num2str(v)];

		model_p = svmtrain(train_label, train_feature, str);
		[predict_label, accuracy, prob_est] = svmpredict(test_label, test_feature, model_p);


		accuracy_c_v_array(i, j) = accuracy(1);
	end
end

[M_RBF, I_RBF] = max(accuracy_c_v_array);

[R, V] = max(M_RBF);

C = I_RBF(V) - 6;

V = V - 16;

fprintf('The value of C: 2^%i\n', C);
fprintf('The value of V: 2^%i\n', V);



