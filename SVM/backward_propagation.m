%speicfy_range_pre = [1:14,16:17,21:34,36:37];%total 32 features
%speicfy_range_pre = [1:14,16:17,21:34,37];%1 feature removed
%speicfy_range_pre = [1,3:14,16:17,21:34,37];%2 features removed (s-CON)
%speicfy_range_pre = [1:14,16:17,21:27,29:34,37];%2 features removed
%(p-SVA)

%first scenario
%speicfy_range_pre = [1,4:14,16:17,21:34,37];%3 features removed (s-COR)
%speicfy_range_pre = [1,4:14,16:17,21:27,29:34,37];%4 features removed (p-SVA)
%speicfy_range_pre = [1,4:14,16:17,22:27,29:34,37];%5 features removed (p-ASM)
%speicfy_range_pre = [1,4:8,10:14,16:17,22:27,29:34,37];%6 features removed (s-ENT)
%speicfy_range_pre = [1,4:8,10:14,16:17,22:27,29:33,37];%7 features removed (p-CLP)
%speicfy_range_pre = [1,4:8,10:14,16:17,22:27,29,31:33,37];%8 features removed (p-DEN)
%speicfy_range_pre = [1,4:8,10:12,14,16:17,22:27,29,31:33,37];%9 features removed (s-CLS)
%speicfy_range_pre = [1,4:8,10:12,16:17,22:27,29,31:33,37];%10 features removed (s-CLP)
%speicfy_range_pre = [1,4:7,10:12,16:17,22:27,29,31:33,37];%11 features removed (s-SVA)
%the last removement is p-VAR

%second scenario
%speicfy_range_pre = [1,3:14,16:17,21:23,25:34,37];%3 features removed (p-VAR)
%speicfy_range_pre = [1,3:14,16:17,21:23,25:32,34,37];%4 features removed (p-CLS)
%speicfy_range_pre = [1,3:14,16:17,21:23,25:27,29:32,34,37];%5 features removed (p-SVA)
%speicfy_range_pre = [1,3:13,16:17,21:23,25:27,29:32,34,37];%6 features removed (s-CLP)
%speicfy_range_pre = [1,3:13,16:17,21:23,25:27,29:32,37];%7 features removed (p-CLP)

%third scenario
%speicfy_range_pre = [1,3:14,16:17,21:32,34,37];%3 features removed (p-CLS)
%speicfy_range_pre = [1,3:14,16:17,21:27,29:32,34,37];%4 features removed (p-SVA)
%speicfy_range_pre = [1,3:13,16:17,21:27,29:32,34,37];%5 features removed (s-CLP)
%speicfy_range_pre = [1,3:13,16:17,21:27,29:32,37];%6 features removed (p-CLP)
%speicfy_range_pre = [1,4:13,16:17,21:27,29:32,37];%7 features removed (s-COR)
%speicfy_range_pre = [1,4:12,16:17,21:27,29:32,37];%8 features removed (s-CLS)
%speicfy_range_pre = [1,4:12,16:17,22:27,29:32,37];%9 features removed (p-ASM)
speicfy_range_pre = [4:12,16:17,22:27,29:32,37];%10 features removed (s-ASM)

accuracy_delete_final = ones(37, 2);

for i = 1:37
	if find(speicfy_range_pre == i)
		speicfy_range = speicfy_range_pre(find(speicfy_range_pre~=i));
		train_feature = experiment_feature(1:2000,speicfy_range);
		train_label = experiment_label(1:2000,:);
		test_feature = experiment_feature(2401:3000,speicfy_range);
		test_label = experiment_label(2401:3000,:);

		model_post = svmtrain(train_label, train_feature, '-c 1 -g 0.07');
		%model_post = svmtrain([train_label; val_label], [train_feature; val_feature], '-c 1 -g 0.07');
		[predict_label, accuracy_delete_temp, prob_est] = svmpredict(test_label, ...
		        test_feature, model_post);

		accuracy_delete_final(i,2) = accuracy_delete_temp(1);
    end
    accuracy_delete_final(i,1) = i;
end

[M_d, I_d] = max(accuracy_delete_final(speicfy_range));

%{
speicfy_range_pre = [1,3,4,5,6,7,8,9,10,11,12,13,14,16,17,...
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,...
                37];
%}