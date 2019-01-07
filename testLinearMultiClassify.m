%train perceptron with function p8 and 100 iterations

load A1

%define WInit with random values for size d +1, m
WInit = randn(10, size(X_train, 2) + 1); 

W = linearWeights(X_train, Y_train, 100, WInit, 0.01);


C = linearMultiClassify(W, X_test);

C_r = linearMultiClassify(W, X_train);

[err, conf] = errorRate(C, Y_test);
[err_r, conf_r] = errorRate(C_r, Y_train);

%Discussion
% training error: 0.0646
% test error = 0.1725
% the test error is a liittle higher than in the KNN-classifier
% looking at the confusion matrix it appears the two most confused values
% are: 9 being confused for an 7, but not vice versa 
% this is a bit different than our KNN classifier from 3b, 
% as 9 was being confused with 4 often in 3b
% 