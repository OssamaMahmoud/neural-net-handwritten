%use logisticRegressionWeights, the linear classifier, to compare results from p5b the random
%classifier

load A1

[X_train_new, Y_train_new] = transformLabel(X_train, Y_train, 4, 9);
[X_test_new, Y_test_new] = transformLabel(X_test, Y_test, 4, 9);


wInit = ones( size(X_train_new,2 )+1, 1);

%find training error
w = logisticRegressionWeights(X_train_new, Y_train_new, 30,wInit, 0.1);
Y_train_C = linearClassify(w, X_train_new);
[err_train, ~] = errorRate(Y_train_C, Y_train_new); 
display (err_train);

%find test error
Y_test_C = linearClassify(w, X_test_new);
[err, ~] = errorRate(Y_test_C, Y_test_new);
display (err);


%Discussion
%   both the training error and test error are significantly lower than in
%   the random weight calculations(p5).
%err_train =
%    0.0356
%err =
%    0.0627