
%test the error for our random weight training funciton  
%for digits 3 and 8 for different values of iterNum
load A1
[X_train_new, Y_train_new] = transformLabel(X_train, Y_train, 4, 9);
[X_test_new, Y_test_new] = transformLabel(X_test, Y_test, 4, 9);


w = randomWeights(X_train_new, Y_train_new, 100);


Y_test_C = linearClassify(w, X_test_new);
Y_train_C = linearClassify(w, X_train_new);
[err_train, ~] = errorRate(Y_train_C, Y_train_new); 
display (err_train)

[err, ~] = errorRate(Y_test_C, Y_test_new);
display (err);

w = randomWeights(X_train_new, Y_train_new, 1000);
Y_train_C = linearClassify(w, X_train_new);
[err_train, ~] = errorRate(Y_train_C, Y_train_new); 
display (err_train)

Y_test_C = linearClassify(w, X_test_new);
[err, ~] = errorRate(Y_test_C, Y_test_new);
display (err);

w = randomWeights(X_train_new, Y_train_new, 10000);
Y_train_C = linearClassify(w, X_train_new);
[err_train, ~] = errorRate(Y_train_C, Y_train_new); 
display (err_train)

Y_test_C = linearClassify(w, X_test_new);
[err, ~] = errorRate(Y_test_C, Y_test_new);
display (err);

%Discussion
%   the test error seems to decrease as we use more iterations. But since
%   this is a random process, better classification is not certain with
%   more iterations. this makes this classifier not bery realiable.
%   It also appears that the training error is less than the test error
%   This could be due to us selecting the w that lowers the training error
%   As expected the training error is reduced as the number of iters
%   increase
%
%100 iter
%err_train =
%    0.2901
%err =
%    0.3709
%1000 iter
%err_train =
%    0.2262
%err =
%    0.3083
%10000 iter
%err_train =
%    0.1455
%err =
%    0.2381




