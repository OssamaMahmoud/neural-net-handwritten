%used to examine several values of k 

load A1

[err, ~] = errorRate(kNNClassify(X_train, Y_train,X_test, 1), Y_test);
fprintf('k = 1');
display(err)
[err, ~] = errorRate(kNNClassify(X_train, Y_train,X_test, 3), Y_test);
fprintf('k = 3');
display(err)

[err, ~] = errorRate(kNNClassify(X_train, Y_train,X_test, 5), Y_test);
fprintf('k = 5');
display(err)

[err, ~] = errorRate(kNNClassify(X_train, Y_train,X_test, 7), Y_test);
fprintf('k = 7');
display(err)

% Discussion
%test error values for various k values, for matrix X_train
% [err, _] = errorRate(kNNClassify(X_train, Y_train,X_test, 1), Y_test)
% k : err
% 1 : 0.0835
% 3 : 0.0790
% 5 : 0.0850
% 7 : 0.0885
%   it appears the best value for k is 3, as above 3 the error increases
%   and below 3 it also increases

% Confusion matrix of k = 5 
%   4 is being incorrectly classified as 9, it occured 12 times
%   this relationship is somewhat symetric as 9 is being classified by 
%   9 is sometimes confused as 4, 9 times
%   7 is being classified incoorectly as 1
%   this relationship is not symetric since
%   1 is being classified as 7, 0 times
