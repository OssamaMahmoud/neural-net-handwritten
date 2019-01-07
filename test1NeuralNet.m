%p10c
%use p10a and b to train the neural net on the data set and report error
load A1

H = (100);
regularizerWeight = 0.8;
[net,valErr] = buildNeuralNet(X_train,Y_train, H, regularizerWeight);
display(valErr);
[err, CONF] = netClassify(X_test, Y_test, net);
display(err);

%Discussion
%10c
%   valErr =
%       0.0547
%   err = 
%       0.0915
%   test errors are less than both the perceptron and the softmax linear
%   classifiers, achieving better classiification 



