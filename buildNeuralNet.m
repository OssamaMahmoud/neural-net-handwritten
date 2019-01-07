function [net,valErr] = buildNeuralNet(X_train,Y_train, H, regularizerWeight)
%Builds and trains neural net for multiclass classificatin given the number
%of nodes in each layer as H, and the features and labels and the
%regularization weught
    
    net = patternnet(H);
    net.divideParam.testRatio = 0;
    net.divideParam.valRatio = 0.3;
    net.divideParam.trainRatio = 0.7;
    net.performParam.regularization = regularizerWeight;
    X = transpose(X_train);
    %make matrix Y have d rows, and n cols
    
    %convert from the labels view to a target view
    % with zeros and a 1 at the correct class
    Y = full(ind2vec(transpose(Y_train)));
    [net,tr] = train(net,X,Y);
    
    X_val = X_train(tr.valInd, :);
    Y_val = Y_train(tr.valInd);
    
    [valErr, ~] = netClassify(X_val, Y_val, net);
   

end

