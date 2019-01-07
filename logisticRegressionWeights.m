function [ w ] = logisticRegressionWeights( X_train, Y_train, iterNum, wInit, alpha )
%finds weights w using logistic regression, given initial weights and
%X_train being the samples and Y_train as their true labels, and alpha as
%the training rate

    %create augmented feature matrix
    Y_train(Y_train == 2) = 0;
    Z = [ ones( size(X_train, 1) ,  1 ) , X_train ];
    a = wInit;
    for i = 1:iterNum
        %apply per loss function
        sig = arrayfun(@mySig, Z * a);
        prod = repmat( (Y_train - sig ), 1, size(Z, 2) ) .* Z;
        
        %sum the loss to find total loss
        loss = alpha .* transpose( sum(prod, 1) );
        %correct class of this is either 0 or 1
        %display(sum(loss, 1));
        a = a + loss;
    end
    w = a;

end

function y = mySig(x)
    y = 1./(1 + exp(-1.*(x)));
    %y = sigmf(x, [1 0]);

end

