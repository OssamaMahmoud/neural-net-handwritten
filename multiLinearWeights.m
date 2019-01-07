function [ W ] = mutliLinearWeights( X_train, Y_train, iterNum, WInit, alpha )
%finds weights W for linear multiclass classficication
%using softmax single rule loss function, given initial weights and
%X_train being the samples and Y_train as their true labels, and alpha as
%the training rate

     %augment X to include bias
    len = size(X_train, 1);
    X = [ ones(len, 1), X_train ];
    W = WInit;
    for j = 1:iterNum
        for i = 1:len 
           %vector with current features
           xi = transpose(X(i,:));
           prod = W * xi;
           %if calculate loss
           yi = zeros(size(W, 1), 1);
           yi(Y_train(i)) = 1;
           loss = (yi - softmax(prod)) * transpose(xi);
           
           W = W + alpha .* loss;
        end     
    end


end

function Y = softmax(X)
    expsum = sum(arrayfun(@exp, X));
    Y = arrayfun(@exp, X) ./ expsum;
    


end
