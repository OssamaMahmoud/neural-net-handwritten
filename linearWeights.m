function [ W ] = linearWeights( X_train, Y_train, iterNum, WInit, alpha )
%finds weights W for linear multiclass classficication
%using Perceprron single rule loss function, given initial weights and
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
           %find index of max value 
           [~,ind] = max(prod);
           %if index is not the same as the true value
           if ind ~= Y_train(i)
               loss = zeros(size(W));
               loss(ind, :) = -1.*transpose(xi); 
               loss(Y_train(i), :) = transpose(xi);
               W = W + alpha .* loss;            
           end
        end     
    end


end

