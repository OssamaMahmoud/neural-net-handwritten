function [ w, leastErr ] = randomWeights( X_train, Y_train, iterNum )
%Tests random weights and selects the w that has the least error
%X_train is the sample features as rows, y_train is the true value and
%iterNUm is the number of values to test for the weights

    %number of features + 1 for the bias 
    lenW = size( X_train, 2) + 1;
    
    leastErr = 1;
    for i = 1:iterNum
        wCur = randn(lenW, 1);
        %use p2 and p4 to find the error of every iteration
        estLabels = linearClassify(wCur, X_train);
        [err, ~ ] = errorRate(estLabels, Y_train);
        if err < leastErr
            leastErr = err;
            w = wCur;
        end
    end
    
    
end

