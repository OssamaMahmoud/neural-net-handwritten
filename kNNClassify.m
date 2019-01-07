function  C  = kNNClassify( X_train, Y_train, X_test, k)
%Performs kNN classifcation. where X train, Y train, and X test are the training samples, the true class of training samples, and the test samples, respectively, and k
%is the parameter for kNN classifer. The output column vector C stores the classes
%assigned to the test samples.
    
    numValues = size(X_train, 3);
    numTests = size(X_test, 1);
    C = zeros(numTests, 1);
    %go through all samples
    for row = 1:numTests
        %remap current row to be matrix of size like samples
        curTest = repmat(X_test(row, :), numValues, 1);
        %find distance from smaple to neighbours
        diff = abs(X_train - curTest).^2;
        diff = sum(diff, 2);
        %sort list to find closest
        [~, indexes] = sort(diff);
        neighboursInd = indexes(1 : k);
        neighbourLabels = Y_train(neighboursInd);
        result = mode(neighbourLabels);
        C(row) =  result;
    end
end


