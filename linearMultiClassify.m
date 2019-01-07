function [ C ] = linearMultiClassify( W, X )
%Performs linear multi-class classification given the weight matrix W and features X
%returns the computed labels.

    %augment X to include bias
    len = size(X, 1);
    X = [ ones(len, 1), X ];
    
    %find wX for samples
    prod = W * transpose(X) ;
    %will give us a matrix with each column being the results of each 
    %sample's classigication, we would classiify this class based on which
    %is the highest
    
    %should return a row vector, with the index of maxes of each col
    [~, IndexesOfMax] = max(prod);
    
    C = transpose(IndexesOfMax);
   

end

