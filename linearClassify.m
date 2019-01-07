function [ C ] = linearClassify( w, X )
%Perfroms linear classification on a 2 class classification problem
%X is the matrix containing the sample features as rows
% w is the weight vector 

    %augment X to include bias
    len = size(X, 1);
    X = [ ones(len, 1), X ];
    
    %find wX for samples
    prod = X * w ;
    
    %use simple sign function to achieve classification
    prod( prod > 0) = 1;
    prod( prod < 0) = 2;
    
    C = prod;    
    

end

