function [err, CONF] = errorRate(C, T)
%Calculates error rate and confusion matrix for predicted labels C and True labels T

    m = max(max([C, T]));
    CONF = accumarray( [T, C],ones(size(C, 1), 1), [m, m] );
    err = size(find(C - T), 1) / size(C, 1); 
