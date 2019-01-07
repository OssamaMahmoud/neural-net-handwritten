function [ err, CONF ] = netClassify( X_test, Y_test, net)
%Does classification of X using the input net, returning the error and the
%confusion matrixs

    X = transpose(X_test);
    result = net(X);
    C = transpose(vec2ind(result));
    [err, CONF] = errorRate(C, Y_test);

end

