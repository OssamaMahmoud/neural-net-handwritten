function [ X_out, Y_out ] = transformLabel( X, Y, l1, l2 )
%Transforms a multiclass label to a binary label based on l1 and l2
%Selects from X and Y, the features and labels the features whos true
%labell is l1 or l2 and returns it in X_out and Y_out

    indexL1 = find(Y == l1);
    indexL2 = find(Y == l2);
    tempL1 = X(indexL1,:);
    X_out = X(indexL2, :);
    X_out = vertcat(tempL1, X_out);
    if l2 > l1
        l2 = 2;
        l1 = 1;
    else    
        l1 = 1;
        l2 = 2;
    end
    tempL1 = zeros(size(indexL1, 1), 1) + l1;
    Y_out =  zeros(size(indexL2, 1), 1) + l2;
    Y_out = vertcat(tempL1, Y_out);
end

