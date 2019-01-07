%find most ideal values of hyperparameters using a linear search
load A1

lowest = 1;
lowest2 = 1;
lowest3 = 1;
for i = 140:20:200
    for k = 50:20:100
        for j = 0.7:0.1:1
            H = [i  k];
            [~,valErr] = buildNeuralNet(X_train,Y_train, H, j);
            if valErr < lowest3
                if valErr < lowest2
                    if valErr < lowest
                        lowest = valErr;
                        lowestH = H;
                        lowestReg = j;
                    else
                        lowest2 = valErr;
                        lowestH2 = lowestH;
                        lowestReg2 = j;
                    end
                else
                    lowest3 = valErr;
                    lowestH3 = lowestH;
                    lowestReg3 = j;
                end
            end
        end
    end
end
display(lowestH)
display(lowestReg)
display(lowestH2)
display(lowestReg2)
display(lowestH3)
display(lowestReg3)


[net,valErr] = buildNeuralNet(X_train, Y_train, lowestH, lowestReg);
display(valErr);
[err, CONF] = netClassify(X_test, Y_test, net);
display(err);

%10d
%   Using the top program the best values of H and regulaization weight
%   were found to be, the first layer has 180 nodes, while 
%   the second layer has 70 nodes. The val error is less than the
%   unvalidted error for both the validation error and the test error
%       lowestH = 
%           [140  70]
%       lowestReg =
%           0.9000
%       valErr =
%           0.0480
%       err =
%           0.0795