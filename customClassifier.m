function C = customClassifier(X_test)
%classifies using   %neural net with top 3 value for hyper parameters H and regulariion weights 
%as found with the p10d function on the data set
    load A1_full;
  
    
    H = [180, 70];
    regulizationWeights = 0.9;
    [net1, valErr] = buildNeuralNet(X_train_full, Y_train_full, H, regulizationWeights);
    fprintf("done 1");
    save('net1.mat', 'net1');
   H = [140, 70];
   regulizationWeights = 0.7;
   [net2, valErr] = buildNeuralNet(X_train_full, Y_train_full, H, regulizationWeights);
   save('net2.mat', 'net2');

   fprintf("done 2");
   H = [140, 80];
   regulizationWeights = 0.8;
   [net3, valErr] = buildNeuralNet(X_train_full, Y_train_full, H, regulizationWeights);
   save('net3.mat', 'net3');

    fprintf("done3");
    X = transpose(X_test);
    result = net1(X);
    C1 = transpose(vec2ind(result));
    result = net2(X);
    C2 = transpose(vec2ind(result));
    result = net3(X);
    C3 = transpose(vec2ind(result));
    
    
    save('p12_mat.mat', 'C1', 'C2', 'C3');

    %use mode to get most frequent value in each row
    C = mode([C1, C2, C3], 2);
    fromMatrixToCVS(C, "result_p12");

    

    
end