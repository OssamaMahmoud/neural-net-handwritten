function label = recognize_digit(img, net)

%disp('replace the call to this function with your classifier!')
%img = imcrop(img,[7 7 15 15]);
img = imresize(img, [16, 16]);
imgVector = (img(:));

result = net(imgVector);
label  = transpose(vec2ind(result)) - 1;

