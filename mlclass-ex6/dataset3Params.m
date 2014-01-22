function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


value = transpose([0.01 0.03 0.1 0.3 1 3 10 30]);

% Initialise predErr
predErr = zeros(size(value));

% Run for loops for C and for Sigma
for i=1:size(value,1), % Loop for C
    for j=1:size(value,1), % Loop for sigma
        % The syntaxis for svmTrain is --> model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        model = svmTrain(X, y, value(i), @(x1, x2) gaussianKernel(x1, x2, value(j))); 
        predict = svmPredict(model,Xval);
        predErr(i,j) = mean(double(predict ~= yval));
        % Recall index i -> C / row and j -> sigma / column
    end;                
end;

% Extract optimal values and return them

% Find minimum col value 
[jmins, indices_row] = min(predErr);
% Find minimum row value 
[minerr, imins] = min(jmins);
sigma = value(imins);
C = value(indices_row(imins));






% =========================================================================

end
