function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

g = X*theta;
h = sigmoid (g);
sigma = sum(-y.*log(h)-(1-y).*log(1-h));
J = ((1/m)*sigma) + (lambda/(2*m))*sum(theta(2:end,:).^2);

sigma = h-y;
grad(1,:) = (X(:,1)'*sigma)/m;
grad(2:end, :) = ((X(:,2:end)'*sigma)/m)+(lambda/m)*theta(2:end,:);

% =============================================================

end
