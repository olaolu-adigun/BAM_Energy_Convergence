function [d] = Sigmoid_derivative(x)
% SIGMOID_DERIVATIVE
% Detailed explanation goes here
d = Sigmoid(x).*(1-Sigmoid(x));
end