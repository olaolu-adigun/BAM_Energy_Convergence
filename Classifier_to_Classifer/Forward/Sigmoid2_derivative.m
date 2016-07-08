function derive = Sigmoid2_derivative(x)
%
d = Sigmoid(x) + 1;
derive = 2* 0.5 * d.*(2-d);
end

