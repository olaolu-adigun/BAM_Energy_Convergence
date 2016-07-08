function [a] = Softmax(o)
% It finds the softmax of an input vector.
% INPUT  -- input vector o.
% OUTPUT -- activation vector a.

num = exp(o);
den = sum(num);
a = (1/den)*num;
end

