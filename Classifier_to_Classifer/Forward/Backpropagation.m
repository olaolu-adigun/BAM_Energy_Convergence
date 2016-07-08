function [opts, Delta_1,Delta_2] = Backpropagation(opts, net)

%% Backpropagate error from Output layer to Hidden layer.

% Output Layer
del_2 = (opts.t' - opts.ak);
Delta_2 = del_2 * opts.ah';

% Hidden layer
DEL1 = del_2'*net.W2;
DEL2 = opts.ah.*(1-opts.ah);
del_1 = DEL2.*DEL1';
Delta_1 = del_1*opts.ax;

% Output Bias 
opts.W2_Bias = del_2;
opts.W1_Bias = del_1;
end