function [opts, net] = Feedforward(opts,net)

%% Feed Forward Propagation. 

% Input Layer
opts.ax = (opts.ox);

% Hidden Layer
opts.oh = (net.W1*opts.ax')+  net.Bias_W1';
opts.ah = Sigmoid(opts.oh);

% Output Layer
opts.ok = net.W2*opts.ah + net.Bias_W2';
opts.ak = Softmax(opts.ok);
end

