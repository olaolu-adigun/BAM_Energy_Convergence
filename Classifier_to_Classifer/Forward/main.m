function [] = main()

clear;
%% Get the Data

load X;
% Training set
train_x = X(1:18000,1:6);
train_y = X(1:18000,7:10);

% Test set
test_x = X(18001:20000,1:6);
test_y = X(18001:20000,7:10);

%% Initialize training parameter  

% Define the optimization parameter
opts.numepochs = 180;
opts.batch = 100;
opts.learning = 0.02;


%% Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', size(train_x,2))
    struct('type', 'hidden', 'size', 10)
    struct('type', 'output','size', size(train_y,2))
    };

%% Initialize the weights
%---Weight and bias random range
e = 0.1;
b = -e;
opts.e = e;

% Initialize the weights 
nn.W1 = unifrnd(-e, e,nn.layers{2}.size,size(train_x,2));
nn.W2 = unifrnd(-e, e,nn.layers{3}.size,  nn.layers{2}.size);

% Initiialize Bias Weight
nn.Bias_W1 = unifrnd(-e, e, 1, nn.layers{2}.size);
nn.Bias_W2 = unifrnd(-e, e, 1, nn.layers{3}.size);

%% Forward Training Training the Network

% Feed-Forward Propagation
train_Entropy = zeros(opts.numepochs,1);
kk = randperm(size(train_x,1));
for iter = 1:1:opts.numepochs
     
     m = (opts.batch*(iter - 1)) + 1;
     n = opts.batch * iter;
     ind = kk(m:n);
     
     batch_x = train_x(ind,:);
     batch_y = train_y(ind,:);
     
     del1 = zeros(size(nn.W1));
     del2 = zeros(size(nn.W2));
     
     del1_bias = zeros(nn.layers{2}.size, 1); 
     del2_bias = zeros(nn.layers{3}.size, 1);
     
     opts.learning = opts.learning * (0.9999^(opts.numepochs));
     
    for i = 1:1:opts.batch
        x = batch_x(i, :);
        y = batch_y(i, :);
        
        opts.ox = x; 
        opts.t  = y;
        
        [opts, nn] = Feedforward(opts, nn);
        [opts, Delta_1, Delta_2] = Backpropagation(opts, nn);
        del1 = del1 + Delta_1;
        del2 = del2 + Delta_2;
        
        del1_bias = del1_bias + opts.W1_Bias;
        del2_bias = del2_bias + opts.W2_Bias;
    end
    
    % Update the Weights
    nn.W1 = nn.W1 + ((1/opts.batch) * opts.learning * del1);
    nn.W2 = nn.W2 + ((1/opts.batch) * opts.learning * del2);
    
    nn.Bias_W1 = nn.Bias_W1 + ((1 / opts.batch)*opts.learning * del1_bias');
    nn.Bias_W2 = nn.Bias_W2 + ((1 / opts.batch)*opts.learning * del2_bias');

    %% Compute the Training Error 
    E = zeros(opts.batch, 1);
    for j = 1:1:opts.batch
        opts.ox = batch_x(j, :); 
        opts.t  = batch_y(j, :);
        [opts, nn] = Feedforward(opts,nn);
        E(j) = -(opts.t* log(opts.ak));
    end
    train_Entropy(iter) = mean(E);
    
end
plot(train_Entropy);

 % Compute the Test Error
 test_ent = zeros(size(test_y,1),1);
 pred = zeros(size(test_y,1),1);
 for q = 1:1:size(test_y,1)
     opts.ox = test_x(q, :); 
     opts.t  = test_y(q, :);
     [opts, nn] = Feedforward(opts,nn);
     test_ent(q) = -(opts.t* log(opts.ak));
     pred(q) = find(opts.ak == max(opts.ak));
end
disp(sum(pred == 1));
disp(sum(pred~=1));
end

