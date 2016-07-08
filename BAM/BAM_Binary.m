function [addP] = BAM_Binary()
% Binary Vector Space
% [Err,addP, P] = BAM_Binary()
% clear;
load X;

train_x = 0.5*(X+1);
axb = train_x(16,:);

%% Initialize trianng parameters and weights  
e = 4;
b = -e;

I = size(axb,2); 
J = 3; 
K = 5;

%% Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', I)
    struct('type', 'input', 'size', J)
    struct('type', 'output','size', K)
    };

%% Initialize the weights 

%% Forward Training Training the Network
num = 1000;
Err = zeros(num, 40);
P = zeros(num,1);

for iter = 1:1:num
    nn.W1 = unifrnd(b, e, J, I);
    nn.W2 = unifrnd(b, e, K, J);
    
    axb = train_x(16,:);
    
    for i = 1:1:40
        ax =  (axb > 0);
        % Forward Pass
        oh = (nn.W1 * ax');
        ah = (oh > 0);
        
        oy = nn.W2 * ah;
        ay = (oy > 0);
        
        % Backward Pass
        ohb = nn.W2'*ay;
        ahb = (ohb > 0) ;
        oxb = nn.W1'*ahb;
        axb = (oxb > 0)';
    
        % Compute Energy
        E1 = ah'*nn.W1*ax'; 
        E2 = ay'*nn.W2* ah;
        Err(iter,i) = -(E1+E2);
    end
     P(iter,1) = issorted(Err(iter,:) * -1);
end
addP = sum(P);

end

