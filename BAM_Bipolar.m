function [Err,addP, P] = BAM_Bipolar()

% Bipolar Vector Space
load X;

train_x = X;
axb = train_x(2,:);

%% Initialize trianng parameters and weights  
e = 4;
b = -e;

I = size(axb,2); 
J = 100; 
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
    
    axb = train_x(2,:);
    
    for i = 1:1:40
        ax =  (2*(axb > 0)) - 1;
        % Forward Pass
        oh = (nn.W1 * ax');
        ah = (2*(oh > 0))-1;
        oy = nn.W2 * ah;
        ay = (2*(oy > 0))-1;
        
        % Backward Pass
        ohb = nn.W2'*ay;
        ahb = (2*(ohb > 0))-1 ;
        oxb = nn.W1'*ahb;
        axb = ((2*(oxb > 0))-1)';
    
        % Compute Energy
        E1 = ah'*nn.W1*ax'; 
        E2 = ay'*nn.W2* ah;
        Err(iter,i) = -(E1+E2);
    end
     P(iter,1) = issorted(Err(iter,:) * -1);
end
addP = sum(P);

end

