function [addP] = BAM_Multi_Binary()
% [Err, addP, P] = BAM_2(H)

load X;

train_x = (0.5*X) + 0.5;
axb = train_x(16,:);

%% Initialize trianng parameters and weights  
e = 4;
b = -e;

I = size(axb,2); 
J = 3; 
K = 3;
L = 5;

%% Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', I)
    struct('type', 'hidden1', 'size', J)
    struct('type', 'hidden2','size', K)
    struct('type', 'output','size', L)
    };

%% Forward Training Training the Network
num = 1000;

E = zeros(num, 40);

P = zeros(num,1);

for iter = 1:1:num
    nn.W1 = unifrnd(b, e, J, I);
    nn.W2 = unifrnd(b, e, K, J);
    nn.W3 = unifrnd(b, e, L, K);
    
    axb = train_x(16,:);
    
    for i = 1:1:40
        ax =  (axb > 0);
        
        % Forward Pass
        oh1 = (nn.W1 * ax');
        ah1 = (oh1 > 0);
        
        oh2 = nn.W2 * ah1;
        ah2 = (oh2 > 0);
        
        oy = nn.W3 * ah2;
        ay = oy > 0;
        
        % Backward Pass
        ohb2 = nn.W3' * ay;
        ahb2 = (ohb2 > 0) ;
        
        ohb1 = nn.W2'*ahb2;
        ahb1 = ohb1 > 0;
                      
        oxb = nn.W1'*ahb1;
        axb = (oxb > 0)';
    
        % Compute Energy
        E1 = ah1'*nn.W1*ax'; 
        E2 = ah2'*nn.W2* ah1;
        E3 = ay'*nn.W3 *ah2;
        
        E(iter,i) = -(E1 + E2 + E3);
    end
    
    P(iter,1) = issorted(E(iter,:) * -1);
end

addP = sum(P);
end

