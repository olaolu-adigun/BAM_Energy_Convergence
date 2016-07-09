function [addP] = BAM_Multi_Bipolar_4(H)

% [Err, addP, P] = BAM_2(H)

load X;

train_x = X;
axb = train_x(16,:);

%% Initialize trianng parameters and weights  
e = 4;
b = -e;

I = size(axb,2); 
J = H; 
K = H;
L = H;
M = H;
O = 5;
%% Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', I)
    struct('type', 'hidden1', 'size', J)
    struct('type', 'hidden2','size', K)
    struct('type', 'hidden3','size', L)
    struct('type', 'hidden4','size', M)
     struct('type', 'hidden4','size', O)
    
    };

%% Forward Training Training the Network
num = 10000;

E = zeros(num, 40);

P = zeros(num,1);

for iter = 1:1:num
    
    nn.W1 = unifrnd(b, e, J, I);
    nn.W2 = unifrnd(b, e, K, J);
    nn.W3 = unifrnd(b, e, L, K);
    nn.W4 = unifrnd(b, e, M, L);
    nn.W5 = unifrnd(b, e, O, M);
    
    axb = train_x(16,:);
    
    for i = 1:1:40
        ax =  (2*(axb > 0)) - 1 ;
        
        % Forward Pass
        oh1 = (nn.W1 * ax');
        ah1 = (2*(oh1 > 0)) - 1;
        
        oh2 = nn.W2 * ah1;
        ah2 = (2*(oh2 > 0)) - 1;
        
        oh3 = nn.W3 * ah2;
        ah3 = (2*(oh3 > 0)) - 1;
        
        oh4= nn.W4 * ah3;
        ah4 = (2*(oh4 > 0)) - 1;
        
        oy = nn.W5 * ah4;
        ay = (2*(oy > 0)) - 1;
        
        % Backward Pass
        ohb4 = nn.W5' * ay;
        ahb4 = (2*(ohb4 > 0)) - 1;
        
        ohb3 = nn.W4' * ahb4;
        ahb3 = (2*(ohb3 > 0)) - 1;
        
        ohb2 = nn.W3'*ahb3;
        ahb2 = (2*(ohb2 > 0)) - 1;
        
        ohb1 = nn.W2'*ahb2;
        ahb1 = (2*(ohb1 > 0)) - 1;
                      
        oxb = nn.W1'*ahb1;
        axb = (2*(oxb > 0))' - 1;
    
        % Compute Energy
        E1 = ah1'*nn.W1*ax'; 
        E2 = ah2'*nn.W2* ah1;
        E3 = ah3'*nn.W3 *ah2;
        E4 = ah4'*nn.W4 *ah3;
        E5 = ay'* nn.W5* ah4;
        
        E(iter,i) = -(E1 + E2 + E3 + E4 + E5);
    end
    
    P(iter,1) = issorted(E(iter,:) * -1);
end
addP = sum(P);
end