function [Err,addP, P] = BAM_Bip()

% Bipolar Vector Space
load X;

train_x = X;
axb = train_x(2,:);

%% Initialize trianng parameters and weights
e = 4;
b = -e;

I = size(axb,2); 
J = 4; 

%% Define the Neural Network
nn.layers = {
    struct('type', 'input', 'size', I)
    struct('type', 'input', 'size', J)
    };

%% Initialize the weights 

%% Forward Training Training the Network
num = 1000;
Err = zeros(num, 40);
P = zeros(num,1);

for iter = 1:1:num
    
    nn.W1 = unifrnd(b, e, J, I);
    axb = train_x(2,:);
    
    for i = 1:1:40
        ax =  (2*(axb > 0)) - 1;
        % Forward Pass
        oy = (nn.W1 * ax');
        ay = (2*(oy > 0)) - 1;
        
        % Backward Pass
        oxb = nn.W1'*ay;
        axb = (2*(oxb > 0))' - 1 ;
    
        % Compute Energy
        E1 = ay'*nn.W1*ax'; 
        Err(iter,i) = -(E1);
    end
     P(iter,1) = issorted(Err(iter,:) * -1);
end
addP = sum(P);

end

