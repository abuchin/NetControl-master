close all; clear all

rng(1) % seed random number
global n frprime fvprime x0 mNoise rNoise nScale

%% Network topology
n = 100; % number of neurons
wN = 1; % network coupling strength
% graphType = 4; % sample network topologies
               % 1: complete graph, 2: path graph, 3: random graph,
               % 4: distance-based graph

%% Sensors and Actuators

load('Actuators.mat'); % load computed 
%R = randperm(n,50); % actuator neurons

all_neurons=1:1:n;
%all_neurons(R)=[];
S = randsample(all_neurons,20); % sensor neurons, randomly in the body

wS = 1; % sensor signal strength
wR = 1; % actuator signal strength

%% Rate model
tauR = ones(n,1)*0.1; % neuron time scales
betaR = 20; TR = 0.5; % sigmoidal parameters
% Potential Function
% fsigR = @(y)(1./(1+exp(-betaR.*(y-TR)))); % sigmoidal function
%  fsigR = @(y)(y>1); % threshold linear
  fsigR = @(y)(y); % linear function

%% Volume model
tauV = 100; % volume time scale
vNom = 1; % Nominal Volume
tlag = 0.1; % Time lag of actuation
betaV = 1000; TV = 5; % sigmoidal parameters
% Potential Function
% fsigV = @(y)(1./(1+exp(-betaV*(y-TV)))); % sigmoidal function
fsigV = @(y)(tauV*(y>5)); % step function

%% Additive Noise
nScale = 10; % discrete noise rate (nScale/sec)
nMagMeas = 0; %0.01; % measurement noise
nMagRate = 0; %0.01; % rate noise 

%% Simulation detail
tfinal = 800; % Run time of simulation
r0 = rand(n,1); % Initial rates
v0 = rand(1); % Initial volume
x0 = [r0; v0];
options = odeset('RelTol',1e-4,'AbsTol',(1e-4)*ones(length(x0),1));

%% Network Type

load('Adjacency_20X5.mat')
Adj=A_tube;

G = graph(Adj~=0); % encode graph
J = wN*Adj/n; % coupling matrix
b = zeros(n,1); b(S) = wS; % input vector of 'sensor' neurons
c = zeros(n,1); c(R) = wR; % output vector of 'sensor' neurons
mNoise = nMagMeas*randn(n,nScale*(tfinal+1));
rNoise = nMagRate*randn(n,nScale*(tfinal+1));

frprime = @(r,v) (inv(diag(tauR))*(-r + fsigR(J*r+diag(b)*v)));
fvprime = @(v,r) (1/tauV*(-v + vNom - fsigV(c'*r)));

tspan = [0 tfinal];
sol=dde23(@funcD_RV_lag,tlag,@func_hist_lag, tspan,options);
t = sol.x; x = sol.y';

%% Plot network, sensor and actuators
figure
subplot(3,1,1)
hold on
j = jet;
colormap('jet')
p = plot(G);
p.NodeCData = b./wS;
axis off
title('Topology and Sensor Locations')
pMarker = plot(NaN,NaN,'o','MarkerFaceColor',j(end,:));
if length(S)~=n; % Display legend if not all neurons are 'sensors'
    legend(pMarker,'Sensors','Location','Best') 
end

subplot(3,1,2)
hold on
colormap('jet')
p = plot(G);
p.NodeCData = c./wR;
axis off
title('Topology and Actuator Locations')
pMarker = plot(NaN,NaN,'o','MarkerFaceColor',j(end,:));
if length(R)~=n; % Display legend if not all neurons are 'actuators'
    legend(pMarker,'Actuators','Location','Best') 
end

subplot(3,1,3);
set(gca,'Fontsize',20)
p = plot(G);
p.NodeCData = x(end,1:n);
axis off
title('Rate Distribution at t_{final}')
colorbar('EastOutside')
subplot(3,1,1);
h = colorbar('EastOutside');
set(h,'visible','off')
subplot(3,1,2);
h = colorbar('EastOutside');
set(h,'visible','off')

%% Plot transfer functions
figure
subplot(2,1,1)
rSample = linspace(0,2,100);
plot(rSample,fsigR(rSample));
xlabel('Cumulative local network and Volume sensing')
ylabel('\Phi_r')
subplot(2,1,2)
vSample = linspace(0,10,100);
plot(vSample,fsigV(vSample));
set(gca,'Fontsize',20)
xlabel('Cumulative rate')
ylabel('\Phi_v')

%% Plot time series data
figure
subplot(2,1,1)
plot(t,x(:,1:n))
xlabel('time')
ylabel('Firing rate')
subplot(2,1,2)
plot(t,x(:,n+1))
set(gca,'Fontsize',20)
xlabel('time')
ylabel('Volume')

