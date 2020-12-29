%% Document for visualizing the different activation functions used
clear; clc;

x = -5:0.01:5; 

%% Tanh 
y = tanh(x);
figure(1)
plot(x,y)
title('Tanh')
xlabel('x')
ylabel('y')
grid on
%% Sigmoid
y = 1./(1+exp(-x));
figure(2)
plot(x,y)
title('Sigmoid')
xlabel('x')
ylabel('y')
grid on