clc; clear;


%% Plot ICO single motor

%% Plot ICO right and left motor

%% Plot ICO angle
% Load files
ico_ang = load('../docker/ros-fence-inspection/assets/ico_logs/ico_ang.txt');
ico_ang_col = load('../docker/ros-fence-inspection/assets/ico_logs/ico_ang_col.txt');

figure(1)
time = ico_ang(:,2);
input = ico_ang(:,3);
weight = ico_ang(:,4);
output = ico_ang(:,5);

subplot(3,1,1);
plot(time, input);
title('Time vs. Input')
xlabel('Time [s]')
ylabel('Angle normalized')

subplot(3,1,2);
plot(time, weight);
title('Time vs. Weight')
xlabel('Time [s]')
ylabel('Weight')

subplot(3,1,3);
plot(time, output);
title('Time vs. Output')
xlabel('Time [s]')
ylabel('Motor decrease normalized')


figure(2)
plot(ico_ang_col(:,1), ico_ang_col(:,2))
title('ICO Angle reflective signal')
xlabel('Time [s]')
ylabel('Reflex signal')