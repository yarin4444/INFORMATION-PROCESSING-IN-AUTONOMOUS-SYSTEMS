%{
#INFORMATION PROCESSING IN AUTONOMOUS SYSTEMS#
#Written by: Yarin Hausler#
#Last update: 31.07.2023#
%}
clc
clear all
close all
disp("Welcome to the 'Localization Project'");
disp("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

%% Parameters
%N = 1000;              % Rocomended number of particles
N = input('Number of particles: ');    % User choise: Number of particles
sim_time = 100;         % Simulation time (in seconds)
dt = 1;                 % Time step (in seconds)
noise_std_prox = 0.4;   % Standard deviation of proximity sensor noise
noise_std_odom = 0.1;   % Standard deviation of odometer noise
StartX = 0;             % Initial x coordinate
StartY = 0;             % Initial y coordinate
%EndX = 6;              % Recomended final x coordinate
%EndY = 9;              % Recomended final y coordinate
EndX = input('Final x coordinate(0-10): '); % User choise: Final x coordinate
EndY = input('Final y coordinate(0-10): '); % User choise: Final y coordinate

%% Generate the real-world with beacons and the agent's trajectory
%Nb = 10;                                     % Number of beacons
Nb = input('Number of beacons: ');            % User choise: Number of beacons
beacon_coords = rand(Nb, 2) .* [EndX, EndY];  % Randomly generate beacon coordinates

% Generate agent's trajectory as a straight line from (StartX,StartY) to (EndX,EndY)
agent_trajectory = [linspace(StartX, EndX, sim_time / dt + 1)', linspace(StartY, EndY, sim_time / dt + 1)'];

% Particle Filter Initialization
particles = rand(N, 2) .* [EndX, EndY];       % Initialize particles uniformly within the real-world
weights = ones(N, 1) / N;                     % Initialize particle weights uniformly
estimated_trajectory = zeros(size(agent_trajectory));
adaptive_filter = input('1 to "Turn ON" ESS, 0 to "Turn OFF" ESS: ');

%% Particle Filter Algorithm and Animation
figure;

for t = 1:size(agent_trajectory, 1)
    % Prediction Step
    particles = particles + randn(N, 2) .* noise_std_odom;        % Add odometer noise to particles
    
    % Update Step (Measurement Update)
    z_k = agent_trajectory(t, :) + randn(1, 2) .* noise_std_prox; % Add proximity sensor noise to agent's true position
    
    % Calculate likelihoods based on proximity sensor readings
    likelihoods = ones(N, 1);         % Initialize likelihoods to 1
    for i = 1:Nb
        beacon_distance = vecnorm(particles - beacon_coords(i, :), 2, 2);
        likelihoods = likelihoods .* normpdf(beacon_distance, norm(z_k - beacon_coords(i, :)), noise_std_prox);
    end
    
    weights = weights .* likelihoods; % Update particle weights based on likelihoods
    weights = weights + eps;          % Adjust particle weights to avoid numerical issues
    weights = weights / sum(weights); % Normalize weights

    if adaptive_filter == 1
        %#Reduce the number of particles#
        % Calculate Effective Sample Size (ESS)
        ESS = 1 / sum(weights.^2);
        
        % Adaptive Particle Filtering: Adjust the number of particles
        if ESS < N / 2  % Adjust this threshold as needed
            % Resample to increase the number of particles
            resampled_indices = randsample(1:N, N, true, weights);
            particles = particles(resampled_indices, :);
            weights = ones(N, 1) / N;
        end

    elseif adaptive_filter == 0
        %#Regular process#
        % Resampling Step (Systematic Resampling)
        cumulative_weights = cumsum(weights);
        u = (rand + (0:N-1))' / N;
        resampled_indices = zeros(N, 1);
        i = 1;
        j = 1;
        while i <= N
            if u(i) < cumulative_weights(j)
                resampled_indices(i) = j;
                i = i + 1;
            else
                j = j + 1;
            end
        end
        particles = particles(resampled_indices, :);
        weights = ones(N, 1) / N; % Reset weights to be uniform after resampling
        
    else
        disp('Unknown option. Please RESET');
    end

    % Estimated Agent's Position
    estimated_position = sum(particles .* weights, 1); % Weighted average of particle positions
    estimated_trajectory(t, :) = estimated_position;
    
    % Plot Results
    clf;
    subplot(2, 1, 1);
    plot(agent_trajectory(:, 1), agent_trajectory(:, 2), 'b-', 'LineWidth', 1.5);
    hold on;
    scatter(beacon_coords(:, 1), beacon_coords(:, 2), 'r', 'filled');
    scatter(particles(:, 1), particles(:, 2), 'k', '.');
    plot(estimated_trajectory(:, 1), estimated_trajectory(:, 2), 'g-', 'LineWidth', 1.5);
    xlabel('X');
    ylabel('Y');
    title('Agent Trajectory and Estimated Position');
    legend('Agent True Trajectory', 'Beacons', 'Particles', 'Estimated Trajectory');
    grid on;

    drawnow;
    pause(0.1); % Adjust the pause duration to control the animation speed
end

%% Calculate Mean Squared Estimation Error over time
mse = mean((agent_trajectory - estimated_trajectory).^2, 2);
total_mse_error = mean(mse, "all");

%% Plot Mean Squared Estimation Error
hold on;
subplot(2, 1, 2);
plot(0:dt:sim_time, mse, 'm-', 'LineWidth', 1.5);
xlabel('Time (seconds)');
ylabel('Mean Squared Estimation Error');
title('Mean Squared Estimation Error over Time');
grid on;
