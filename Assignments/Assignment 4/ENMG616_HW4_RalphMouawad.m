% Ralph Mouawad - ID 202204667
% ENMG 616 - Assignment 4 - Dr. Maher Nouiehed

% 1- Solving Quadratic Constrained Optimization 
%% question 1 & 2-  
n = 100;
B = randn(n,n);
Q = 0.5*(B*B');
Q = Q +(min(eig(Q))+10)*eye(n);
q = 10*randn(n,1);
L = max(eig(Q));
x0 = ones(n,1);
%% question 3 & plot
x1 = x0;
step1 = 1/L;
grad = Q*x1 + q;
errors_1 = zeros(2000, 1);

for i = 1:2000  
    x1 = x1 - step1 * grad; % compute new iteration
    x1(x1 < 1) = 1; % projection to boundary when x < 1
    x1(x1 > 3) = 3; % projection to boundary when x > 3 
    y1 = x1 - grad;
    y1(y1 > 3) = 3; % I am projecting the second term of the error onto the feasible set before computing the total error
    y1(y1 < 1) = 1; 
    errors_1(i) = norm(x1 - y1)^2; % error at iteration i 
    grad = Q * x1 + q; % compute new gradient
end
errors_1;
%Plot errors for Question 3
figure;
plot(1:2000, errors_1, 'LineWidth', 1.5);
title('Errors for Question 3');
xlabel('Iteration');
ylabel('Error');
0.5*x1'*Q*x1 + q'*x1;
%% question 4 & plot 
x2 = x0;
grad = Q*x2 + q;
errors_2 = zeros(2000,1);
for j = 1:2000
    step2 = 5 * log(j) / j * L; % new step size
    x2 = x2 - step2 * grad; % new iteration
    x2(x2 < 1) = 1; % projection
    x2(x2 > 3) = 3; % projection
    y2 = x2 - grad; % second term of the error  
    y2(y2 > 3) = 3; % projection of second term of error 
    y2(y2 < 1) = 1; % projection of second term of error
    errors_2(j) = norm(x2 - y2)^2; %error at step i
    grad = Q * x2 + q; % compute new gradient
end
errors_2;
% Plot errors for Question 4
figure;
plot(1:2000, errors_2, 'LineWidth', 1.5);
title('Errors for Question 4');
xlabel('Iteration');
ylabel('Error');

%% Question 5 - Frank Wolfe Method & plot
x3 = ones(n, 1);
step = 1/L;
errors_3 = zeros(2000,1);
for i = 1:2000
    grad = Q * x3 + q;
    d = zeros(n, 1);
    d(grad >= 0) = 1; % the element in 'd' where the corresponding one in the gradient is positive should be 1 
    d(grad < 0) = 3; % same but here when element of grad is negative we take 3
    x3 = x3 + step * (d - x3);
    y3 = x3 - grad;
    y3(y3 < 1) = 1;
    y3(y3 > 3) = 3;
    errors_3(i) = norm(x3 - y3)^2;
end
x3;
errors_3;

% Plot errors for Question 5
figure;
plot(1:2000, errors_3, 'LineWidth', 1.5);
title('Errors for Question 5');
xlabel('Iteration');
ylabel('Error');


%% Part 1.1 - Dual Problem 

dual_function = @(lambda) -((1/2) * (lambda(1) - lambda(2) + q)' * inv(Q) * (lambda(1) - lambda(2) + q) - 3 * lambda(1) + lambda(2));
grad_dual = @(lambda) gradient(dual_function, lambda);
lambda = zeros(1,2); 
L_dual = max(eig(inv(Q)));
for iteration = 1:2000
    % Compute the gradient of the objective function
    grad = grad_dual(lambda);

    % Perform the gradient descent step
    lambda = lambda - 1/L_dual * grad;
    % Project onto the non-negativity constraints
    lambda(lambda < 0) = 0;
end
lambda;
dual_function(lambda);

