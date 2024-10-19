% PROBLEM 2
% question 1

n = 50;
B = randn(n,n);
Q = B*B';
MinEig = min(eig(Q));
Q = Q + (MinEig + rand(1))*eye(n);
q = 10*randn(n,1);

% question 2
X_FOSS = -Q\q;
X_FOSS;

%question 3
%part a Exact Line Search

g = @(x) (Q*x + q); % gradient fuction
f = @(x) (1/2)*x'*Q*x + q'*x; % obj function
a = 0.1;
x0 = zeros(n,1);
x = x0;
while norm(g(x)) >= 0.001 % stopping criteria
    h = @(a) (f(x - a*g(x)));
    a = fminsearch(h,0);
    x = x - a*g(x);
end
x_opt1 = x;
x_opt1;
for i = 1:1000
    h = @(a) (f(x - a*g(x)));
    a = fminsearch(h,0);
    x = x - a*g(x);
end
x1000 = x;
e1 = log((norm(x1000 - x_opt1))/norm(x0 - x_opt1));
e1

% % part b Armijo Rule / Back Tracking
g = @(x) (Q*x + q); % gradient fuction
f = @(x) (1/2)*x'*Q*x + q'*x; % obj function
a2 = 0.1;
x0 = zeros(n,1);
x = x0;
sigma = 0.2; 
beta = 0.5;
while f(x) - f(x - a2*beta*g(x)) <= a2*sigma*beta*g(x)'*g(x)
    a2 = beta*a2;
    x = x - a2*g(x);
end
x_opt2 = x;
x_opt2;
for i = 1:1000
    a2 = beta*a2;
    x = x - a2*g(x);
end
x1000 = x;
e2 = log((norm(x1000 - x_opt2))/norm(x0 - x_opt2));
e2

%  part c Diminishing Step Size
x0 = zeros(n,1);
x = x0;
g = @(x) (Q*x+q);
f = @(x) (0.5*x'*Q*x + q'*x);
r = 1;
while norm(g(x)) >= 0.001
    a3 = 0.1/sqrt(r); %when I divided by r, the code was running infinitely.
    x = x - a3*g(x);
    r = r+1;
end
x_opt3 = x;
x_opt3;
for i = 1:1000
    a3 = 0.1/sqrt(r); 
    x = x - a3*g(x);
    r = r+1;
end
x1000 = x;
e3 = log((norm(x1000 - x_opt3))/norm(x0 - x_opt3));
e3

% % part d constant step-size
g = @(x) (Q*x + q);
x0 = zeros(n,1);
x = x0;
L = max(eig(Q));
while norm(g(x)) >= 0.001
    x = x -(1/L)*g(x);
end
x_opt4 = x;
x_opt4;
for i = 1:1000
    x = x - (1/L)*g(x);
end
x1000 = x;
e4 = log((norm(x1000 - x_opt4))/norm(x0 - x_opt4));
e4

% % question 4 Newton's Method

g = @(x) (Q*x + q); %gradient. Hessian is inv(Q)
x0 = zeros(n,1);
x = x0;
L = max(eig(Q));
while norm(g(x)) >= 0.001
    d = -inv(Q)*g(x);
    a5 = 1/L;
    x = x + a5*d;
end
x_opt5 = x;
x_opt5;
for i = 1:1000
    d = -inv(Q)*g(x);
    a5 = 1/L;
    x = x + a5*d;
end 
x1000 = x;
e5 = log((norm(x1000 - x_opt5))/norm(x0 - x_opt5));
e5

%  question 5 Accelerated Gradient Descent Method
g=@(x)(Q*x+q);
a=[0 0];
X=zeros(n,2);
L=max(eig(Q));
gradx=1;

while norm(gradx)>0.0001
    %updating the point
    a(2)=0.5*(1+sqrt(4*(a(1))^2)+1);
    y=X(1:50,2)+((a(1)-1)/a(2))*(X(1:50,2)-X(1:50,1));
    X(1:50,2)=y-(1/L)*g(y);
    %updating the list of X and a
    a(1)=a(2);
    X(1:50,1)=X(1:50,2);
    gradx=g(X(1:n,1));
end
x_opt6 = X(1:50,1);
x_opt6
a=[0 0]; % repeat for 1000 iterations
X=zeros(n,2);
L=max(eig(Q));
gradx=1;
for i = 2:1000
    a(2)=0.5*(1+sqrt(4*(a(1))^2)+1);
    y=X(1:50,2)+((a(1)-1)/a(2))*(X(1:50,2)-X(1:50,1));
    X(1:50,2)=y-(1/L)*g(y);
    a(1)=a(2);
    X(1:50,1)=X(1:50,2);
    gradx=g(X(1:n,1));
end
x1000 = X(1:50,1);
e6 = log((norm(x1000 - x_opt6)/norm(zeros(n,2) - x_opt6)));
e6

kappa = max(eig(Q))/min(eig(Q));
kappa
