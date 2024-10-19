% PROBLEM 1
% questions 1-2

data = readmatrix('income.data.csv');
X = data(2:499,2); % remove first row where headers are written, and just take numerical values
Y = data(2:499,3);
X;
Y;

% question 3

n = length(X);
X2 = [X,ones(n,1)];
X2;
theta = X2'*X2 \ X2'*Y;
theta;

% question 4

L = max(eig(X*X'));
L;

% question 5

thetaG = [1,1]';
gradient = (2/n)*(X2'*X2*thetaG - X2'*Y);
a = 1/L;
stop = 10^-3;
while norm(gradient)>=stop
    thetaG = thetaG - a*gradient;
    gradient = (2/n)*(X2'*X2*thetaG - X2'*Y);
end
thetaG;

% question 6

scatter(X, Y); 
hold on;
plot(0:0.5:10,theta(2) + theta(1)*(0:0.5:10))

% Part 1.1
% question 1&2 

maxValue = max(X);
maxValue;
Xnew = ones(n,1);
Ynew = ones(n,1);
for i = 1:n
    Xnew(i) = X(i)/maxValue;
    Ynew(i) = Y(i)/maxValue;
end
Xnew;
Ynew;

%question 3
Lnew = max(eig(Xnew*Xnew'));
Lnew;

%question 4

X2new = [Xnew, ones(n,1)];
thetaGnew = [1,1]';
gradientNew = (2/n)*(X2new'*X2new*thetaGnew - X2new'*Ynew);
aNew = 1/Lnew;
stop = 10^-3;
while norm(gradientNew)>=stop
    thetaGnew = thetaGnew - aNew*gradientNew;
    gradientNew = (2/n)*(X2new'*X2new*thetaGnew - X2new'*Ynew);
end
thetaGnew;


        

