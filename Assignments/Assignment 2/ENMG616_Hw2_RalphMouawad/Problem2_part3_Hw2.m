% PROBLEM 2.2
% Let's re-write the previous codes
n = 50;
B = randn(n,n);
Q = B*B';
MinEig = min(eig(Q));
Q = Q + (MinEig + 10)*eye(n);
q = 10*randn(n,1);

grad=@(x)(Q*x+q);
X=zeros(n,2);
x0=zeros(n,1);
L=max(eig(Q));
gradx=1;
kapp=max(eig(Q))/min(eig(Q));
ckap=floor(sqrt(kapp));
xop=-inv(Q)*q;
x1=[];
k1=[];
T=0;
a1=[0 0];
while norm(gradx)>0.0001
    if mod(T,ckap)==0
        a1=[0 0];
    end
    %updating the point
    a1(2)=0.5*(1+sqrt(4*(a1(1)^2)+1));
    y=X(1:n,2)+((a1(1)-1)/a1(2))*(X(1:n,2)-X(1:n,1));
    X(1:n,1)=X(1:n,2);
    X(1:n,2)=y-(1/L)*grad(y);
    %updating the list of X and a
    a1(1)=a1(2);
    gradx=grad(X(1:n,1));
    k1=[k1 T];
    e=log(norm(X(1:n,1)-xop)/norm(x0-xop));
    x1=[x1 e];
    T=T+1;
end

X=zeros(n,2);
a2=[0 0];
gradx=1;
x2=[];
k2=[];
T=0;
while norm(gradx)>0.0001
    if mod(T,5*ckap)==0
        a2=[0 0];
    end
    %updating the point
    a2(2)=0.5*(1+sqrt(4*(a2(1)^2)+1));
    y=X(1:n,2)+((a2(1)-1)/a2(2))*(X(1:n,2)-X(1:n,1));
    X(1:n,1)=X(1:n,2);
    X(1:n,2)=y-(1/L)*grad(y);
    %updating the list of X and a
    a2(1)=a2(2);
    gradx=grad(X(1:n,1));
    k2=[k2 T];
    e=log(norm(X(1:n,1)-xop)/norm(x0-xop));
    x2=[x2 e];
    T=T+1;
end

X=zeros(n,2);
a3=[0 0];
gradx=1;
x3=[];
k3=[];
T=0;
while norm(gradx)>0.0001
    if mod(T,20*ckap)==0
        a3=[0 0];
    end
    %updating the point
    a3(2)=0.5*(1+sqrt(4*(a3(1)^2)+1));
    y=X(1:n,2)+((a3(1)-1)/a3(2))*(X(1:n,2)-X(1:n,1));
    X(1:n,1)=X(1:n,2);
    X(1:n,2)=y-(1/L)*grad(y);
    %updating the list of X and a
    a3(1)=a3(2);
    gradx=grad(X(1:n,1));
    k3=[k3 T];
    e=log(norm(X(1:n,1)-xop)/norm(x0-xop));
    x3=[x3 e];
    T=T+1;
end 
X=zeros(n,2);
a4=[0 0];
gradx=1;
x4=[];
k4=[];
T=0;

while norm(gradx)>0.0001
    %updating the point
    a4(2)=0.5*(1+sqrt(4*(a4(1)^2)+1));
    y=X(1:n,2)+((a4(1)-1)/a4(2))*(X(1:n,2)-X(1:n,1));
    X(1:n,1)=X(1:n,2);
    X(1:n,2)=y-(1/L)*grad(y);
    %updating the list of X and a
    a4(1)=a4(2);
    gradx=grad(X(1:n,1));
    k4=[k4 T];
    e=log(norm(X(1:n,1)-xop)/norm(x0-xop));
    x4=[x4 e];
    T=T+1;
end
figure(1)
plot(k1,x1)
figure(2)
plot(k2,x2)
figure(3)
plot(k3,x3)
figure(4)
plot(k4,x4)


