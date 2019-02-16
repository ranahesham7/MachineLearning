clc
clear all
close all
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',250);
T = read(ds);
m= length(T{:,1});
s=ceil(m*0.8);
s1=ceil(m*0.1);
x=T{1:s,1:13};
X=[ones(s,1) x./5 x.^3 x.^4];
%Y=T{1:s,14}/mean(T{1:s,14});
Y=T{1:s,14};
y=T{1:s,14};
Alpha=0.01;
lambda=100;

%compute cost and gradient
 n=length(X(1,:)); 
 for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
 end
 theta=zeros(n,1);
 h=1./(1+exp(-X*theta));  %sigmoid function
 k=1;
 J(k)=-(1/s)*sum(Y.*log(h)+(1-Y).*log(1-h))+(lambda/(2*s))*sum((theta).^2);  %cost function
 grad=zeros(size(theta,1),1);     %gradient vector
 for i=1:size(grad)
     grad(i)=(1/s)*sum((h-Y)'*X(:,i));
 end
R=1;
while R==1
Alpha=Alpha*1;
theta=theta-(Alpha/s)*X'*(h-Y);
h=1./(1+exp(-X*theta));  %sigmoid function
k=k+1
J(k)=(-1/s)*sum(Y.*log(h)+(1-Y).*log(1-h))+(lambda/(2*s))*sum((theta).^2);
if J(k-1)-J(k) <0 
    break
end 
q=(J(k-1)-J(k))./J(k-1);
if q <.00001
    R=0;
end
end 

 

x1=T{s+1:s+s1,1:13};
X1=[ones(s1,1) x1./5 x1.^3 x1.^4];
%Y1=T{s+1:s+s1,14}/mean(T{s+1:s+s1,14});
y1=T{s+1:s+s1,14};
Y1=y1;
n1=length(X1(1,:)); 
 for w1=2:n1
    if max(abs(X1(:,w1)))~=0
    X1(:,w1)=(X1(:,w1)-mean((X1(:,w1))))./std(X1(:,w1));
    end
 end
 theta1=theta;
 h1=1./(1+exp(-X1*theta1));  %sigmoid function
 k1=1;
 J1(k1)=-(1/s1)*sum(Y1.*log(h1)+(1-Y1).*log(1-h1))+(lambda/(2*s1))*sum((theta1).^2);  %cost function


s2=s+s1+1;
p=m-(s+s1);
x2=T{s2:end,1:13};
X2=[ones(p,1) x2./5 x2.^3 x2.^4];
%Y1=T{s+1:s+s1,14}/mean(T{s+1:s+s1,14});
y2=T{s2:end,14};
Y2=y2;
n2=length(X2(1,:)); 
 for w2=2:n2
    if max(abs(X2(:,w2)))~=0
    X2(:,w2)=(X2(:,w2)-mean((X2(:,w2))))./std(X2(:,w2));
    end
 end
 
 theta2=theta1;
 h2=1./(1+exp(-X2*theta2));  %sigmoid function
 k2=1;
 J2(k2)=-(1/s1)*sum(Y2.*log(h2)+(1-Y2).*log(1-h2))+(lambda/(2*s1))*sum((theta2).^2);  %cost function

 
plot(J)