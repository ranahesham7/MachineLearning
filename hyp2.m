clc
clear all
close all
ds = datastore('house_data_complete.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=.01;
m=length(T{:,1});
s=ceil(m*0.6);
s1=ceil(m*0.2);
U0=T{1:s,2};
U=T{1:s,4:19};
U1=T{1:s,20:21};
U2=U1.^3;
U3=U1.^4;
U4=U1.^5;
X=[ones(s,1) U2 U3 U4];
n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end
Y=T{1:s,3}/mean(T{1:s,3});
Theta=zeros(n,1);
k=1;
E(k)=(1/(2*s))*sum((X*Theta-Y).^2);
R=1; %Flag

%cross validation with 60%
while R==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/s)*X'*(X*Theta-Y);
k=k+1
E(k)=(1/(2*s))*sum((X*Theta-Y).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.001;
    R=0;
end
end
plot(E)

%cross validation with 20%
UU0=T{s+1:s+s1,2};
UU=T{s+1:s+s1,4:19};
UU1=T{s+1:s+s1,20:21};
UU2=UU1.^3;
UU3=UU1.^4;
UU4=UU1.^5;
XX=[ones(s1,1) UU2 UU3 UU4 ];
Theta1=Theta;
YY=T{s+1:s+s1,3}/mean(T{s+1:s+s1,3});
nn=length(XX(1,:));

for w=2:nn
    if max(abs(XX(:,w)))~=0
    XX(:,w)=(XX(:,w)-mean((XX(:,w))))./std(XX(:,w));
    end
end
kk=1;
EE(kk)=(1/(2*s1))*sum((XX*Theta1-YY).^2);



%cross validation for the last 20%
s2=s+s1+1;
p=m-(s+s1);
UU0last=T{s2:end,2};
UUlast=T{s2:end,4:19};
UU1last=T{s2:end,20:21};
UU2last=UU1last.^3;
UU3last=UU1last.^4;
UU4last=UU1last.^5;

XXlast=[ones(p,1) UU2last UU3last UU4last];

Theta2=Theta1;
YYlast=T{s2:end,3}/mean(T{s2:end,3});
nnlast=length(XXlast(1,:));
for w1=2:nnlast
    if max(abs(XXlast(:,w1)))~=0
    XXlast(:,w1)=(XXlast(:,w1)-mean((XXlast(:,w1))))./std(XXlast(:,w1));
    end
end
kk2=1;
EE2(kk2)=(1/(2*s1))*sum((XXlast*Theta2-YYlast).^2);