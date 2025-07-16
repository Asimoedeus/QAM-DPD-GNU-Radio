clear,clc
close all;

%% z(n)
for ii = 1
    Nfft = 8192;
    Nsc = 8192;
    Q16s13 = 2^13;
    rng('shuffle');

    % -- QPSK -- %
    qpsk = 2;
    tb = round(rand(Nsc*qpsk,1));
    x = zeros(Nsc,1);
    for i = 1:Nsc
        if(tb(qpsk*i-1)==0 && tb(qpsk*i)==0)
            x(i) = 1+1i;
        elseif(tb(qpsk*i-1)==0 && tb(qpsk*i)==1)
            x(i) = -1+1i;
        elseif(tb(qpsk*i-1)==1 && tb(qpsk*i)==1)
            x(i) = -1-1i;
        elseif(tb(qpsk*i-1)==1 && tb(qpsk*i)==0)
            x(i) = 1-1i;
        else
            disp('tb bits error');
        end
    end
    x = round(x/sqrt(2)*Q16s13);


    xf = zeros(Nfft,1);
    IDX = (Nfft-Nsc)/2+1:(Nfft-Nsc)/2+Nsc;
    xf(IDX) = x;

    x_t = ifft(xf,Nfft);
end
z = x_t;
z = z/max(abs(z))*0.8;

% z = linspace(0,1,Nfft).';
snr = 80;
z = awgn(z,snr,'measured');
%% PA                                                             Digital Predistortion of Power Amplifiers for Wireless Applications,Example 3.4,page 41.
f = linspace(-1/2,1/2,Nfft).';
H = (1+0.5*exp(-1i*4*pi*f))./(1-0.2*exp(-1i*2*pi*f));
G = (1-0.1*exp(-1i*4*pi*f))./(1-0.4*exp(-1i*2*pi*f));

b = [1.0108+1i*0.0858;0.0879-1i*0.1583;-1.0992-1i*0.8891];



figure;
hold on;
grid on;
box on;
title('H(z)')
plot(f,20*log10(abs(H))-max(20*log10(abs(H))))
xlabel('Normilized frequency');
ylabel('Power /dB');
hold off;


figure;
hold on;
grid on;
box on;
title('G(z)')
plot(f,20*log10(abs(G))-max(20*log10(abs(G))))
xlabel('Normilized frequency');
ylabel('Power /dB');
hold off;


% figure;
% hold on;
% grid on;
% box on;
% title('AM/AM')
% plot(z,abs(memlessNonlinFunc( z,b,length(b)*2-1)))
% xlabel('Input amplitude');
% ylabel('Output amplitude');
% hold off;
%
% figure;
% hold on;
% grid on;
% box on;
% title('AM/PM')
% plot(z,angle(memlessNonlinFunc( z,b,length(b)*2-1)))
% xlabel('Input amplitude');
% ylabel('Phase deviation /rad');
% hold off;




%% y(n)
y = ifft(fft(memlessNonlinFunc( z,b,length(b)*2-1)).*H.*G);

Gain = mean(abs(y))/mean(abs(z));

%% Predistorter
K = 5;
Q = 9;
U = zeros(Nfft,(K+1)/2*(Q+1));
for q = 0:Q
    for k = 1:2:K
        idx = (k+1)/2 + q*((K+1)/2);
        for n = 1:Nfft
            U(n,idx) = y(mod(n-q-1,Nfft)+1)/Gain*abs(y(mod(n-q-1,Nfft)+1)/Gain)^(k-1);
        end
    end
end

a = inv(U'*U)*U'*z;


%%  DPD输出
for ii = 1
    Nfft = 8192;
    Nsc = 2048;
    Q16s13 = 2^13;
    rng('shuffle');

    % -- QPSK -- %
    qpsk = 2;
    tb = round(rand(Nsc*qpsk,1));
    x = zeros(Nsc,1);
    for i = 1:Nsc
        if(tb(qpsk*i-1)==0 && tb(qpsk*i)==0)
            x(i) = 1+1i;
        elseif(tb(qpsk*i-1)==0 && tb(qpsk*i)==1)
            x(i) = -1+1i;
        elseif(tb(qpsk*i-1)==1 && tb(qpsk*i)==1)
            x(i) = -1-1i;
        elseif(tb(qpsk*i-1)==1 && tb(qpsk*i)==0)
            x(i) = 1-1i;
        else
            disp('tb bits error');
        end
    end
    x = round(x/sqrt(2)*Q16s13);


    xf = zeros(Nfft,1);
    IDX = (Nfft-Nsc)/2+1:(Nfft-Nsc)/2+Nsc;
    xf(IDX) = x;

    x_t = ifft(xf,Nfft);
end

x = x_t;
x = x/max(abs(x))*0.8;
x = awgn(x,80,'measured');

% 理想功放输出
out_ideal = x*Gain;

% 记忆非线性功放输出
out_PA = ifft(fft(memlessNonlinFunc( x,b,length(b)*2-1)).*H.*G);

% 经过DPD和记忆非线性功放输出
tem = zeros(Nfft,1);
for n = 1:Nfft
    for q = 0:Q
        for k = 1:2:K
            idx = (k+1)/2 + q*((K+1)/2);
            tem(n) = tem(n) + a(idx)*(x(mod(n-q-1,Nfft)+1))*abs(x(mod(n-q-1,Nfft)+1))^(k-1);
        end
    end
end
out_DPD_PA = ifft(fft(memlessNonlinFunc( tem,b,length(b)*2-1)).*H.*G);



% 转换到频域
tem_ideal = fft(out_ideal);
tem_PA = fft(out_PA);
tem_DPD_PA = fft(out_DPD_PA);

figure;
title('PSD');
box on;
grid on;
hold on;
plot(f,20*log10(abs(tem_PA)),'r')
plot(f,20*log10(abs(tem_DPD_PA)),'c')
plot(f,20*log10(abs(tem_ideal)),'b')
xlabel('Normilized frequency');
ylabel('Power /dB');
legend('PA','DPD','IDEAL PA')
hold off;


figure;
title('Constelation');
box on;
grid on;
hold on;
scatter(real(tem_PA(IDX)),imag(tem_PA(IDX)),'r')
scatter(real(tem_DPD_PA(IDX)),imag(tem_DPD_PA(IDX)),'c')
scatter(real(tem_ideal(IDX)),imag(tem_ideal(IDX)),'b','Marker','+','LineWidth',10)
xlabel('Real part');
ylabel('Imag part');
legend('PA','DPD','IDEAL PA')
hold off;

