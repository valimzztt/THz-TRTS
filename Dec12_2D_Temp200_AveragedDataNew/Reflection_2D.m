function Reflection_2D()

Selection = 19;     % Start menu selection
[xlimit, ylimit, zlimit] = deal([0 10]);
[Hold_Axe, Fix_Axe, Slide] = deal(0);
[Grid, Title] = deal(1);
[tl, fl, ffl, tpl, range] = deal(1:10);
[Title_Label, Averages, LockIn_1, LockIn_2] = deal([]);
label = 0*Averages*LockIn_1*LockIn_2;
Init;                           % Reads initial parameters

tl = (1:1000)';     % Scan time plotted range 
fl = (8:151)';      % Frequency plotted range
ffl = (12:101)';    % Frequency fitted range
tpl = 1:16;         % 2D Data plotted range
range = 1:151;        % For data export
label = 0*Averages*LockIn_1*LockIn_2;
Init;                           % Reads initial parameters

PumpTimes  = dlmread('PumpTimes.dat')';
PumpDelays = length(PumpTimes);

% Fits = dlmread('Fits.dat');
E1 = dlmread('E1.dat');
E2 = dlmread('E2.dat');

%First column for time
for i = 2:PumpDelays+1
    E1(:,i) = E1(:,i) - mean(E1(:,i));	%Remove Baseline
	E2(:,i) = E2(:,i) - mean(E2(:,i));	%Remove Baseline
end

Time = E1(:,1);
E_Ref  = E1(:, 2:end) - E2(:, 2:end);
E_Pump = E1(:, 2:end) + E2(:, 2:end);
dE    = 2*E2(:, 2:end);
Time_Samp = Time(2) - Time(1);
Freq = 1/Time_Samp/length(E_Ref) * (0:length(E_Ref)-1)';

E_Ref_Amp   =  abs(fft(E_Ref))/length(E_Ref)*2;
E_Ref_Phase = -unwrap(angle(fft(E_Ref)));

E_Pump_Amp   =  abs(fft(E_Pump))/length(E_Pump)*2;
E_Pump_Phase = -unwrap(angle(fft(E_Pump)));

dE_Amp   =  abs(fft(dE))/length(dE)*2;
dE_Phase = -unwrap(angle(fft(dE)));

R_Amp  = E_Pump_Amp ./ E_Ref_Amp;
Phi = E_Pump_Phase - E_Ref_Phase;
Phi = wrapToPi(Phi);

R = R_Amp .* ( cos(Phi) + 1i*sin(Phi) );

d = 7*1e-7;         % Penetration depth (before: 1.3e-6)
w = 2*pi*1e12;
Z0 = 376.73;

e = 1.602176565e-19;
m = 9.10938291e-31;
eps0 = 8.854187817e-12;
epsi = 10;

gam  = 1*1e11;
w0  = 4.2*w;       % w0/2pi = 1.96 THz
wpL = 1.65e8;

tauD = 5.5e-12;
magD = 80;

dim = ones(PumpDelays,1)';

function err = fitfun(c)
    tau  = c(1,:) * init(1);
    wp   = c(2,:) * init(2);
    sigma_drude = bsxfun(@rdivide, eps0*tau.*wp.^2, 1 - 1i*w*Freq(ffl)*tau ); % The photoexcited part can be described by a Drude Lorentz model

    eps_debye = magD ./(1 - 1i*w*Freq(ffl)*tauD);
    eps_lorentz_1 = 1i*bsxfun(@rdivide, wpL.^2,  eps0*( 1i*(w0.^2 - (w*Freq(ffl)).^2) + w*Freq(ffl)*gam) );
    eps_lorentz = bsxfun(@rdivide, wpL.^2,  eps0*(w0.^2 - (w*Freq(ffl)).^2 + 1i* (w*Freq(ffl) * gam)));
    display(eps_lorentz_1(7))
    display(eps_lorentz(7))
    eps = epsi + eps_lorentz + eps_debye;
    eps = eps*dim;
    n = sqrt(eps);

    fun = (1-n-377*d*sigma_drude) ./ (1+n+377*d*sigma_drude) ./ ( (1-n)./(1+n) ); % Tinkham equations
    err = ( abs(R(ffl,:)) - abs(fun) ).^2 + ( angle(R(ffl,:)) - angle(fun) ).^2;
end
eval = 2^12;
mini = 1e-10;
init  = [ 70e-15  8.5e13 ]'; % Initial guess
init = [(1/9.85e+12), 6.8*1e13 ]';
lb = 0.01 * ones(4,length(PumpTimes));
v0 =   1  * ones(4,length(PumpTimes));
ub =  100 * ones(4,length(PumpTimes));
options = optimoptions(@lsqnonlin, 'Algorithm', 'Trust-Region-Reflective', 'Display', 'none',... 'Display', 'final-detailed', ......'Display', 'iter-detailed', ...
            'TolX', mini, 'TolFun', mini, 'DiffMinChange', mini, 'MaxFunEvals', eval, 'MaxIter', eval);
v = lsqnonlin(@(x)fitfun(x), v0, lb, ub, options);

tau  = v(1,:) * init(1);
wp   = v(2,:) * init(2);
display(tau)
display(wp)
sigma_drude = bsxfun(@rdivide, eps0*tau.*wp.^2, 1 - 1i*w*Freq*tau );

nfree = eps0*(0.2*m)*wp.^2/(e^2);
mu = e*tau/(0.2*m)*100^2;   %0.2 effective mass, 100^2 cm^2 units

eps_debye = magD ./ (1 - 1i*w*Freq*tauD);
eps_lorentz = 1i*bsxfun(@rdivide, wpL.^2,  eps0*( 1i*(w0^2 - (w*Freq).^2) + w*Freq*gam) );  %This should be an epsilon, no change in wpL

eps = epsi + eps_lorentz + eps_debye;
eps = eps*dim;

sigma_dark = bsxfun(@times, 1i*eps0*w*Freq, epsi - eps);

n = sqrt(eps);
r0 = (1-n) ./ (1+n);

Re = -(1+r0)./r0 .* ( Z0*d*sigma_drude ) ./ ( 1 + n + Z0*d*sigma_drude ) + 1; %photoonductivity of surface layer

factor = 2^5;
Freq_int = 1/Time_Samp/length(E_Ref) * (0:1/factor:length(E_Ref)-1)';
fl_int = factor*(fl(1)-1)+1 : factor*(fl(end)-1)+1;

sigma_drude_int = bsxfun(@rdivide, eps0*tau.*wp.^2, 1 - 1i*w*Freq_int*tau );
eps_debye_int = magD ./ (1 - 1i*w*Freq_int*tauD);
eps_lorentz_int = 1i*bsxfun(@rdivide, wpL.^2,  eps0*( 1i*(w0^2 - (w*Freq_int).^2) + w*Freq_int*gam) );
eps_int = epsi + eps_lorentz_int + eps_debye_int;
eps_int = eps_int * dim;
sigma_dark_int = bsxfun(@times, 1i*eps0*w*Freq_int, epsi - eps_int); %Complex-valued photoconductivity
n_int = sqrt(eps_int);
r0_int = (1-n_int)./ (1+n_int);
Re_int = -(1+r0_int)./r0_int .* ( Z0*d*sigma_drude_int ) ./ ( 1 + n_int + Z0*d*sigma_drude_int ) + 1;
epsilon_int = eps_int + bsxfun(@rdivide, + 1i*sigma_drude_int, eps0*w*Freq_int);
loss_int = -imag(1./epsilon_int);
screening_int = real(1./epsilon_int);

sigma = 1 / ( Z0 * d ) * ...
        ( bsxfun(@times, R-1, n.^2 - 1 ) ./ ...  % Multiply matrix R by vector ns
        ( bsxfun(@times, R-1, 1-n) + 2));

epsilon = eps + bsxfun(@rdivide, + 1i*sigma, eps0*w*Freq); %corrected second addition of epsi here
loss = -imag(1./epsilon);
screening = real(1./epsilon);

set(groot, 'defaultAxesColorOrder', [0 0.447 0.741; 0.850 0.325 0.098]);  % Use only two colors for plots
Font = struct('FontName', 'Calibri Light', 'FontWeight', 'normal', ...
              'FontAngle', 'normal', 'FontUnits', 'points', 'FontSize', 28);

fig = figure('Name', pwd,...        'MenuBar','none',...
            'NumberTitle', 'off',...
            'Visible', 'off',...
            'Units', 'pixels',...
            'Position', [0 0 1200 900],...
            'CreateFcn', {@movegui,'center'},...
            'KeyPressFcn', @keyPressFig,...
            'Color', get(0, 'DefaultUIControlBackgroundColor'));

Slider = uicontrol('Style', 'Slide',...
            'Position', [75 25 200 15],...
            'Min', 1, 'Max',PumpDelays, 'Val', 7,...
            'SliderStep', [1/PumpDelays ceil(1/5)],...
            'KeyPressFcn' ,@keyPressFig,...
            'Callback', @Update_Plot);

Time_m= uimenu('Label','Time Domain');
        uimenu(Time_m, 'Label', 'E Reference', 'Callback', {@Update, 1});
        uimenu(Time_m, 'Label', 'E Pumped', 'Callback', {@Update, 2});
        uimenu(Time_m, 'Label', '\DeltaE ', 'Callback', {@Update, 3});
        uimenu(Time_m, 'Label', 'E Reference + E Pumped', 'Callback', {@Update, 4});
        uimenu(Time_m, 'Label', 'E Reference + \DeltaE', 'Callback', {@Update, 5});
%         uimenu(Men,'Label','Quit','Callback','exit','Separator','on','Accelerator','Q');

Freq_m= uimenu('Label','Frequency Domain');
        uimenu(Freq_m, 'Label', 'E Reference Amplitude', 'Callback', {@Update, 10});
        uimenu(Freq_m, 'Label', 'E Reference Phase', 'Callback', {@Update, 11});
        uimenu(Freq_m, 'Label', 'E Pumped Amplitude', 'Callback', {@Update, 12});
        uimenu(Freq_m, 'Label', 'E Pumped Phase', 'Callback', {@Update, 13});
        uimenu(Freq_m, 'Label', '\DeltaE Amplitude', 'Callback', {@Update, 14});
        uimenu(Freq_m, 'Label', '\DeltaE Phase', 'Callback', {@Update, 15});
        uimenu(Freq_m, 'Label', 'R - 1', 'Callback', {@Update, 16});
        uimenu(Freq_m, 'Label', 'Phi', 'Callback', {@Update, 17});
        uimenu(Freq_m, 'Label', 'R - 1 + Phi', 'Callback', {@Update, 18});
        uimenu(Freq_m, 'Label', 'R - 1 + Phi + Fit', 'Callback', {@Update, 19});
        uimenu(Freq_m, 'Label', 'Refractive Index', 'Callback', {@Update, 20});
        uimenu(Freq_m, 'Label', 'Reflection coefficient', 'Callback', {@Update, 21});
        uimenu(Freq_m, 'Label', 'Dark Conductivity', 'Callback', {@Update, 22});
        uimenu(Freq_m, 'Label', 'Conductivity', 'Callback', {@Update, 23});
        uimenu(Freq_m, 'Label', '\sigma_2 / \sigma_1', 'Callback', {@Update, 24});
        uimenu(Freq_m, 'Label', 'Conductivity + Fit', 'Callback', {@Update, 25});
        uimenu(Freq_m, 'Label', 'Dielectric Function ', 'Callback', {@Update, 26});
        uimenu(Freq_m, 'Label', 'Dielectric Function + Fit', 'Callback', {@Update, 27});
        uimenu(Freq_m, 'Label', 'Loss + Screening', 'Callback', {@Update, 28});
        uimenu(Freq_m, 'Label', 'Loss + Screening + Fit', 'Callback', {@Update, 29});
        uimenu(Freq_m, 'Label', 'Integrated conductivity', 'Callback', {@Update, 40});
        uimenu(Freq_m, 'Label', 'Free charge density', 'Callback', {@Update, 41});
        uimenu(Freq_m, 'Label', 'Mobility', 'Callback', {@Update, 42});

P_2D_m= uimenu('Label','2D Data');
        uimenu(P_2D_m, 'Label', 'E Reference', 'Callback', {@Update, 50});
        uimenu(P_2D_m, 'Label', 'E Pumped', 'Callback', {@Update, 51});
        uimenu(P_2D_m, 'Label', '\DeltaE', 'Callback', {@Update, 52});
        uimenu(P_2D_m, 'Label', 'E Reference Amplitude', 'Callback', {@Update, 53});
        uimenu(P_2D_m, 'Label', 'E Pumped Amplitude', 'Callback', {@Update, 54});
        uimenu(P_2D_m, 'Label', '\DeltaE Amplitude', 'Callback', {@Update, 55});
        uimenu(P_2D_m, 'Label', 'R - 1', 'Callback', {@Update, 56});
        uimenu(P_2D_m, 'Label', 'Phi', 'Callback', {@Update, 57});
        uimenu(P_2D_m, 'Label', 'Conductivity - real', 'Callback', {@Update, 58});
        uimenu(P_2D_m, 'Label', 'Conductivity - imag', 'Callback', {@Update, 59});
        uimenu(P_2D_m, 'Label', 'Loss', 'Callback', {@Update, 60});
        uimenu(P_2D_m, 'Label', 'Screening', 'Callback', {@Update, 61});

Opt_m = uimenu('Label','Options');
        uimenu(Opt_m, 'Label', 'Fix Axis',  'Callback', @Fix_Axis_Fun);
        uimenu(Opt_m, 'Label', 'Hold Axis', 'Callback', @Hold_Axis_Fun);
        uimenu(Opt_m, 'Label', 'Font',      'Callback', @Font_Fun);
        uimenu(Opt_m, 'Label', 'Grid',      'Callback', @Grid_Fun, 'Checked', 'on');
        uimenu(Opt_m, 'Label', 'Title',     'Callback', @Title_Fun, 'Checked', 'on');
        uimenu(Opt_m, 'Label', 'Slider',    'Callback', @Slider_Fun);
        uimenu(Opt_m, 'Label', 'Export',    'Callback', @Export_Fun);

Update_Plot();

function Export_Fun(varargin)
    assignin('base','PumpTimes', PumpTimes);
    assignin('base','Time', Time);
    assignin('base','E_Ref', E_Ref);
    assignin('base','E_Pump', E_Pump);
    assignin('base','dE', dE);
    assignin('base','Freq', Freq);
    assignin('base','E_Ref_Amp', E_Ref_Amp);
    assignin('base','E_Ref_Phase', E_Ref_Phase);
    assignin('base','E_Pump_Amp', E_Pump_Amp);
    assignin('base','E_Pump_Phase', E_Pump_Phase);
    assignin('base','dE_Amp', dE_Amp);
    assignin('base','dE_Phase', dE_Phase);
    assignin('base','R_Amp', R_Amp);
    assignin('base','R_Amp_1', R_Amp-1);
    assignin('base','Phi', Phi);
    assignin('base','R', R);
    assignin('base','sigma_drude', sigma_drude);
    assignin('base','tau', tau);
    assignin('base','tauD', tauD);
    assignin('base','wp', wp);
    assignin('base','wpL', wpL);
    assignin('base','eps_debye', eps_debye);
    assignin('base','eps_lorentz', eps_lorentz);
    assignin('base','eps', eps);
    assignin('base','sigma_dark', sigma_dark);
    assignin('base','n', n);
    assignin('base','r0', r0);
    assignin('base','dR', Re);
    assignin('base','sigma', sigma);
    assignin('base','epsilon', epsilon);
    assignin('base','loss', loss);
    assignin('base','screening', screening);
    assignin('base','Freq_int', Freq_int);
    assignin('base','sigma_drude_int', sigma_drude_int);
    assignin('base','eps_debye_int', eps_debye_int);
    assignin('base','eps_lorentz_int', eps_lorentz_int);
    assignin('base','eps_int', eps_int);
    assignin('base','n_int', n_int);
    assignin('base','sigma_dark_int', sigma_dark_int);
    assignin('base','dR_int', Re_int);
    assignin('base','r0_int', r0_int);
    assignin('base','epsilon_int', epsilon_int);
    assignin('base','loss_int', loss_int);
    assignin('base','screening_int', screening_int);
    assignin('base','dim', dim);

    RPhiERef = Freq(range) * ones(1,4*min(size(R_Amp)));
    RPhiERef(:, 2:4:end) = R_Amp(range, :);
    RPhiERef(:, 3:4:end) = Phi(range, :);
    RPhiERef(:, 4:4:end) = E_Ref_Amp(range, :);
    assignin('base','RPhiERef', RPhiERef);
    LossScreen = Freq(range) * ones(1,2*min(size(loss)));
    LossScreen(:, 1:2:end) = loss(range, :);
    LossScreen(:, 2:2:end) = screening(range, :);
    assignin('base','LossScreen', LossScreen);
end

function keyPressFig(h,evt) %#ok<INUSL>
    if strcmp(evt.Key,'escape')
        delete(fig);
    end
    if strcmp(evt.Key, 'rightarrow')==1
        if get(Slider, 'Value') < PumpDelays
            set(Slider, 'Value', get(Slider, 'Value')+1);
            Update_Plot();
        end
    elseif strcmp(evt.Key, 'leftarrow')==1
        if get(Slider, 'Value') > 1
            set(Slider, 'Value', get(Slider, 'Value')-1);
            Update_Plot();
        end
    end
end

%Function to keep axis constant
function Fix_Axis_Fun(varargin)
    if strcmp(get(gcbo, 'Checked'),'off')
        Fix_Axe = 1;
        xlimit = get(gca,'xlim');
        ylimit = get(gca,'ylim');
        zlimit = get(gca,'zlim');
        set(gcbo, 'Checked', 'on');
    else
        Fix_Axe = 0;
        Update_Plot();
        set(gcbo, 'Checked', 'off');
    end
end

% Hold on/off
function Hold_Axis_Fun(varargin)
    if strcmp(get(gcbo, 'Checked'),'off')
        Hold_Axe = 1;
        hold on;
        set(gcbo, 'Checked', 'on');
    else
        Hold_Axe = 0;
        hold off;
        set(gcbo, 'Checked', 'off');
    end
end

% Slider on/off
function Slider_Fun(varargin)
    if strcmp(get(gcbo, 'Checked'),'off')
        Slide = 1;
        set(Slider, 'Visible', 'on')
        set(gcbo, 'Checked', 'on');
    else
        Slide = 0;
        set(Slider, 'Visible', 'on')
        set(gcbo, 'Checked', 'off');
    end
end

% Grid on/off
function Grid_Fun(varargin)
    if strcmp(get(gcbo, 'Checked'),'off')
        Grid = 1;
        grid on;
        set(gcbo, 'Checked', 'on');
    else
        Grid = 0;
        grid off;
        set(gcbo, 'Checked', 'off');
    end
end

% Title on/off
function Title_Fun(varargin)
    if strcmp(get(gcbo, 'Checked'),'off')
        Title = 1;
        title(Title_Label);
        set(gcbo, 'Checked', 'on');
    else
        Title = 0;
        title('');
        set(gcbo, 'Checked', 'off');
    end
end

function Font_Fun(varargin)
    Font = uisetfont(gca, 'Update Font');
end

function Update(varargin)
    set(get(Time_m, 'Children'), 'Checked', 'off');
    set(get(Freq_m, 'Children'), 'Checked', 'off');
    set(get(P_2D_m, 'Children'), 'Checked', 'off');
    set(gcbo, 'Checked', 'on');
    Selection = varargin{3};
    Title_Label = get(gcbo, 'Label');
%     assignin('base','Title', Title);
    Update_Plot();
end

function Update_Plot(varargin)

    tp = int32(get(Slider,'value'));

    if Selection == 1
        plot(Time(tl), E_Ref(tl, tp));
        xlabel('Time (ps)');
        ylabel('E Reference');
    elseif Selection == 2
        plot(Time(tl), E_Pump(tl, tp));
        xlabel('Time (ps)');
        ylabel('E Pumped');
    elseif Selection == 3
          plot(Time(tl), dE(tl, tp));
        xlabel('Time (ps)');
        ylabel('\DeltaE');
    elseif Selection == 4
        plot(Time(tl), E_Ref(tl,tp), Time(tl), E_Pump(tl ,tp));
        xlabel('Time (ps)');
        ylabel('E Ref + E Pump');
    elseif Selection == 5
        plot(Time(tl), E_Ref(tl, tp), Time(tl), dE(tl, tp));
        xlabel('Time (ps)');
        ylabel('\DeltaE');

    elseif Selection == 10
        plot(Freq(fl), E_Ref_Amp(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('E Reference Amplitude');
    elseif Selection == 11
        plot(Freq(fl), E_Ref_Phase(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('E Reference Phase');
    elseif Selection == 12
        plot(Freq(fl), E_Pump_Amp(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('E Pumped Amplitude');
    elseif Selection == 13
        plot(Freq(fl), E_Pump_Phase(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('E Pumped Phase');
    elseif Selection == 14
        plot(Freq(fl), dE_Amp(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('\DeltaE Amplitude');
    elseif Selection == 15
        plot(Freq(fl), dE_Phase(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('\DeltaE Phase');
	elseif Selection == 16
        plot(Freq(fl), R_Amp(fl,tp)-1, 'x-');
        xlabel('\omega (THz)');
        ylabel('|R|-1');
    elseif Selection == 17
        plot(Freq(fl), Phi(fl,tp)./Freq(fl), 'x-');
        xlabel('\omega (THz)');
        ylabel('\Phi / freq');
    elseif Selection == 18
        plot(Freq(fl), R_Amp(fl,tp)-1, 'x-', Freq(fl), Phi(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('|R|-1 + Phi');
    elseif Selection == 19
        plot(Freq(fl), abs(( R(fl,tp)))-1, 'x', Freq(fl), angle(( R(fl,tp)))-1, 'x');
        %plot(Freq(fl), abs(( R(fl,tp)))-1);
        hold on;
        plot(Freq_int(fl_int), abs(Re_int(fl_int,tp))-1, Freq_int(fl_int), angle(Re_int(fl_int,tp)));
        xlabel('\omega (THz)');
        ylabel('|R|-1 + Phi');
	elseif Selection == 20
        plot(Freq(fl), real(n(fl,tp)), 'x', Freq(fl), angle(n(fl,tp)), 'x');
        hold on;
        plot(Freq_int(fl_int), real(n_int(fl_int,tp)), Freq_int(fl_int), imag(n_int(fl_int,tp)));
        xlabel('\omega (THz)');
        ylabel('n');
	elseif Selection == 21
        plot(Freq_int(fl_int), abs(r0_int(fl_int, tp)));
        hold on;
        plot(Freq(fl), abs(r0(fl, tp)), 'x', Freq(fl), abs(r0(fl, tp)), 'x');
        xlabel('\omega (THz)');
        ylabel('Reflection coefficient');
    elseif Selection == 22
        plot(Freq(fl), real(sigma_dark(fl, tp)), 'x', Freq(fl), imag(sigma_dark(fl, tp)), 'x');
        hold on;
%         plot(Freq_int(fli), real(sig_int(fli)),'x', Freq_int(fli), imag(sig_int(fli)),'x');
        plot(Freq_int(fl_int), real(sigma_dark_int(fl_int, tp)), Freq_int(fl_int), imag(sigma_dark_int(fl_int, tp)));
        xlabel('\omega (THz)');
        ylabel('\sigma');        
	elseif Selection == 23
        plot(Freq(fl), real(sigma(fl,tp)), 'x-', Freq(fl), imag(sigma(fl,tp)), 'x-');
        xlabel('\omega (THz)');
        ylabel('\sigma');
    elseif Selection == 24
        plot(Freq(fl), imag(sigma(fl,tp)) ./ real(sigma(fl,tp)), 'x-');
        xlabel('\omega (THz)');
        ylabel('\sigma');
	elseif Selection == 25
        plot(Freq(fl), real(sigma(fl,tp)), 'x', Freq(fl), imag(sigma(fl,tp)), 'x');
%         plot(Freq(l), real(sigmaFit(l)), Freq(l), imag(sigmaFit(l)), 'LineWidth',2);
        hold on;
        plot(Freq_int(fl_int), real(sigma_drude_int(fl_int,tp)), Freq_int(fl_int), imag(sigma_drude_int(fl_int,tp)));         
        xlabel('\omega (THz)');
        ylabel('\sigma');
    elseif Selection == 26
        plot(Freq(fl), real(epsilon(fl,tp)), 'x-', Freq(fl), imag(epsilon(fl,tp)), 'x-');
        xlabel('\omega (THz)');
        ylabel('\epsilon');
    elseif Selection == 27
        plot(Freq(fl), real(epsilon(fl,tp)), 'x', Freq(fl), imag(epsilon(fl,tp)), 'x');
        hold on;
        plot(Freq_int(fl_int), real(epsilon_int(fl_int,tp)), Freq_int(fl_int), imag(epsilon_int(fl_int,tp)));
        xlabel('\omega (THz)');
        ylabel('\epsilon');
    elseif Selection == 28
        plot(Freq(fl), loss(fl,tp), 'x-', Freq(fl), screening(fl,tp), 'x-');
        xlabel('\omega (THz)');
        ylabel('1/\epsilon');
    elseif Selection == 29
        plot(Freq(fl), loss(fl,tp), 'x', Freq(fl), screening(fl,tp), 'x');
        hold on;
        plot(Freq_int(fl_int), loss_int(fl_int,tp), Freq_int(fl_int), screening_int(fl_int,tp));
        xlabel('\omega (THz)');
        ylabel('1/\epsilon');
    
    elseif Selection == 40
        sig1int = trapz(real(sigma(ffl,:)));
        plot(PumpTimes, sig1int, 'x-');
        xlabel('\tau_p (ps)');
        ylabel('Integrated \sigma_1 (\omega)');
        set(gca,'xscale','log');
    elseif Selection == 41
        plot(PumpTimes, nfree, 'x-');
        xlabel('\tau_p (ps)');
        ylabel('n_{free}');
%         set(gca,'xscale','log');
    elseif Selection == 42
        plot(PumpTimes, mu, 'x-');
        xlabel('\tau_p (ps)');
        ylabel('\mu (cm²/(Vs))');
%         set(gca,'xscale','log');
    
	elseif Selection == 50
        cla;
        contourf(Time(tl), PumpTimes(tpl), (E_Ref(tl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(E_Ref(tl, :))) max(max(E_Ref(tl, :)))])
        xlabel('Time (ps)');
        ylabel('\tau_p (ps)');
    elseif Selection == 51
        cla;
        contourf(Time(tl), PumpTimes(tpl), (E_Pump(tl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(E_Pump(tl, :))) max(max(E_Pump(tl, :)))])
        xlabel('Time (ps)');
        ylabel('\tau_p (ps)');
    elseif Selection == 52
        cla;
        contourf(Time(tl), PumpTimes(tpl), (dE(tl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(dE(tl, :))) max(max(dE(tl, :)))])
        xlabel('Time (ps)');
        ylabel('\tau_p (ps)');
	elseif Selection == 53
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (E_Ref_Amp(fl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(E_Ref_Amp(fl, :))) max(max(E_Ref_Amp(fl, :)))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 54
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (E_Pump_Amp(fl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(E_Pump_Amp(fl, :))) max(max(E_Pump_Amp(fl, :)))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 55
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (dE_Amp(fl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(dE_Amp(fl, :))) max(max(dE_Amp(fl, :)))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 56
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (R_Amp(fl, tpl)-1)', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(R_Amp(fl, :)-1)) max(max(R_Amp(fl, :)-1))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 57
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (Phi(fl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(Phi(fl, :))) max(max(Phi(fl, :)))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 58
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (real(sigma(fl, tpl)))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(real(sigma(fl, :)))) max(max(real(sigma(fl, :))))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 59
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (imag(sigma(fl, tpl)))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(imag(sigma(fl, :)))) max(max(imag(sigma(fl, :))))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 60
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (loss(fl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(loss(fl, :))) max(max(loss(fl, :)))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
    elseif Selection == 61
        cla;
        contourf(Freq(fl), PumpTimes(tpl), (screening(fl, tpl))', 15, 'EdgeColor', 'None');
        colorbar;
        caxis([ min(min(screening(fl, :))) max(max(screening(fl, :)))])
        xlabel('\omega (THz)');
        ylabel('\tau_p (ps)');
%     elseif Selection > 49
%         if label == 1
%             ylabel('Fluence (\muJ/cm²)');
%         elseif label == 2
%             ylabel('\lambda (nm)');
%         end
    end

    set(gca, Font);
    set(findobj(gcf, 'type', 'line'), 'LineWidth', 1.5)
    str = ['\tau_p = ', num2str(PumpTimes(tp), 10), ' ps'];
    
%     if label == 1
%         str = [num2str(PumpTimes(tp), 10), ' \muJ/cm²'];
%         if Selection >= 50
%             ylabel('Fluence (\muJ/cm²)');
%         elseif Selection >= 30
%             xlabel('Fluence (\muJ/cm²)');
%             set(gca,'xscale','linear');
%         end
% 	elseif label == 2
%         str = ['\lambda = ', num2str(PumpTimes(tp), 10), ' nm'];
%         if Selection >= 50
%             ylabel('\lambda (nm)');
%         elseif Selection >= 30
%             xlabel('\lambda (nm)');
%             set(gca,'xscale','linear');
%         end
% 	elseif label == 3
%         str = ['T = ', num2str(PumpTimes(tp), 10), ' K'];
%         if Selection >= 50
%             ylabel('Temperature (K)');
%         elseif Selection >= 30
%             xlabel('Temperature (K)');
%             set(gca,'xscale','linear');
%         end        
%     end

    if label ~= 0
        if label == 1
            str = [num2str(PumpTimes(tp), 10), ' \muJ/cm²'];
            labelText = 'Fluence (\muJ/cm²)'; 
        elseif label == 2
            str = ['\lambda = ', num2str(PumpTimes(tp), 10), ' nm'];
            labelText = '\lambda (nm)';
        elseif label == 3
            str = ['T = ', num2str(PumpTimes(tp), 10), ' K'];
            labelText = 'Temperature (K)';
        end
        if Selection >= 50
            ylabel(labelText);
        elseif Selection >= 40
            xlabel(labelText);
%             set(gca,'xscale','linear');
        end
    end
    
    text('String', str, 'Units', 'normalized', 'Position', [0.9, 1.05], Font);
    
    if Fix_Axe  == 1 , set(gca,'xlim', xlimit, 'ylim', ylimit, 'zlim', zlimit);    end
    if Hold_Axe == 1 , hold on;    else     hold off;   end
    if Grid     == 1 , grid on;    end
    if Title    == 1 , title(Title_Label, 'FontWeight', 'normal'); end
    if Slide    == 1 , set(Slider, 'Visible', 'on'); else set(Slider, 'Visible', 'off');    end
    
end

set(fig, 'Visible', 'on');
drawnow;
warning off MATLAB:Axes:NegativeDataInLogAxis
if ispc
    warning off MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame
    set(get(handle(fig),'JavaFrame'),'Maximized',1);
end

end