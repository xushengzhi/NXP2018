%% VIDEO    
frame_t = [];
    
vidObj1 = VideoReader('Q:\Running PROJECTS\iCave\Dolphin data\2017-11-09 outdoor car bicycle pedestrian experiment\Video\Re-encoded radar rate\FILE0051-0054_joined.mp4');
vidHeight = vidObj1.Height;
vidWidth = vidObj1.Width;
SecsToSkip = 454;     % Video starts at 10:14:18 , section of interest starts at 10:22:27 and lasts 71 seconds
MatVid = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
'colormap',[]);                                                         % Create a MATLAB movie structure array, s.  
vidObj1.CurrentTime = SecsToSkip;                                       % Skip to start Radar time
k = 1;
MatVid(k).cdata = imresize(readFrame(vidObj1),.5);
frame_t(k) = vidObj1.CurrentTime;
while vidObj1.CurrentTime <= SecsToSkip + 71                                
    if  hasFrame(vidObj1)
        k = k+1;
        MatVid(k).cdata = imresize(readFrame(vidObj1),.5);
        frame_t(k) = vidObj1.CurrentTime;
    end

end

NFramesVideo = k;

%% RADAR

fid = fopen('Q:\Running PROJECTS\iCave\Dolphin data\2017-11-09 outdoor car bicycle pedestrian experiment\RADAR\GPS_09-11-2017-11-16-22.txt');
T = textscan(fid,'%s','delimiter','\n');
gnggai = find(contains(T{1},'$GNGGA'));
B = T{1};
l1 = char(B(gnggai(1))); lend = char(B(gnggai(end)));

RadarStartTime = str2double(l1(8:13));
RadarStopTime = str2double(lend(8:13));

h1 = str2double(l1(8:9)); hend = str2double(lend(8:9));
m1 = str2double(l1(10:11)); mend = str2double(lend(10:11));
s1 = str2double(l1(12:13)); send = str2double(lend(12:13));

ElapsedTimeInSec = send-s1 + (mend-m1)*60 + (hend-h1)*3600;

RadarStartTimeinSec = s1 + m1*60 + h1*3600;

NBursts = 22190;
RadarFPS = NBursts/ElapsedTimeInSec;
RadarT = 1/RadarFPS;
% We want beatsignals from 10:22:27 and lasts 71 seconds, same as video

BurstsToSkip = round(10*3600+22*60+27 - RadarStartTimeinSec)*RadarFPS;
NbBursts = 71*RadarFPS; % Number of bursts of interest

%% Loading BeatSignals 14001 - 18000
% load('Q:\Running PROJECTS\iCave\Dolphin data\2017-11-09 outdoor car bicycle pedestrian experiment\RADAR\Matlab beat signals\Driving back - 09-11-2017-11-16-22\BS_14001_16000.mat');
% load('Q:\Running PROJECTS\iCave\Dolphin data\2017-11-09 outdoor car bicycle pedestrian experiment\RADAR\Matlab beat signals\Driving back - 09-11-2017-11-16-22\BS_16001_18000.mat');
% BeatSignals = cat(5,BeatSignals_14001_16000,BeatSignals_16001_18000);
BeatSignals = cat(5,BeatSignals_801_1000,BeatSignals_1001_1200,BeatSignals_1201_1400,BeatSignals_1401_1600);

%% Processing settings
[Ntx,NRx,NSweeps,NSamp,NBursts] = size(BeatSignals);

NFFTR = 1024;   % FFT length range
NFFTD = 512;   % FFT length Doppler
NFFTA = 128; % FFT length angle

Ts  = settings.Chirp_time - settings.Reset_time - settings.DwellTime;    % Duration of the ramp section of the chirp in s (Sweep Time)
S = settings.BW/Ts;

% Range axis
Range  = 3e8/(2*S)*linspace(0,settings.Fs,NFFTR);      % in meters


window = hamming(settings.NChirps)*hann(settings.NSamples)'; % window range-Doppler
RWindow = (permute(repmat(hann(settings.NSamples),1,3,4),[2,3,1])); % Hann windowing for range FFT

max_range = 25;         % max range for AOA plots
max_range_idx = find(Range <= max_range,1,'last');
RAOA = [fliplr(Range(2:max_range_idx)) Range(2:max_range_idx)]; % range axis for AOA plot

% Velocity axis
fDop = linspace(-1/(2*settings.Chirp_time*settings.NTx),1/(2*settings.Chirp_time*settings.NTx),NFFTD); % Doppler frequency in Hz
Vr  = 3.6*3e8/(2*settings.Fc)*fDop; % Radial velocity in km/h

% Angle axis
k = linspace(-1,1,NFFTA);
aoa = 90 - asind(k);
aoa2=90-linspace(-90,90,361);

[AZ,F] = meshgrid(deg2rad(aoa),Range(1:max_range_idx));
[x,y]=pol2cart(AZ,F);

[AZ2,F2] = meshgrid(deg2rad(aoa2),Range(1:max_range_idx));
[x2,y2]=pol2cart(AZ2,F2);

my_Corcoeff = [
   0.0128 + 0.1034i
   0.1178 + 0.1354i
  -0.0567 - 0.1022i
  -0.0923 - 0.1740i
  -0.0012 + 0.0976i
   0.0821 + 0.1158i
  -0.0213 - 0.1079i
  -0.0374 - 0.1577i
  -0.0312 - 0.0864i
  -0.0753 - 0.1046i
   0.0983 + 0.1323i
   0.0627 + 0.1936i];

%% DISPLAY
RadarFPS = 15;
fig = figure;
SS = get(0,'ScreenSize');
set(fig,'pos',[100 100 SS(3)/2 SS(4)/2]);

% v = VideoWriter('EWICarPark_800_1126.mp4','MPEG-4');
% v.FrameRate = round(RadarFPS);
% open(v);

nBurst0 = 401; % Start radar burst
nBurst = nBurst0;
nFrame = 793 + 25 + nBurst0; % Video frame
i = 1;

while nBurst <= 1126-800
    % AOA
    BS = squeeze(BeatSignals(:,:,129,:,nBurst));  % [NTx,NRx,NSamp,NBursts] beat signals (8th sweep NO DOPPLER PROCESSING)
    RP =  fft(squeeze(BS(:,:,:)).*RWindow,NFFTR,3);
    data = reshape(permute(RP,[2,1,3])  ,12,NFFTR);

    % Ruoyu's monopulse processing
    [monopulse_AOA] = monopulse_realdata_processing4(data, settings, my_Corcoeff);
    AOA2 =( monopulse_AOA(:,1:max_range_idx));
    AOA3 = (AOA2-max(AOA2(:)));
    AOA3(:,Range<=2)=-50;
    hAOA_mp = subplot(2,2,4); 
    surf(-x2,y2,AOA3');
    axis image
    view(2);
    shading interp;
    grid off;
    caxis([-40 -10]);
    zlim([-50,0])
    hAOA_mp.YAxis.Visible = 'off'; 
    hAOA_mp.Color = hAOA_mp.Parent.Color;
    colormap jet
    xlabel('Range (m)');
    title('AOA - MIMO-Monopulse (MS3)')
 
    % My correction
     % Spectrum in k space with correction
     window =  (ones(NFFTR,1)*hann(12)');
     corr_data = data.*repmat(my_Corcoeff,1,NFFTR);
    spec_k_corr = (fft(  window  .*  corr_data.',  NFFTA, 2)).'; 
    AOA_corr = fftshift(20*log10(abs(spec_k_corr(:,1:max_range_idx))),1);
    AOA_corr2 = AOA_corr-max(AOA_corr(:));
    AOA_corr2(:,Range<=2)=-50;
    idxMin50 = find(AOA_corr2<=-50);
    AOA_corr2(idxMin50) = -50;
    hAOA_corr = subplot(2,2,2); 
    surf(-x,y,squeeze(AOA_corr2)');
    axis image
    view(2);
    shading interp;
    caxis([-40 -10]);
    grid off;
    zlim([-50 0])
    hAOA_corr.YAxis.Visible = 'off'; 
    hAOA_corr.Color = hAOA_corr.Parent.Color;
    colormap jet
    xlabel('Range (m)');
    title('AOA - Beamforming (FFT)')

    % Range - Doppler
    RD = fft2(window.*detrend(squeeze(BeatSignals(1,1,:,:,nBurst))')',NFFTD,NFFTR);
    subplot(2,2,3);imagesc(Vr,Range(1:end/2),fftshift(20*log10(abs(RD(:,1:NFFTR/2)')/max(abs(RD(:)))),2));
    axis xy; ylim([0 max_range]);xlabel('Radial velocity (km/h)');ylabel('Range (m)')
    caxis([-40 -10]);colormap jet;title(['Burst #' num2str(nBurst+801)]);
      
    % Video
    subplot(2,2,1)
    imshow(MatVid(nFrame).cdata)
    nFrame = nFrame + 1;
    nBurst = nBurst + 1;
    i = i+1;

    pause(.05);
    
%     f = getframe(fig);
%     writeVideo(v,f);
end
% close(v);
