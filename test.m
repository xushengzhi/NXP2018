close all;
clc;
clear all;

%% Load data
% cd 'D:\20171107EWI'
% load('Matlab beatsignals\RadarSettings.mat');
% 
% DecimationRate = settings.DecimationRate;
% BW = settings.BW;
% Chirp_time = settings.Chirp_time;
% DwellTime = settings.DwellTime;
% Reset_time =settings.Reset_time;
% NSamples = settings.NSamples;
% Fs = settings.Fs;
% Nchirps = settings.NChirps;
% Fc = settings.Fc;
% NRx = settings.NRx;
% NTx = settings.NTx;
% FrameRate = settings.FrameRate;
% MIMO_coding_matrix = settings.MIMO_coding_matrix;
% 
% clear settings

%% Extract 
% load('Matlab beatsignals\BeatSignals_1_600.mat')
% 
% save('1_100.mat', 'BeatSignals_1_100');
% clear BeatSignals_1_100 
% 
% save('101_200.mat', 'BeatSignals_101_200');
% clear BeatSignals_101_200
% 
% save('201_300.mat', 'BeatSignals_201_300');
% clear BeatSignals_201_300
% 
% save('301_400.mat', 'BeatSignals_301_400');
% clear BeatSignals_301_400
% 
% save('401_600.mat', 'BeatSignals_401_600');
% clear BeatSignals_401_600

% load('Matlab beatsignals\BeatSignals_601_1200.mat')
% 
% save('1001_1200.mat', 'BeatSignals_1001_1200', '-v7.3');
% clear BeatSignals_1001_1200
% 
% save('601_800.mat', 'BeatSignals_601_800', '-v7.3');
% clear BeatSignals_601_800

% save('801_1000.mat', 'BeatSignals_801_1000', '-v7.3');
% clear BeatSignals_801_1000

% save('301_400.mat', 'BeatSignals_301_400');
% clear BeatSignals_301_400
% 
% save('401_600.mat', 'BeatSignals_401_600');
% clear BeatSignals_401_600

%% Test

% load('1001_1200.mat')
Data = squeeze(BeatSignals_1001_1200(:,:, 1:64, :, 91));

%% Virtual array

VirData = reshape(permute(Data, [1, 2, 3, 4]), 12, 64, 512);

figure()
ftdata = normlize(db(fftshift(fft2(squeeze(VirData(:,:,:)), 1024, 1024) )));
imagesc(ftdata')
colormap(jet)
caxis([-40, 0])
colorbar()




