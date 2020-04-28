load('1050_1150.mat')

for i=1:100
    name = sprintf('%dth_slice.mat', i+1049)
% save('1001_1200.mat', 'BeatSignals_1001_1200', '-v7.3');
% clear BeatSignals_1001_1200
    Dat = data(i, :, :, :, :);
    save(name, 'Dat')
end