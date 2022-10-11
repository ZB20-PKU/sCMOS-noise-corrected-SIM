% GT = imread('0 demo512.tif');
% GT = 0.75 / 256 * GT;I = 25;bg = 5;GT = 2*(GT.*I+bg);
% imwritestack(GT, '1 GT.tif')
% save('GT.mat','GT')
% load('GT.mat')

% run start.m
% clear all;

load('gainsub.mat')
load('varsub.mat')

%%   pattern generation
pIGT = 0;
if pIGT == 1
load('illuminationPattern.mat')
elseif pIGT == 0
load('moduleDepth.mat')
load('noisyRawData_patternXCoor.mat')
load('noisyRawData_patternYCoor.mat')
load('phase.mat')
% moduleDepthInverse = 1 ./ moduleDepth;
moduleDepth = [1 moduleDepth(1) moduleDepth(2) ...
               1 moduleDepth(3) moduleDepth(4)...
               1 moduleDepth(5) moduleDepth(6)];
nangles = 3;
nphases = 3;
% for coefficientIndex = 1: nangles
%     reconstructionCoefficient(nphases * (coefficientIndex - 1) + 2: nphases * coefficientIndex) = ...
%         mean(moduleDepthInverse((coefficientIndex - 1) * (nphases - 1) + 1: coefficientIndex * (nphases - 1)));
% end

InitialPhase = ones(1, nangles);
for phaseIndex = 1: nangles
    InitialPhase(phaseIndex) = sign(phase(:,:, (phaseIndex - 1) * (nphases - 1) + 1)) ...
        * mean(abs(phase(:,:,(phaseIndex - 1) * (nphases - 1) + 1: phaseIndex * (nphases - 1))));
end
dphase = linspace(0, 2*pi, nphases+1);
dphase = dphase(1: nphases);
phaseList = [InitialPhase(1)+dphase InitialPhase(2)+dphase InitialPhase(3)+dphase];
% phaseList(:)=0;

imgSize=512;
x = linspace(- imgSize/2, imgSize/2-1, imgSize);
y = linspace(- imgSize/2, imgSize/2-1, imgSize)';
xx2 = repmat(x, imgSize, 1); yy2 = repmat(y, 1, imgSize);
Irtest = zeros(imgSize, imgSize,nphases * nangles);
for index=1: 1: nphases * nangles
    kytest = 2 * pi * (patternXCoor(index, :) - imgSize) / (imgSize);
    kxtest = 2 * pi * (patternYCoor(index, :) - imgSize) / (imgSize);
    Irtest(:, :, index) = moduleDepth(index) .* exp(1i * (kxtest * xx2 + kytest * yy2));
end
illuminationPattern = [];
for index1 = 1: 1: nangles
    for index2 = 1: 1: nphases
        illuminationPattern(:, :, index2 + nphases*(index1-1)) = ...
            Irtest(:, :, 1 + nphases*(index1-1))+...
            Irtest(:, :, 2 + nphases*(index1-1)) .* exp(-1i * phaseList(index2 + nphases*(index1-1)))+...
            Irtest(:, :, 3 + nphases*(index1-1)) .* exp(1i * phaseList(index2 + nphases*(index1-1)));
    end
end
illuminationPattern = 0.5.*illuminationPattern;
% imwritestack(abs(illuminationPattern), 'illuminationPatternEst.tif')
end

%%  Initialization
load('clearRawData.mat')
load('noisyRawData.mat')
load('otfComplex512.mat')
otf = otfComplex512;
close all;

% rawImg = noisyRawData;
rawImg = noisyRawData;
weight = gainsub.^2 ./ varsub;
weight(:,:)=1;
% imwritestack(weight, 'weight.tif')
weightStack = repmat(weight, [1 1 9]);
stepSize1 = 0.4;%   0.2~0.6 (set smaller when w is very large >3 or small =0)
stepSize2 = 1;
stepSize3 = 0;% high bandpass filter para 0, 0.5/((stepSize1+0.1)*stepSize2)
stepSize4 = 2;% wiener para 0£¬2
iterationTime = 1e8;
decay = 0;
groupNum = 1;
stopTH = 1e-4;
maskWidth = 0.25;
close all

% otfMask = zeros(size(otf));otfMask(otf > 1e-4) = 1;
otfStack = repmat(otf, [1 1 9]);
otfConjStack = repmat(conj(otf), [1 1 9]);
imgsz = 512; Pixelsize = 50e-12;NA = 1.1*2;Lambda = 488e-12;
highPassFilter = genfilter(imgsz,Pixelsize,NA,Lambda);
highPassFilter = highPassFilter./max(highPassFilter(:));
i = (1: iterationTime) - 1;
stepSize1 = stepSize1 + 0.1 * exp(- decay * i);
ReconImgStack = zeros(size(otf,1),size(otf,2), groupNum);

%%  Generate spatial mask
% load('ReconImgStack w0 error0.0001 Ap.mat')
sigma = maskWidth; paddingSize = 0;
x = 1: (imgSize + 2 * paddingSize);
y = (1: (imgSize + 2 * paddingSize))';
mask = repmat(sigmoid(sigma * (x - paddingSize)) - sigmoid...
    (sigma * (x - imgSize - paddingSize - 1)), ...
    imgSize + 2 * paddingSize, 1) .* ...
    repmat(sigmoid(sigma * (y - paddingSize)) - sigmoid...
    (sigma * (y - imgSize - paddingSize - 1)), ...
    1, imgSize + 2 * paddingSize);
% figure;imshow(mask,[])
rawImg = rawImg .* mask;
% figure;imshow(rawImg(:,:,1),[])

%%  Iteration
for imgInd = 1: groupNum
% rawData = rawImg(:,:,1:9);
rawData = rawImg(:,:,1 + 9*(imgInd-1):9+ 9*(imgInd-1));

%   initilization
% otf = abs(otf);
ReconImg = 2 * sum(rawData, 3)./ 9;
% otfMask = zeros(size(otf));otfMask(otf > 1e-4) = 1;
% ReconImg = imgShiftIFFT(imgShiftFFT(ReconImg).*otf.*otfMask./((abs(otf)).^2.*otfMask + 1e-2));
% imwritestack(abs(ReconImg), 'WF.tif')
% imwritestack(abs(imgShiftFFT(ReconImg))./abs(imgShiftFFT(GT)), 'diff.tif')
% imwritestack(angle(imgShiftFFT(ReconImg))-angle(imgShiftFFT(GT)), 'diff.tif')
% figure;imshow(ReconImg, [])
rawData = rawData .* weightStack;
% otfStack = repmat(otf, [1 1 9]);
% otfConjStack = repmat(conj(otf), [1 1 9]);
% highPassFilter = 1 - abs(highPassFilter);
% % figure;imshow(highPassFilter, [])
% imwritestack(highPassFilter, 'highPassFilter.tif')
rawDataFFT = imgShiftFFT(rawData);
errorList = [];

%   iteration
% stepSize1 = 0.01;
% iterationTime = 60;
% decay = 0.1;
% figure;plot([1: length(stepSize1)], stepSize1)
tic
for i = 1: iterationTime
    laterReconImg = ReconImg;
    %   Step 1
    intermediateVariable = illuminationPattern .* repmat(ReconImg, [1 1 9]);
    %   Step 2
    intermediateVariableFFT = imgShiftFFT(intermediateVariable);
    tempVariable = imgShiftFFT(weightStack .* imgShiftIFFT(intermediateVariableFFT .* otfStack));
    % figure;imshow(log(abs(intermediateVariableFFT(:,:,4))+1), [])
    intermediateVariableFFT = intermediateVariableFFT + ...
        stepSize1(i) .* otfConjStack .* (rawDataFFT - tempVariable);
    intermediateVariable = imgShiftIFFT(intermediateVariableFFT);
    %   Step 3
    ReconImg =  ReconImg ...
        - stepSize2 .* sum(illuminationPattern .* (-intermediateVariable + illuminationPattern .* repmat(ReconImg, [1 1 9])), 3) ...
        - stepSize3 .* stepSize1(i) .* stepSize2 .* imgShiftIFFT((highPassFilter.^2) .* imgShiftFFT(ReconImg))...
        - 0.005 * (stepSize4) ^2 .* stepSize1(i) .* stepSize2 .* ReconImg;
   
%     errorMap = abs((ReconImg - GT).^2);
    errorMap = abs((ReconImg - laterReconImg));
    error = mean(errorMap(:));
    errorList = [errorList; error];
    disp(['group:' num2str(imgInd) ' / ' num2str(groupNum) ', round:' num2str(i) ' / ' num2str(iterationTime)])
    disp(['error:' num2str(error)])
    if length(errorList)>1
        if error <= stopTH || errorList(end)>errorList(end-1)
            disp('Terminating!')
            break;
        end
    end
end
start = floor(length(errorList)/3);
figure;plot([start: length(errorList)], errorList(start:end))
ReconImgStack(:,:,imgInd) = ReconImg;
toc
end

%%  Apodization
% load(['ReconImgStack w' num2str(stepSize4) ' error' num2str(stopTH) '.mat'])
[k_x, k_y] = meshgrid(-(imgSize) / 2: (imgSize) / 2 - 1, -(imgSize) / 2: (imgSize) / 2 - 1);
k_r = sqrt(k_x.^2 + k_y.^2); fc = 100; patternWaveVectorLength = 105;
apodizationRadius = patternWaveVectorLength + fc;
% apodizationRadius = apodizationRadiusRate * patternWaveVectorLength;
ApodizationMatrix = cos(pi * k_r / (2 * apodizationRadius)); 
indi = find(k_r > apodizationRadius); ApodizationMatrix(indi) = 0;
ReconImgStack = imgShiftIFFT(ApodizationMatrix .* imgShiftFFT(ReconImgStack));

%%  Output
% figure;imshow(abs(ReconImg), [])
% figure;imshow(abs(GT), [])
ReconImgStack(ReconImgStack<0)=0;
imwritestack(ReconImgStack, ['ReconImgStack w' num2str(stepSize4) ' error' num2str(stopTH) '.tif'])
% save(['ReconImgStack w' num2str(stepSize4) ' error' num2str(stopTH) '.mat'],'ReconImgStack')
% imwritestack(abs(imgShiftFFT(ReconImg))./abs(imgShiftFFT(GT)), '1 diff.tif')
% imwritestack(angle(imgShiftFFT(ReconImg))-angle(imgShiftFFT(GT)), '2 diff.tif')







