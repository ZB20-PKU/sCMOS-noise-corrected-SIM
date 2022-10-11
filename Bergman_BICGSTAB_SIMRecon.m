% GT = imread('0 demo512.tif');
% GT = 0.75 / 256 * GT;I = 25;bg = 5;GT = 2*(GT.*I+bg);
% imwritestack(GT, '1 GT.tif')
% save('GT.mat','GT')
% load('GT.mat')

% run start.m
clear all;
load('gainsub.mat')
load('varsub.mat')

%%   pattern generation
pIGT = 0;
if pIGT == 1
load('illuminationPattern.mat')
elseif pIGT == 0
load('moduleDepthGT.mat')
load('patternXCoor.mat')
load('patternYCoor.mat')
load('phaseGT.mat')
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
illuminationPatternConj = conj(illuminationPattern);

%%  Initialization
% load('clearRawData.mat')
load('noisyRawData.mat')
% load('otfComplex512.mat')
% otf = otfComplex512;
otf = imreadstack('otf512.tif');
% rawImg = noisyRawData;
rawImg = noisyRawData;

avgFlag = 0;% 0:Weight, 1:Average
wienerFilerPara = 2;% wiener para 0£¬2
hpPara = 0;% high bandpass filter para 0, 1
BergmanTermParaAlpha = 1e-3;% alpha in equation (25) 0, 1
decay = 0;
groupNum = 1;
stopTH = 1e-4;
maskWidth = 0.25;
iterationTime = 1e8;
nonNegIterationTime = 6;% 5, 10, 20
weightGroupNum = 1;
BergmanTermParaLamda = 1;% lamda in equation (25) 0, 1
close all

%%  Weight
% weight = varsub./(gainsub.^2);
% weightStack = repmat(weight, [1 1 9]);
% for imgInd = 1: 9
% %     tmpWeight = weight+...
% %         max(mean(rawImg(:,:,imgInd:9:imgInd+9*(weightGroupNum-1)), 3),0)+1;
%     tmpWeight = weight;
%     tmpWeight = 1./tmpWeight;
%     tmpWeight = tmpWeight./mean(tmpWeight(:));
%     weightStack(:,:,imgInd) = tmpWeight;
% end
% if avgFlag == 1
%     weightStack(:,:,:)=1;
% end

weight = gainsub.^2 ./ varsub;
weight = weight./mean(weight(:));
if avgFlag == 1
    weight(:,:)=1;
end
% imwritestack(weight, 'weight.tif')
weightStack = repmat(weight, [1 1 9]);

%%  OTF
% otfMask = zeros(size(otf));otfMask(otf > 1e-4) = 1;
otfStack = repmat(otf, [1 1 9]);
otfConjStack = repmat(conj(otf), [1 1 9]);
imgsz = 512; Pixelsize = 50e-12;NA = 1.1*2;Lambda = 488e-12;
highPassFilter = genfilter(imgsz,Pixelsize,NA,Lambda);
highPassFilter = highPassFilter./max(highPassFilter(:));
xUpdateIndex = (1: iterationTime) - 1;
% stepSize1 = stepSize1 + 0.1 * exp(- decay * i);
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
poolobj = gcp('nocreate');
if isempty(poolobj)
    disp(['[*] Launch parallel computing.']);
    CoreNum = 2; 
    parpool(CoreNum);
end
parfor_progress(groupNum);
tic
% parfor imgInd = 1: groupNum
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
% rawData = rawData .* weightStack;
% otfStack = repmat(otf, [1 1 9]);
% otfConjStack = repmat(conj(otf), [1 1 9]);
% highPassFilter = 1 - abs(highPassFilter);
% % figure;imshow(highPassFilter, [])
% imwritestack(highPassFilter, 'highPassFilter.tif')
% rawDataFFT = imgShiftFFT(rawData);
g = rawData .* sqrt(weightStack);
g0 = g; Y0 = 0; ZY0 = 0;
gk = g0; Yk = Y0; ZYk = ZY0;

b = compute_Bergman_b(gk, weightStack, illuminationPatternConj, otfConjStack, ...
    Y0, ZY0, BergmanTermParaAlpha, BergmanTermParaLamda);
Ax0 = compute_Bergman_Ax(ReconImg, illuminationPattern, illuminationPatternConj, ...
                otfStack, otfConjStack, weightStack,...
                highPassFilter, hpPara, wienerFilerPara, BergmanTermParaAlpha, ...
                BergmanTermParaLamda);
r0 = b - Ax0; r0_hat = r0; rou0 = 1; alpha = 1; w0 = 1; v0 = 0; p0 = 0;
ri_1 = r0; roui_1 = rou0; wi_1 = w0; pi_1 = p0; vi_1 = v0;
% figure; imshow(r0, [])
%   iteration
% stepSize1 = 0.01;
% iterationTime = 60;
% decay = 0.1;
% figure;plot([1: length(stepSize1)], stepSize1)
% tic
for allVariableUpdateIndex = 1: nonNegIterationTime
    errorList = [];
    for xUpdateIndex = 1: iterationTime
        roui = sum(sum(r0_hat.* ri_1));
        belta = (roui./roui_1)*(alpha./wi_1);
        pI = ri_1+ belta*(pi_1-wi_1.*vi_1);
        vi = compute_Bergman_Ax(pI, illuminationPattern, illuminationPatternConj, ...
                otfStack, otfConjStack, weightStack,...
                highPassFilter, hpPara, wienerFilerPara, ...
                BergmanTermParaAlpha, BergmanTermParaLamda);
        alpha = roui/(sum(sum(r0_hat.* vi)));
        s = ri_1 - alpha*vi;
        t = compute_Bergman_Ax(s, illuminationPattern, illuminationPatternConj, ...
                    otfStack, otfConjStack, weightStack,...
                    highPassFilter, hpPara, wienerFilerPara, ...
                    BergmanTermParaAlpha, BergmanTermParaLamda);
        wi = sum(sum(t.* s))/sum(sum(t.* t));
        laterReconImg = ReconImg;
        ReconImg = ReconImg+alpha.*pI+wi.*s;
        ri = s - wi.*t;
        %   Update
        ri_1 = ri;
        roui_1 = roui;
        wi_1 = wi;
        pi_1=pI;
        vi_1=vi;
    %     errorMap = abs((ReconImg - GT).^2);
        errorMap = abs((ReconImg - laterReconImg));
        error = mean(errorMap(:))
        errorList = [errorList; error];
%         disp(['group:' num2str(imgInd) ' / ' num2str(groupNum) ...
%             ',All round:' num2str(allVariableUpdateIndex) ' / ' num2str(nonNegIterationTime) ...
%             ',ReconImg round:' num2str(xUpdateIndex) ' / ' num2str(iterationTime)])
%         disp(['error:' num2str(error)])
        if length(errorList)>1
            if error <= stopTH %|| errorList(end)>errorList(end-1)
%                 disp('ReconImg Update Terminating!')
                break;
            end
        end
    end
    
    Yk = ReconImg + ZYk;
%     figure;imshow(ReconImg, [])
    Yk(Yk < 0) = 0;
%     figure;imshow(Yk, [])
    gk = gk + g - compute_Bergman_Jx(ReconImg, ...
                         illuminationPattern, otfStack, weightStack);
%     figure;imshow(gk(:,:,3), [])
%     figure;imshow(g(:,:,1), [])
    ZYk = ZYk + ReconImg - Yk;
    
    b = compute_Bergman_b(gk, weightStack, illuminationPatternConj, otfConjStack, ...
        Yk, ZYk, BergmanTermParaAlpha, BergmanTermParaLamda);
    Ax0 = compute_Bergman_Ax(ReconImg, illuminationPattern, illuminationPatternConj, ...
                    otfStack, otfConjStack, weightStack,...
                    highPassFilter, hpPara, wienerFilerPara, ...
                    BergmanTermParaAlpha, BergmanTermParaLamda);
    r0 = b - Ax0; r0_hat = r0; rou0 = 1; alpha = 1; w0 = 1; v0 = 0; p0 = 0;
    ri_1 = r0; roui_1 = rou0; wi_1 = w0; pi_1 = p0; vi_1 = v0;
end
start = floor(length(errorList)/3);
% figure;plot([start: length(errorList)], errorList(start:end))
ReconImgStack(:,:,imgInd) = ReconImg;
parfor_progress;
% toc
end
parfor_progress(0);
toc

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
% saveName = ['ReconWegImgStack w' num2str(wienerFilerPara) ' Berg' num2str(BergmanTermPara) ' error' num2str(stopTH)];
% if avgFlag == 1
%     saveName = ['ReconAvgImgStack w' num2str(wienerFilerPara) ' Berg' num2str(BergmanTermPara) ' error' num2str(stopTH)];
% end
ReconImgStack(ReconImgStack<0)=0;
saveName = ['weg-Tik-nonnegative'];
imwritestack(ReconImgStack, [saveName '.tif'])
% save([saveName '.mat'],'ReconImgStack')
% imwritestack(abs(imgShiftFFT(ReconImg))./abs(imgShiftFFT(GT)), '1 diff.tif')
% imwritestack(angle(imgShiftFFT(ReconImg))-angle(imgShiftFFT(GT)), '2 diff.tif')







