function Ax = compute_Bergman_Ax(x, ...
                         illuminationPattern, illuminationPatternConj, ...
                         otfStack, otfConjStack, weightStack,...
                         highPassFilter, hpPara, wienerFilerPara, ...
                         BergmanTermParaAlpha, BergmanTermParaLamda)
	tempVariable1 = illuminationPattern .* repmat(x, [1 1 9]);
    tempVariableFFT = imgShiftFFT(tempVariable1);
    tempVariable1 = imgShiftFFT(weightStack .* imgShiftIFFT(tempVariableFFT...
        .* otfStack));
    tempVariable1 = imgShiftIFFT(otfConjStack .* tempVariable1);
    tempVariable1 = BergmanTermParaLamda .* sum(illuminationPatternConj .* tempVariable1, 3);
    
    tempVariable2 = hpPara .* imgShiftIFFT((highPassFilter.^2) .* imgShiftFFT(x));
    
    tempVariable3 = 0.005 * (wienerFilerPara) ^2 .* x;
    
    tempVariable4 = BergmanTermParaAlpha .* x;
    
    Ax = tempVariable1 + tempVariable2 + tempVariable3 + tempVariable4;
end