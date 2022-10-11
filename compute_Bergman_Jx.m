function Jx = compute_Bergman_Jx(x, ...
                                 illuminationPattern, otfStack, weightStack)
	Jx = illuminationPattern .* repmat(x, [1 1 9]);
    tempVariableFFT = imgShiftFFT(Jx);
    Jx = sqrt(weightStack) .* imgShiftIFFT(tempVariableFFT .* otfStack);   
end