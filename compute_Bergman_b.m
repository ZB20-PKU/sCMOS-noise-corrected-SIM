function b = compute_Bergman_b(g, weightStack, ...
                       illuminationPatternConj, ...
                       otfConjStack, Y, ZY, ...
                       BergmanTermParaAlpha, BergmanTermParaLamda)
    b = g .* sqrt(weightStack);
    b = imgShiftFFT(g);
	b = imgShiftIFFT(otfConjStack .* b);
    b = sum(illuminationPatternConj .* b, 3);
    b = BergmanTermParaLamda .* b + BergmanTermParaAlpha .* (Y - ZY);
end