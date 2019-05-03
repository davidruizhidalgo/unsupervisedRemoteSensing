function dataBand= spectralBand(data,wavelength,bandWavelength)

  [mdata, isdata]=min(abs(wavelength-bandWavelength));
  dataBand=data(:,:,isdata);


end
