```
conda install -c conda-forge gnuradio
```
- **NMSE** is reported in dB. The smaller the better. Perfect alignment yields 0 dB while more
  negative values indicate larger errors. After DPD an NMSE around −25 dB or
  lower is typically considered good.
- **EVM** is reported as a percentage. The smaller the better. 0% is ideal. Communication systems often
  aim for EVM below a few percent after linearization.
