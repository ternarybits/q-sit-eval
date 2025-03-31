# Evaluating Q-SiT

See https://github.com/zzc-1998/Q-SiT

This repo contains some test code and assets to elaborate on the issue described in 
https://github.com/zzc-1998/Q-SiT/issues/12

Usage:
```
uv run qsit.py [-h] [--model {q-sit,q-sit-mini}] input_path
```

Q-align evaluation done using https://github.com/chaofengc/IQA-PyTorch

Results: https://docs.google.com/spreadsheets/d/1nSPoOuRhsN4Feyub2Ep0bTAlS7C-hAymTmwuzhtE7xI/edit?usp=sharing

Output when run on a MacBook Pro (M4 Pro, 24 GB RAM):

```
uv run qsit.py assets

Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Rating token IDs: {'Excellent': 49655, 'Good': 15216, 'Fair': 60795, 'Poor': 84103, 'Bad': 17082}
Processing 19 image(s)...
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [15.890625  12.3984375  9.984375   9.5078125  8.9453125]
probs [9.65427927e-01 2.93820502e-02 2.62825848e-03 1.63192280e-03
 9.29841582e-04]
PXL_20250305_022911068.jpg: Score (0-1): 0.9892 (Processed in 6.14 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.8671875 11.6640625  9.03125    8.8984375  8.7265625]
probs [0.95395569 0.03876401 0.0027862  0.00243968 0.00205442]
PXL_20250305_023036427.jpg: Score (0-1): 0.9850 (Processed in 5.98 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [13.8046875  10.875       7.38671875  5.05078125  5.0625    ]
probs [9.47541055e-01 5.06116995e-02 1.54635649e-03 1.49563237e-04
 1.51326241e-04]
PXL_20250305_001429222.PORTRAIT.jpg: Score (0-1): 0.9863 (Processed in 6.05 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [13.8359375  10.9921875   6.2890625   4.37890625  4.5703125 ]
probs [9.44369554e-01 5.49688997e-02 4.98397447e-04 7.37913558e-05
 8.93577383e-05]
PXL_20250305_002553676.MP.jpg: Score (0-1): 0.9859 (Processed in 5.99 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [12.7265625   9.96875     5.6328125   3.22460938  4.00390625]
probs [9.39409754e-01 5.95871089e-02 7.79971231e-04 7.01793361e-05
 1.52986669e-04]
PXL_20241226_215331485.MP.jpg: Score (0-1): 0.9845 (Processed in 5.98 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [12.8671875   9.8828125   6.30859375  3.7265625   3.8671875 ]
probs [9.50371949e-01 4.80613554e-02 1.34751167e-03 1.01899162e-04
 1.17285216e-04]
PXL_20250302_232023207.jpg: Score (0-1): 0.9871 (Processed in 5.92 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.265625  11.1953125 10.640625  11.546875  10.921875 ]
probs [0.85155866 0.03951799 0.02269327 0.05616638 0.0300637 ]
PXL_20250305_023529592.jpg: Score (0-1): 0.9066 (Processed in 6.01 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.0078125 11.6015625  7.3984375  4.9765625  5.1640625]
probs [9.15948615e-01 8.25752720e-02 1.23440030e-03 1.09559255e-04
 1.32153688e-04]
PXL_20241126_032640070.PORTRAIT.jpg: Score (0-1): 0.9785 (Processed in 5.96 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [13.7734375  11.125       7.44921875  4.59375     4.734375  ]
probs [9.32161176e-01 6.59613019e-02 1.67077399e-03 9.61177031e-05
 1.10630798e-04]
PXL_20250303_005541055.jpg: Score (0-1): 0.9825 (Processed in 5.95 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [13.484375  10.2109375  7.03125    4.265625   4.375    ]
probs [9.61851635e-01 3.64310967e-02 1.51548454e-03 9.53796738e-05
 1.06403715e-04]
PXL_20250303_190711606.jpg: Score (0-1): 0.9900 (Processed in 6.06 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [13.875      10.734375    7.29296875  4.859375    5.12109375]
probs [9.57010946e-01 4.13962306e-02 1.32549183e-03 1.16273492e-04
 1.51057998e-04]
PXL_20250224_162500348 (1).jpg: Score (0-1): 0.9887 (Processed in 6.05 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [11.34375     7.73828125  8.2265625  10.1328125   9.890625  ]
probs [0.62374764 0.01695016 0.0276205  0.18582555 0.14585615]
44009500.jpg: Score (0-1): 0.6967 (Processed in 2.06 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [13.4296875  10.421875    6.96875     4.453125    4.28515625]
probs [9.51297298e-01 4.69937272e-02 1.48719104e-03 1.20183460e-04
 1.01600724e-04]
PXL_20250302_232000476.jpg: Score (0-1): 0.9873 (Processed in 6.16 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [15.3671875 11.9453125  8.65625    7.875      8.140625 ]
probs [9.66040660e-01 3.15423449e-02 1.17617591e-03 5.38492572e-04
 7.02326716e-04]
PXL_20250305_023019695.jpg: Score (0-1): 0.9904 (Processed in 6.25 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.234375  11.5625     8.8828125  7.7109375  7.       ]
probs [9.29322076e-01 6.42370643e-02 4.40567230e-03 1.36481361e-03
 6.70373922e-04]
PXL_20250305_022922793.jpg: Score (0-1): 0.9800 (Processed in 6.11 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.9765625 11.65625    9.21875    8.84375    8.3671875]
probs [0.95894039 0.03465758 0.00302835 0.00208135 0.00129234]
PXL_20250305_023530082.jpg: Score (0-1): 0.9870 (Processed in 6.08 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.125      11.5546875   7.88671875  5.4296875   5.1875    ]
probs [9.26987173e-01 7.09253012e-02 1.81060035e-03 1.55148466e-04
 1.21777427e-04]
PXL_20250224_010659202.MP.jpg: Score (0-1): 0.9811 (Processed in 6.16 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.859375   12.2734375   9.6328125   8.3671875   7.67578125]
probs [9.23385287e-01 6.95543928e-02 4.96038856e-03 1.39914184e-03
 7.00789885e-04]
PXL_20250305_022803886.jpg: Score (0-1): 0.9784 (Processed in 6.20 seconds)
Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
logprobs [14.890625   11.7265625   9.125       8.2265625   7.87109375]
probs [9.54601656e-01 4.03354865e-02 2.99118353e-03 1.21802615e-03
 8.53647349e-04]
PXL_20250305_022918444.jpg: Score (0-1): 0.9867 (Processed in 6.14 seconds)

Average score (0-1) across 19 images: 0.9659
Average time per image: 5.86 seconds
Total time: 111.25 seconds
```
