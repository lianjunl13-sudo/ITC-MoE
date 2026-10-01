# Default hyperparameters

| Setting | Value |
|---|---|
| Anchor coefficient | 0.1; the zero-anchor ablation refits gains |
| Candidate size | 48 (6 x top-8); initial candidate experiments used 24 |
| Hot residual rank | 11 |
| Hot distance / confidence | 3 / 0.7, with two protected high-confidence tokens |
| Rank learning | Adam, lr 0.02, 90 steps, nine continuous rank variables per layer |
| Soft-mask temperature | Exponential decay from 96 to 8, scaled by dimension/2048 |
| Rank token weights | 1 + 2*(confidence-min)/(max-min), in [1,3] |
| Rank grid | gate/up (4,16,32), down (4,32,16) |
| Whitening | Input whitening; inverse-direction output whitening for down only |
| Cholesky regularization | 0.01 |
| Randomized decomposition | Oversample 16, one power iteration, seed 13037+31*layer+role |
| Gain fitting | Adam lr 0.01, 32 steps, batch at most 128, no weight decay |
| Gain constraints | Initial value 1, clamp [0,2], gradient norm clip 0.1 |
| Development selection | Zero residual, SVD initialization, and steps 8/16/24/32 |
| Residual covariance | Squared routing weights; beta=n_eff/(n_eff+128); regularization 1e-4 times mean scale |
| Residual SVD | Rank 11, oversample 24, three power iterations |
| Hot calibration | 32 training-source prompts: 24 train and 8 development, two rounds |
| Evaluation seeds | 16037 + sample index |
| Decoding | Batch 1, block 32, steps 32, maximum 4096, acceptance threshold 0.95, speculative |
| Sampling | Temperature 1, top_k 1, top_p 1 |

Rank loss is a token-weighted quadratic output error under a fixed normalized
rank budget. The value 65536 is a backward numerical scale that is divided back
out; it is not a loss coefficient.

Gain fitting minimizes normalized reconstruction MSE plus the anchor coefficient
times normalized distance from the SVD-initialized output. Both terms use the
teacher-output mean square as denominator. Development selection uses only the
main reconstruction error.

A computed token is Cold if its distance exceeds the Hot distance threshold,
it is outside the protected top two, and its vocabulary confidence is below
the Hot confidence threshold. Other computed tokens are Hot. Vocabulary
confidence, routing confidence, and the decoding acceptance threshold are
distinct quantities. Candidate restriction still computes all expert router
scores; no O(M) router-scoring implementation is claimed.
