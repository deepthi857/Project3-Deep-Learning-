# Final model

- Architecture: two-tower retrieval, 64-dimensional user and item embeddings
- Training: 3 epochs, Adam (lr 0.001), batch size 1024, 10% sample of the training rows
- Hardware: Google Colab T4 GPU
- Validation (50,000 sampled rows, 40,772 candidate items):
  - top-1: 0.0018
  - top-5: 0.0088
  - top-10: 0.0135
  - top-50: 0.0393
  - top-100: 0.0624
- Serving: L2-normalised embeddings exported as .npy files and loaded into Redis (489,553 user vectors plus a 100-item popular fallback)
