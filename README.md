# rcsb-bfactor-transformer
Deep learning approach for predicting protein b-factors from sequence

Demonstrates implementation of transformer architecture for learning protein embeddings to predict b-factors, achieving state-of-the-art performance. This approach reaches an average Pearson correlation coefficient of 0.82, compared to 0.80 (Pandey et al.) for b-factor prediction, while requiring fewer parameters.

The model weights and training pipeline are made public.

TODO:
   - scale loss similar to Pandey et al. (https://www.cell.com/patterns/pdf/S2666-3899(23)00160-5.pdf)
   - modeling prediction error/confidence

Sources:
1. Pandey, Akash et al. “B-factor prediction in proteins using a sequence-based deep learning model.” Patterns (New York, N.Y.) vol. 4,9 100805. 4 Aug. 2023, doi:10.1016/j.patter.2023.100805
