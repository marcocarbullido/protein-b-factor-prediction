# rcsb-bfactor-transformer
Deep learning approach to predict protein b-factors from sequence

Demonstrates that self-attention can improve modeling of long-range dependencies in protein sequences. Through the use of transformer models and protein embeddings, this approach exceeds the performance of state-of-the-art models, reaching an average Pearson correlation coefficient of 0.822, compared to 0.799 (Pandey et al.), amid requiring far fewer parameters.

The model weights and training pipeline are made public.

TODO:
   - scale loss similar to Pandey et. al. (https://www.cell.com/patterns/pdf/S2666-3899(23)00160-5.pdf)
   - modeling prediction error/confidence

Sources:
1. Pandey, Akash et al. “B-factor prediction in proteins using a sequence-based deep learning model.” Patterns (New York, N.Y.) vol. 4,9 100805. 4 Aug. 2023, doi:10.1016/j.patter.2023.100805
