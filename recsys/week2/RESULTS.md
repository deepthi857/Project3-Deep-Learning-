# Week 2 results (validation set)

| Metric | Value |
|---|---|
| top-1 accuracy | 0.0018 |
| top-5 accuracy | 0.0077 |
| top-10 accuracy | 0.0126 |
| top-50 accuracy | 0.0374 |
| top-100 accuracy | 0.0608 |

Candidate set: 40,772 items. Random guessing would give about 0.25% at top-100.

## Tuning notes
Validation top-100 was best after epoch 1 (0.083) and fell to 0.061 by epoch 3, while training loss kept dropping. This is overfitting, so fewer epochs or regularisation would help.
