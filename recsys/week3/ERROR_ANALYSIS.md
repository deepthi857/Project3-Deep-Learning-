# Error analysis (Week 3)

- 31.57% of validation purchases come from users unseen in training. These users have no history, so they get the popular-items fallback.
- 4.51% of validation purchases involve items unseen in training.
- 14.92% of training rows have an empty purchase history (first purchase in the window).
- 74.83% of customers and 74.48% of items are cold-start (very little history in the 12-week window).
- Validation top-100 peaked after epoch 1 and fell by epoch 3 while training loss kept dropping, so the model overfits. Fewer epochs or regularisation would help.
