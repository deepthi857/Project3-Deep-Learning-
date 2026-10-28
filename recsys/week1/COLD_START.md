# Cold-start strategy
- **Cold user:** < 3 purchases in the window, or no purchases at all. Their history is "PAD", so the user
  tower relies on demographics (age bucket, club status, news frequency) and time context.
  Users with no embedding in Redis get the **popular-items fallback** at serving time.
- **Cold item:** < 5 purchases. The item tower still gets category/colour/department embeddings,
  so new items get a meaningful vector from metadata alone.
- **Unknown values** map to the OOV index of each StringLookup layer.

