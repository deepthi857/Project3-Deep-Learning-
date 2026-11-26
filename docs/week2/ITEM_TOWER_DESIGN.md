# Item tower design

| Feature | Type | Vocabulary size | Embedding size |
|---|---|---|---|
| article_id | categorical | 40772 | 64 |
| product_type_name | categorical | 124 | 16 |
| colour_group_name | categorical | 50 | 16 |
| department_name | categorical | 238 | 16 |
| garment_group_name | categorical | 21 | 16 |
| item_pop_log | numeric | - | - |

All embeddings and the popularity value are concatenated, passed through a
Dense(128, relu) layer and a Dense(64) output layer. The output is the item vector.