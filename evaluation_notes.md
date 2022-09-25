# Model Evaluation Notes

## Metrics Used
Precision@K: Fraction of recommended items that are relevant
Recall@K: Fraction of relevant items that were recommended
Coverage: Percentage of catalog the system can recommend

## Results (Content-Based, K=5)
The Dark Knight: 4/5 relevant -> Precision 0.80
Inception: 3/5 relevant -> Precision 0.60
The Avengers: 5/5 relevant -> Precision 1.00
Average Precision@5: 0.73

## Limitations
No real user ratings available for offline evaluation
Cold-start problem for obscure movies
