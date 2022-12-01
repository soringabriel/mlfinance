## Phase 1 - Classification using percentage differences from a single index

First phase of the experiment was a failure. This are the accuracies for 5 min interval:

Accuracy for knn 30 0.5025380710659898
Accuracy for knn max 0.5025380710659898
Accuracy for svm 0.4619289340101523
Accuracy for naive bayes 0.37055837563451777
Accuracy for decision tree 0.5634517766497462
Accuracy for rf 0.5279187817258884

On 197 predictions. For 15 min, 30 min and so on, the results were even worse.

## Phase 2 - Classification using percentage differences from multiple similar indexes (NASDAQ, SPX, DJ30 etc.) - More training data better chances of better results + Increase past data size