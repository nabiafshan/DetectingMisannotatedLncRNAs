def get_stats(predhis, y_true):
  n_values = np.max(y_true) + 1
  mask = np.eye(n_values)[y_true]
  for epoch in range(len(predhis)):
    thisEpochProbs = predhis[epoch]
    thisEpochPreds = thisEpochProbs.argmax(axis = 1).reshape(-1, 1)

    # For mean, sd
    thisMaskedProbs = (thisEpochProbs * mask).sum(axis = 1).reshape(-1, 1)   

    # Save
    if epoch == 0:
      allEpochPreds = thisEpochPreds
      allEpochClassProbs = thisMaskedProbs
    elif epoch > 0:
      allEpochPreds = np.hstack((allEpochPreds, thisEpochPreds))
      allEpochClassProbs = np.hstack((allEpochClassProbs, thisMaskedProbs))
  correctness = allEpochPreds.mean(axis=1)
  confidence = allEpochClassProbs.mean(axis=1)
  variability = allEpochClassProbs.std(axis=1)
  return correctness, confidence, variability