# Predict aggregation propensity, not solubility

The original spec framed the goal as rescuing "solubility", but WALTZ-DB 2.0 labels hexapeptides as amyloid-forming or not, and no available dataset measures solubility directly. Peep-tide therefore predicts aggregation propensity and defines a rescue as a candidate whose conservative bound on that probability is below 0.5. Calling the output "solubility" would claim a measurement the training data never made.
