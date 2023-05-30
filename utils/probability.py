import random

# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    index = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return index
        index += 1


# distr: [float] -> (float)
# Normalize a list of floats to a probability distribution.
# Gamma is an egalitarianism factor which tempers the distribution
# toward being uniform as it grows from 0 to 1.
def exp_distr(weights, gamma=0.0):
    weights_sum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / weights_sum) + (gamma / len(weights)) for w in weights)


def mean(aList):
    theSum = 0
    count = 0

    for x in aList:
        theSum += x
        count += 1

    return 0 if count == 0 else theSum / count
