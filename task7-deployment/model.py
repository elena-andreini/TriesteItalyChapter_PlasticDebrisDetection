import numpy as np

import time  #TODO: Remove once the model is in place

#TODO: Cache result when further along in testing/prototyping
def predict(data):
    # Pretend to do some computation.
    time.sleep(1)

    # For now, just make a mostly-black result with a small white area
    # as our fake prediction.
    predictions = np.zeros(shape=data.shape[1:])  # All black

    # White area (our positives)
    predictions[
        (data.shape[1] // 3):(2 * data.shape[1] // 3),  # Middle third
        (data.shape[2] // 3):(2 * data.shape[2] // 3)   # Middle third
    ] = 1

    # Repeat single channel to make a black-and-white RGB image.
    predictions = np.stack([predictions for _ in range(3)], axis=-1)

    return predictions