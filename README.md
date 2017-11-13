# TrainAClock

Starts with a image of hands, and rotates them by an angle to make a clock showing time with multiple of 5 minutes.
These images are used for training a simple convolution network. 
This CNN has two fully connected layers at  the head, they both feed a cross-entropy layer that minimize erros individually for minutes and hands.

TODO:
    - Can we train a network to get minutes that is not a multiple of 5? I.e. 60 outputs for minutes.
    - It's probably faster to create 12 minute hand images (does not apply to hours) with diff images.
    - Should we tru and compress the network?