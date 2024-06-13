## CV Lab Project

The goal of the project WiSAR was to locate moving people in the forest by tackling an issue of strong occlusion of people due to the trees. Therefore, unsupervised localization algorithm was used. Our approach employed convolutional autoencoder for reconstruction of input images. The difference between input and reconstructed images was then used for applying classical methods for detection of objects. Prediction algorithm was tested on validation set.

### DATASET
Provided training data was first pre-processed by integrating the ten images from each camera for each individual timestep by using the provided homographies, resulting in seven integrated images per training folder. Resulting images were cropped to the area where the FOV of all cameras converge. As the resulting dataset was very small, data augmentation in the form of random cropping of size 128x128px. was used for the training.