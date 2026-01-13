This repository tracks the progress of this model's training and configuration.

This model is being trained on the FRIEhand datatset, which consists of 32,560 images, each dimensions 224x224 and in RGB color scheme.
The dataset contains several targets like masks, bounding boxes, and more, but this only uses the 21 hand position coordinates. 

The hand position coordinates are cartesian coordinates of key-components of the hand. There is the main coordinate, the wrist, and the 5 appendages extending from it are the fingers. In the raw data, each coordinate of the 21 values are in x, y, z format. The original dataset used images with known focal points and depths, which allows the additional element of depth to be considered.

This model will not be used with a known depth, so it should not be trained to predict or use them. To factor the z-coordinate out of the orignial coordinates, the dataset's K values are used. The K-values are a 3x3 matrix containing information about the camera's position or the image's depth. The current state of the coordinates is a 21x3 matrix, so to multiply against a 3x3 matrix, we transpose the 21x3 matrix to get a 3x21 matrix. This allows us to multiply every images [x,y,z] coordinates against the 3x3 K matrix and return a 3x21 matrix. Then, we slice this matrix along its coordinates to get two matrices; x, y values and z. Dividing by x, y by z gives us the factored x & y coordinates.

The model is trained on the images as input and the xy's as targets. The dataset is split 80 parts training, 10 parts testing, and 10 parts validation.