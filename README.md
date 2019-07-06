# promelpol

## Project

### Description

data.py is used to interact with the dataset (TODO: move dataset into /Resources/dataset).
helpers.py is used to store general purpose functions.

### Requirements

Keras
`conda install -c conda-forge keras`

## Image Generator

### Description

- Sub-project that generates the images in `Resources/imageset/`
- Inside imageset there is a folder for every shop. 
- Each shop is split into a train and a test folder.
- All images are represented by a csv. Every cell represents the pixel values

6 images are created for each item in each shop.
3 for training & 3 for testing.
1 image represents the annual sales of the item (31x12).

### Requirements

JDK 1.8
MariaDB (MySQL)
Leiningen

