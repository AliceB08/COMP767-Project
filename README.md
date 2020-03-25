# Reinforcement Learning Final Project - COMP767

## Getting started

Clone the repository `git clone xxx`.

### Data
Download the data from https://console.cloud.google.com/storage/browser/grid-cells-datasets. To do so you can install `gsutil` package with pip and use the commande `gsutil cp -r gs://grid-cells-datasets/square_room_100steps_2.2m_1000000 .` This will create a folder called "square_room_100steps_2.2m_1000000" containing the data.

The data is available for TensorFlow only. To convert it to Pytorch format without installing TensorFlow, you can go through Google Colab. Here are the steps:

1) in you drive, create a folder "COMP767/Project" and place in it the file `colab_convert_data.py`

2) create in "COMP767/Project" a folder called "data". In that folder, upload all the folder "square_room_100steps_2.2m_1000000" that you downloaded and that contains the Tensorflow data (0000-of-0099.tfrecord, 0001-of-0099.tfrecord, ...)

3) Create in the "data" folder in new folder called "torch": "COMP767/Projet/data/torch" and add to that new folder a random file. For some reason, colab is unable to detect an empty folder so you might get an error message if you don't add a random file. To recap: you should have all the tensorflow data in "COMP767/Projet/data/square_room_100steps_2.2m_1000000" and "COMP767/Projet/data/torch" with a random file in it.

4) You should be good to go! Create a new colab notebook, in the "COMP767/Projet" folder. Add and execute this cell:
```
from google.colab import drive
drive.mount('/content/drive')
!python "drive/My Drive/COMP767/Project/colab_convert_data.py"
```

5) Download the content of the torch folder (and remove the random file you added at the beginning).

**Please do not push to the github the data folder, as it takes several gigas of memory space**.

## Requirements

- Pytoch
- numpy
- matplotlib
- scipy


## Folders

The `path_integration` folder contains the Pytorch implementation of the paper. The repository can be found here: https://github.com/LPompe/gridtorch. The original repo from the paper (TensorFlow implementation) can be found here: https://github.com/deepmind/grid-cells.