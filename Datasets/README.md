# Spiking Heidelberg Digits (SHD) Data Set V1.0 and Spiking Speech Commands (SSC) Data Set v1.0

We provide two distinct classification data sets for spiking neural networks: SHD and SSC are data sets composed of single spoken words which have been converted to spikes. The audio recordings stem from the [Heidelberg Digits](https://compneuro.net/datasets/hd_audio.tar.gz)  and Google's free [Speech Commands](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) data set. The data set are organized in a single HDF5 file for each partition. The data sets are covered in detail [here](https://arxiv.org/abs/1910.07407).

| Name     				| Classes 	| Samples (train/valid/test)	| Parent dataset
| -------------------------------------	| -------------	| ----------------------------- | ---------------------------
| Spiking Heidelberg Digits (SHD) 	| 20		| 8332/-/2088 			| Heidelberg Digits (HD) v1.0
| Spiking Speech Commands (SCC) 	| 35		| 75466/9981/20382		| Speech Commands (SC) v0.2

The data sets are licensed under the [Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). See the `LICENSE` file in this folder for full details. Originally, the data sets were located at [Heidelberg Spikes](https://compneuro.net/datasets/).

## History

Version 1.0 of SHD was released on August 17th 2019 and is based on version 1.0 of HD. Version 1.0 of SSC was released on August 17th 2019 and is based on version 0.02 of SC.

## Collection

SHD cotains the English digits:"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", and "nine", as well as the German ones: "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", and "neun", each of them repeated about 40 times for each speaker.
SSC comprises the twenty core command words, "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine" with most speakers saying each of them 5 times. There are also 10 auxiliary words, "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", and "Wow", repeated only once for each speaker.

## Organization

For maximum compatibility, the partitions of the SHD and the SSC datasets are provided in HDF5 format which can be read by most major programming languages. The HDF5 files are organized as follows:
```
root
|-spikes
   |-times[]
   |-units[]
|-labels[]
|-extra
   |-speaker[]
   |-keys[]
   |-meta_info
      |-gender[]
      |-age[]
      |-body_height[]
```
The `times` and `units` datum consist of two lists that contain the firing times and the unit id of which neuron has fired at the corresponding firing time. The `labels` datum contains a list of the respective word id, whereas the `speaker` datum holds the speaker id. In `keys` a list of strings is kept, holding the transformation from word id to spoken word. The datum `meta_info` contains the information kept in the original `speaker_db.json` of HD and is therefore omitted for SSC.

## Partitioning

SHD is partitioned into training and testing data sets by assigning the digits of 2 speakers exclusively to the testing dataset to create space for well-founded statements on generalization. In more detail, all digits spoken by the speakers 4 and 5 were added to the testing data set. Moreover, 5% of the recordings of each digit and language of all other speakers were appended to the testing data set.
For SSC, the partitioning is done by the intended hasing function with a validation percentage of 10% and a testing percentage of 20%.
 
## Processing

The raw audio files of HD and SC are processed with a 700 channel hydrodynamic [basilar membrane model](https://asa.scitation.org/doi/full/10.1121/1.2204438). The generated movements are converted to spikes by a transmitter pool based [hair cell model](https://asa.scitation.org/doi/10.1121/1.399379). A single layer of Bushy cells is incorporated to increase phase-locking the generated spikes.
The conversion model is freely available from https://github.com/electronicvisions/lauscher

## Example Code

For example code see https://compneuro.net/posts/2019-spiking-heidelberg-digits/


## Citation

If you use the SHD and/or the SSC data set in your work, please cite our article https://ieeexplore.ieee.org/document/9311226:

```
@article{cramer_heidelberg_2020,
	title = {The {Heidelberg} {Spiking} {Data} {Sets} for the {Systematic} {Evaluation} of {Spiking} {Neural} {Networks}},
	issn = {2162-2388},
	doi = {10.1109/TNNLS.2020.3044364},
	journal = {IEEE Transactions on Neural Networks and Learning Systems},
	author = {Cramer, B. and Stradmann, Y. and Schemmel, J. and Zenke, F.},
	year = {2020},
	pages = {1--14},
}
```
