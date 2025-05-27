# Deep learning methods using imagery from a smartphone for recognizing sorghum panicles and counting grains at an on-farm scale


## Summary
- [Deep learning methods using imagery from a smartphone for recognizing sorghum panicles and counting grains at an on-farm scale](#deep-learning-methods-using-imagery-from-a-smartphone-for-recognizing-sorghum-panicles-and-counting-grains-at-an-on-farm-scale)
  - [Summary](#summary)
  - [Contributors](#contributors)
  - [NOTE](#note)
  - [Objective](#objective)
  - [Libraries used](#libraries-used)
  - [Detecting and segmenting](#detecting-and-segmenting)
  - [Counting](#counting)
  - [Estimation to whole panicle](#estimation-to-whole-panicle)
  - [Data availability](#data-availability)
  - [Acknowledgments](#acknowledgments)

---

## Contributors

- Gustavo Nocera Santiago ([@GustavoSantiago113](https://github.com/GustavoSantiago113))
- Pedro Cisdeli ([@cisdei](https://github.com/cisdeli))

---

## NOTE

**The version of the paper is present in the "old-version" branch, the one present in the main is the new and improved version** 

---

## Objective

This study had the main objective to fill the research gap of using imagery obtained from smartphones to detect and segment sorghum panicles and also count the number of grains.It employed convolutional neural networks (CNNs)for effectively removing background under field conditions and counting sorghum grains. Therefore, the specific objectives included i) detect and segment sorghum panicles at field scale using imagery from a smartphone camera, ii) utilize segmented imagery to count the number of grains within manually segmented panicles, and iii) employ another machine learning model combining the panicle image and the counted grains within the image to estimate the total number of grains within a panicle based on the previous counting. Through this innovative approach, we aimed to provide the basis for the development of a mobile app to transfer this knowledge to the farm-level

----

## Libraries used 

The main libraries, frameworks, and tools used in this study includes:

* PyTorch;
* Yolov8;
* OpenCV;
* imgaug;
* Scipy;
* Scikit-learn;
* Pandas;
* Matplotlib;
* Roboflow;
* Labelme;
* numpy;

----

## Detecting and segmenting

The first step is the detection and segmentation of sorghum panicles from smartphone imagery. The images were collected in a sorghum field at Wamego, Kansas-US during the 2022 season using an ordinary smartphone, perpendicularly to the panicles and set on automatic capture mode, against the sunlight.

The images were annotated and augmented using Roboflow software.

We opted to go for a user-guided (for a future mobile application) approach where a rectangle is in the screen and the desired panicle is framed within the rectangle.

To perform the segmentation, different versions of Yolo were tested: Yolov8-n, Yolov8-s, Yolov9-n and Yolov9-s, with Yolov8-n presenting the best performance in less time.

On the file [Segmentation](Segmentation.ipynb) is the complete algorithm were the two frameworks were trained, tested and visually validated.

---

## Counting

To count, 40 images were randomly selected from the original dataset to detect and segment. The main panicle was manually segmented and the background was extracted. The images were then cropped in 3 equal-height patches: the top, the middle, and the bottom of the panicle. Point labels were manually placed in each visible grain using labelme software. The points and the images were then augmented using imgaug library. The augmented label points dataset were converted to density maps using gaussian filter function from scipy library.

Several different architecture models were evaluated in estimating density maps step, including SorghumNet (paper version), and CerealNet (current version model). The models architecture were developed using PyTorch library and can be seen in this [file](DensityMaps.ipynb).

To evaluate the models' perfomance, were used the metrics: Mean Absolute Error (MAE), Main Square Error (MSE), and Kling-Gupta Index (KGI), from the scipy library.

---

## Estimation to whole panicle

Lastly, to estimate the amount of grains from the whole panicle, a different dataset of 198 images of sorghum panicles were obtained during the same day and using the same configurations mentioned before.The panicles were also manually segmented, the background was removed, and the imagery was patched in three again. The images were input into Sorghum-Net model and the total number of grains was estimated as the sum of the three patches.

A deep learning approach combining segmented panicle image and the numerical output of the counting step was used in this step. We tested different models with a regression head in the end and developed another model architecture (GNet). To quantify the accuracy of the model, the scikit-learn library was employed to calculate the following metrics: coefficient of determination (R2), MSE, RMSE, and KGI.

----

## Data availability

The raw data, among the labels are available upon reasonable request to gsantiago@ksu.edu.

---

## Acknowledgments

The authors would like to thank Ciampitti Lab team, specifically to all the research scholars that helped with the data collection and labeling process: Milagros Cobal, Lucas Suguiura, Brian Michiels, Oscar Lanza, Luciano Fello, and Juan Ahunchain. Sorghum Checkoff and Corteva Agriscience for providing support for field trials, investigation, data collection, and labeling process of sorghum imagery. Contribution no. 24-XYZ-J from the Kansas Agricultural Experiment Station.
