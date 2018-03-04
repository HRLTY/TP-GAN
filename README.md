# TP-GAN

Official TP-GAN Tensorflow implementation for the ICCV17 paper "[Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Beyond_Face_Rotation_ICCV_2017_paper.pdf)" by Huang, Rui and Zhang, Shu and Li, Tianyu and He, Ran.

The goal is to **recover a frontal face image of the same person from a single face image under any poses**.

Here are some examples from the paper.![image](images/ownsynthesis.jpg)

### Testing images

Synthesized  testing images of all poses, corresponding illumination in Setting 2 (and its cropped input) in MultiPIE can be obtained here [Google Drive](https://drive.google.com/file/d/1Kx0sMjFTzLX3-rZ03TAVBAj-gcd9rJrd/view?usp=sharing). 

Synthesized images for other illumination condition and/or training set can be obtained upon request. If you would like to access the original MultiPIE dataset, please contact [MultiPIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html).

### Random examples

Here are random examples of 10 testing image pairs for each degree.

15 and 30 degrees: 
<p align="center">
<img src="images/15-rand.png", width="300", border="10">   <img src="images/30-rand.png", width="300", border="10">
</p> 

45 and 60 degrees:
<p align="center">
<img src="images/45-rand.png", width="300", border="10">   <img src="images/60-rand.png", width="304", border="10">
</p> 

75 and 90 degrees:
<p align="center">
<img src="images/75-rand.png", width="300", border="10">   <img src="images/90-rand.png", width="306", border="10">
</p> 

### Note

It was initially written in Tensorflow 0.12.

This is an initial release of code, which may not be fully tested. Refinement, input data example, pre-trained models, and precomputed testing image features will come later.

The input is cropped with the Matlab script `face_db_align_single_custom.m`, which accepts 5 keypoints and outputs a cropped image and transformed keypoints.

Some example cropping outputs is shown in folder `data-example`.

### Citation and Contact

If you like our work or find our code useful, welcome to cite our paper!

Any suggestion and/or comment would be valuable. Please send an email to Rui at huangrui@cmu.edu or other authors.

      @InProceedings{Huang_2017_ICCV,
      author = {Huang, Rui and Zhang, Shu and Li, Tianyu and He, Ran},
      title = {Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      month = {Oct},
      year = {2017}
      }


