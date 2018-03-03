# TP-GAN

Official TP-GAN Tensorflow implementation for the ICCV17 paper "[Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Beyond_Face_Rotation_ICCV_2017_paper.pdf)" by Huang, Rui and Zhang, Shu and Li, Tianyu and He, Ran.

The goal is to recover a frontal face image of the same person from a single face image under any poses.

Here are some examples from the paper.![image](images/ownsynthesis.jpg)

### Testing images

Synthesized  testing images of all poses, corresponding illumination in Setting 2 in MultiPIE can be obtained [here](https://drive.google.com/file/d/1Kx0sMjFTzLX3-rZ03TAVBAj-gcd9rJrd/view?usp=sharing). 

Synthesized images for other illumination condition and/or training set can be obtained upon request.

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
<img src="images/75-rand.png", width="300", border="10"><img src="images/90-rand.png", width="300", border="10">
</p> 

### Note

It was initially written in Tensorflow 0.12.

This is an initial release of code, which may not be fully tested. Refinement, pre-trained models, and precomputed testing image features will come later.

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


