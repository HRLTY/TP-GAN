# TP-GAN

Official TP-GAN Tensorflow implementation for the paper "[Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Beyond_Face_Rotation_ICCV_2017_paper.pdf)" (ICCV17).

Basically, it recovers a frontal face image of the same person from a single face image under any poses.

Here are some examples from the paper.![image](images/ownsynthesis.jpg)

Synthesized images of all poses, frontal illumination in **Testing** Setting 2 in MultiPIE can be obtained [here](). Some of **randomly** selected testing images are shown below.

Synthesized images for other illumination condition and/or training set can be obtained upon request.

Random examples of 10 testing image pairs for each degree:

15 and 30 degrees: 
<p align="center">
<img src="images/30-rand.png", width="300", border="10"><img src="images/30-rand.png", width="300", border="10">
</p> 

45 and 60 degrees:
<p align="center">
<img src="images/45-rand.png", width="300", border="10"><img src="images/60-rand.png", width="304", border="10">
</p> 

75 and 90 degrees:
<p align="center">
<img src="images/75-rand.png", width="300", border="10"><img src="images/90-rand.png", width="306", border="10">
</p> 

This is an initial release of code, which may not be fully tested. Refinement, pre-trained models, and precomputed testing image features will come later.

If you like our work or find our code useful, welcome to cite our paper!

Any suggestion and/or comment would be valuable. Please send an email to Rui, huangrui AT cmu.edu or ruih2 AT cs.cmu.edu or other authors.

      @InProceedings{Huang_2017_ICCV,
      author = {Huang, Rui and Zhang, Shu and Li, Tianyu and He, Ran},
      title = {Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis},
      booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
      month = {Oct},
      year = {2017}
      }


