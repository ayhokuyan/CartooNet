# CartooNet 
## CS484-555 Introduction to Computer Vision Project - Bilkent University
- An improvement project on the [White-Box Cartoonization Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.pdf). 
- See the [original repo](https://github.com/SystemErrorWang/White-box-Cartoonization) for the TensorFlow implementation. For ours, we have used the [equivalent repo](https://github.com/zhen8838/AnimeStylized) that implements in PyTorch. For project completeness, we explained how to code is run but you can also check out the [source repository](https://github.com/zhen8838/AnimeStylized). 
 
 ---
 
### Dataset
The dataset is the same dataset that both repositories have used, which can be downloaded from [here](https://drive.google.com/file/d/10SGv_kbYhVLIC2hLlz2GBkHGAo0nec-3/view?usp=sharing). Download dataset and unzip inside the repository. 
The dataset consists of four folders
 - **face_photo**
	 - 10000 RGB face photos of 256x256 resolution.
 - **face_cartoon**
	 - pa_face and kyoto_face each contain 5000 RGB cartoon face images. We use the pa_face for our implementations.
 - **scenery_photo**
	 - 5000 RGB scenery photos of 256x256 resolution.
 - **scenery_cartoon**
	 - Has three folders of 5000 scenery cartoon images of 356x256 resolution each. Each folder's name represents the artists the images are obtained from namely: shinkai (Makoto Shinkai), Miyozaki Hayao (hayao) and Mamoru Hosoda (hosoda). For our implementation, we use shinkai style image transfer. 

---

### Pretrained VGG-19 Model
This implementation uses a pretrained VGG19 model. Download the model from [here](https://drive.google.com/file/d/1wOBtQmcs6SdyEoAu-mrHdqPRBe3gfsJ2/view?usp=sharing) and unzip under a folder called `models` with the name `vgg19.npy`.

---
### Environment
We have used a conda environment to build the prooject with the following commands.

    conda create -n torch python=3.8
    conda activate torch
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
    pip install pytorch-lightning==1.0.2 opencv-python matplotlib joblib scikit-image torchsummary webdataset albumentations more_itertools
---
### Models 
There are three models that can be run with this repo. Note that all of these models took around 36 hours to train. 

 #### **whitebox**
 
This is the original architecture of whitebox cartoonization. 

**Pretraining**

 `make train CODE=scripts/whiteboxgan_pretrain.py CFG=configs/whitebox_pretrain.yaml`
 
**Training**

Make sure the `pre_trained_ckpt` in `configs/whitebox.yaml` points to the correct .ckpt file e.g. `logs/whitebox_pre/version_0/checkpoints/epoch=1.ckpt`

For training, run:
 `make train CODE=scripts/whiteboxgan.py CFG=configs/whitebox.yaml`
 
To continue from a previous checkpoint run:

 `make train CODE=scripts/whiteboxgan.py CFG=configs/whitebox.yaml CKPT=[path of checkpoint]`
 
 To train silently within a Linux server run:
 
 `nohup make train CODE=scripts/whiteboxgan.py CFG=configs/whitebox.yaml > whitebox.out`
 
 **Inference**
 
 To infer single image, run:
 
 `make infer CODE=scripts/whiteboxgan.py CKPT=[checkpoint path] EXTRA=image_path:[image path]`
 
 where `checkpoint path` is the path of the .ckpt file you want its weights e.g. 
`logs/whitebox/version_3/checkpoints/epoch=19.ckpt` and image path is the path of the image. 

 To infer all images in a directory, run:
 
`make infer CODE=scripts/whiteboxgan.py CKPT=[checkpoint path] EXTRA=image_path:[image dir]` 

The only difference is the input of the directory. 

---

#### **whiteboxirb**
 
This is the first iteration of improvement where we have changed the four ResNet blocks in the middle section of the UNet generator, and replaced them with four Invertible Resnet Blocks (IRBs), which uses depthwise separable convolution for training.

**Pretraining**

 `make train CODE=scripts/whiteboxirbgan_pretrain.py CFG=configs/whiteboxirb_pretrain.yaml`

**Training**

Make sure the `pre_trained_ckpt` in `configs/whiteboxirb.yaml` points to the correct .ckpt file e.g. `logs/whiteboxirb_pre/version_0/checkpoints/epoch=1.ckpt`

For training, run:

 `make train CODE=scripts/whiteboxirbgan.py CFG=configs/whiteboxirb.yaml`
 
To continue from a previous checkpoint run:

 `make train CODE=scripts/whiteboxirbgan.py CFG=configs/whiteboxirb.yaml CKPT=[path of checkpoint]`
 
 To train silently within a Linux server run:
 
 `nohup make train CODE=scripts/whiteboxirbgan.py CFG=configs/whiteboxirb.yaml > irb.out`
 
 **Inference**
 
 To infer single image, run:
 
  `make infer CODE=scripts/whiteboxirbgan.py CKPT=[checkpoint path] EXTRA=image_path:[image path]`
  
where `checkpoint path` is the path of the .ckpt file you want its weights e.g. `logs/whiteboxirb/version_3/checkpoints/epoch=19.ckpt` and image path is the path of the image. 

To infer all images in a directory, run:
`make infer CODE=scripts/whiteboxirbgan.py CKPT=[checkpoint path] EXTRA=image_path:[image dir]` 

---

#### **whiteboxirgf**

This is the second iteration of improvement where we removed the guided filters for training process and increased the `eps` value of guided filter in inference to remove the color disruptions that appear on the original paper. 

**Pretraining**

`make train CODE=scripts/whiteboxirgfgan_pretrain.py 
CFG=configs/whiteboxirgf_pretrain.yaml`

 **This pretrain is identical to the pretrain of `whiteboxirb`, so you can use that pretrain checkpoint, if you have trained a `whiteboxirb_pre`.**  
 
**Training**

Make sure the `pre_trained_ckpt` in `configs/whiteboxirgf.yaml` points to the correct .ckpt file e.g. `logs/whiteboxirgf_pre/version_0/checkpoints/epoch=1.ckpt` or `logs/whiteboxirb_pre/version_0/checkpoints/epoch=1.ckpt`

For training, run:

 `make train CODE=scripts/whiteboxirgfgan.py CFG=configs/whiteboxirgf.yaml`
 
To continue from a previous checkpoint run:

 `make train CODE=scripts/whiteboxirgfgan.py CFG=configs/whiteboxgf.yaml CKPT=[path of checkpoint]`
 
 To train silently within a Linux server run:
 
 `nohup make train CODE=scripts/whiteboxirgfgan.py CFG=configs/whiteboxirgf.yaml > irgf.out`
 
 **Inference**
 
 To infer single image, run:
 
 `make infer CODE=scripts/whiteboxirgfgan.py CKPT=[checkpoint path] EXTRA=image_path:[image path]`
 
where `checkpoint path` is the path of the .ckpt file you want its weights e.g. `logs/whiteboxirgf/version_3/checkpoints/epoch=19.ckpt` and image path is the path of the image.
 
To infer all images in a directory, run:

`make infer CODE=scripts/whiteboxirgfgan.py CKPT=[checkpoint path] EXTRA=image_path:[image dir]` 

---

### Removal of Artifacts
This project aims to remove the visible artifacts that are observed on the original implementation. The success of the algorithm can be observed in the following sample ROIs and the outputs of the models. first column is the sample ROI, the following are whitebox, whitebox_irb and whitebox_irgf respectively.

![](assets/extra/artsamp1.png)
![](assets/extra/artsamp2.png)

---

### Pretrained Networks
You can download the pretrained checkpoints from the links below <\br>

 - [whitebox](https://drive.google.com/file/d/15SbDXGcLRbjcbhvzeiFwYcmJBsZZLhN2/view?usp=sharing) 
 - [whitebox_irb](https://drive.google.com/file/d/11jS22QLH2Sr7-gIpalRq2xdJLTgtHgxy/view?usp=sharing)
 - [whitebox_irgf](https://drive.google.com/file/d/1_ZEs891_xNcmZxtPGd-B4P5EDyqvsOJJ/view?usp=sharing)


---

### Change Face and Scenery Style

To change the cartoon datasets to obtian different styles of any model change the `scene_style` and `face_style` in the `configs/XXX.yaml` and `configs/XXX_pre.yaml` files where `XXX` is the model name.


### Sample Outputs

These sample inputs are taken from the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) in high resolution. Click on the image to see it full-size.

| Original | Whitebox | IRB | IRGF |
|----------|----------|-----|------|
|![](assets/org/0677.png)|![](assets/wb/0677_out.png)|![](assets/irb/0677_out.png)|![](assets/irgf/0677_out.png)|
|![](assets/org/0461.png)|![](assets/wb/0461_out.png)|![](assets/irb/0461_out.png)|![](assets/irgf/0461_out.png)|
|![](assets/org/0649.png)|![](assets/wb/0649_out.png)|![](assets/irb/0649_out.png)|![](assets/irgf/0649_out.png)|
|![](assets/org/0712.png)|![](assets/wb/0712_out.png)|![](assets/irb/0712_out.png)|![](assets/irgf/0712_out.png)|
|![](assets/org/0667.png)|![](assets/wb/0667_out.png)|![](assets/irb/0667_out.png)|![](assets/irgf/0667_out.png)|
|![](assets/org/0713.png)|![](assets/wb/0713_out.png)|![](assets/irb/0713_out.png)|![](assets/irgf/0713_out.png)|
|![](assets/org/0774.png)|![](assets/wb/0774_out.png)|![](assets/irb/0774_out.png)|![](assets/irgf/0774_out.png)|
|![](assets/org/0013.png)|![](assets/wb/0013_out.png)|![](assets/irb/0013_out.png)|![](assets/irgf/0013_out.png)|
|![](assets/org/0010.png)|![](assets/wb/0010_out.png)|![](assets/irb/0010_out.png)|![](assets/irgf/0010_out.png)|
|![](assets/org/0776.png)|![](assets/wb/0776_out.png)|![](assets/irb/0776_out.png)|![](assets/irgf/0776_out.png)|
|![](assets/org/0028.png)|![](assets/wb/0028_out.png)|![](assets/irb/0028_out.png)|![](assets/irgf/0028_out.png)|
|![](assets/org/0014.png)|![](assets/wb/0014_out.png)|![](assets/irb/0014_out.png)|![](assets/irgf/0014_out.png)|
|![](assets/org/0767.png)|![](assets/wb/0767_out.png)|![](assets/irb/0767_out.png)|![](assets/irgf/0767_out.png)|
|![](assets/org/0017.png)|![](assets/wb/0017_out.png)|![](assets/irb/0017_out.png)|![](assets/irgf/0017_out.png)|
|![](assets/org/0407.png)|![](assets/wb/0407_out.png)|![](assets/irb/0407_out.png)|![](assets/irgf/0407_out.png)|
|![](assets/org/0374.png)|![](assets/wb/0374_out.png)|![](assets/irb/0374_out.png)|![](assets/irgf/0374_out.png)|
|![](assets/org/0610.png)|![](assets/wb/0610_out.png)|![](assets/irb/0610_out.png)|![](assets/irgf/0610_out.png)|
|![](assets/org/0764.png)|![](assets/wb/0764_out.png)|![](assets/irb/0764_out.png)|![](assets/irgf/0764_out.png)|
|![](assets/org/0002.png)|![](assets/wb/0002_out.png)|![](assets/irb/0002_out.png)|![](assets/irgf/0002_out.png)|
|![](assets/org/0758.png)|![](assets/wb/0758_out.png)|![](assets/irb/0758_out.png)|![](assets/irgf/0758_out.png)|
|![](assets/org/10.jpg)|![](assets/wb/10_out.jpg)|![](assets/irb/10_out.jpg)|![](assets/irgf/10_out.jpg)|
|![](assets/org/0543.png)|![](assets/wb/0543_out.png)|![](assets/irb/0543_out.png)|![](assets/irgf/0543_out.png)|
|![](assets/org/0768.png)|![](assets/wb/0768_out.png)|![](assets/irb/0768_out.png)|![](assets/irgf/0768_out.png)|
|![](assets/org/0018.png)|![](assets/wb/0018_out.png)|![](assets/irb/0018_out.png)|![](assets/irgf/0018_out.png)|
|![](assets/org/0420.png)|![](assets/wb/0420_out.png)|![](assets/irb/0420_out.png)|![](assets/irgf/0420_out.png)|
|![](assets/org/0743.png)|![](assets/wb/0743_out.png)|![](assets/irb/0743_out.png)|![](assets/irgf/0743_out.png)|
|![](assets/org/0794.png)|![](assets/wb/0794_out.png)|![](assets/irb/0794_out.png)|![](assets/irgf/0794_out.png)|
|![](assets/org/23.jpg)|![](assets/wb/23_out.jpg)|![](assets/irb/23_out.jpg)|![](assets/irgf/23_out.jpg)|
|![](assets/org/0596.png)|![](assets/wb/0596_out.png)|![](assets/irb/0596_out.png)|![](assets/irgf/0596_out.png)|
|![](assets/org/0223.png)|![](assets/wb/0223_out.png)|![](assets/irb/0223_out.png)|![](assets/irgf/0223_out.png)|
|![](assets/org/0380.png)|![](assets/wb/0380_out.png)|![](assets/irb/0380_out.png)|![](assets/irgf/0380_out.png)|
|![](assets/org/AE86.png)|![](assets/wb/AE86_out.png)|![](assets/irb/AE86_out.png)|![](assets/irgf/AE86_out.png)|
|![](assets/org/0034.png)|![](assets/wb/0034_out.png)|![](assets/irb/0034_out.png)|![](assets/irgf/0034_out.png)|
|![](assets/org/0791.png)|![](assets/wb/0791_out.png)|![](assets/irb/0791_out.png)|![](assets/irgf/0791_out.png)|
|![](assets/org/0578.png)|![](assets/wb/0578_out.png)|![](assets/irb/0578_out.png)|![](assets/irgf/0578_out.png)|
|![](assets/org/0744.png)|![](assets/wb/0744_out.png)|![](assets/irb/0744_out.png)|![](assets/irgf/0744_out.png)|
|![](assets/org/0619.png)|![](assets/wb/0619_out.png)|![](assets/irb/0619_out.png)|![](assets/irgf/0619_out.png)|
|![](assets/org/0786.png)|![](assets/wb/0786_out.png)|![](assets/irb/0786_out.png)|![](assets/irgf/0786_out.png)|
|![](assets/org/25.jpg)|![](assets/wb/25_out.jpg)|![](assets/irb/25_out.jpg)|![](assets/irgf/25_out.jpg)|
|![](assets/org/0695.png)|![](assets/wb/0695_out.png)|![](assets/irb/0695_out.png)|![](assets/irgf/0695_out.png)|
|![](assets/org/0455.png)|![](assets/wb/0455_out.png)|![](assets/irb/0455_out.png)|![](assets/irgf/0455_out.png)|
|![](assets/org/0286.png)|![](assets/wb/0286_out.png)|![](assets/irb/0286_out.png)|![](assets/irgf/0286_out.png)|
|![](assets/org/0735.png)|![](assets/wb/0735_out.png)|![](assets/irb/0735_out.png)|![](assets/irgf/0735_out.png)|
|![](assets/org/0442.png)|![](assets/wb/0442_out.png)|![](assets/irb/0442_out.png)|![](assets/irgf/0442_out.png)|
|![](assets/org/0734.png)|![](assets/wb/0734_out.png)|![](assets/irb/0734_out.png)|![](assets/irgf/0734_out.png)|
|![](assets/org/0321.png)|![](assets/wb/0321_out.png)|![](assets/irb/0321_out.png)|![](assets/irgf/0321_out.png)|
|![](assets/org/1.jpg)|![](assets/wb/1_out.jpg)|![](assets/irb/1_out.jpg)|![](assets/irgf/1_out.jpg)|
|![](assets/org/0684.png)|![](assets/wb/0684_out.png)|![](assets/irb/0684_out.png)|![](assets/irgf/0684_out.png)|
