# CNN Vs Transformer | Regression Vs Classification
<img src="https://i.pinimg.com/originals/c9/10/ce/c910ce1b0b2aa0bcb1b095e9c90fd536.jpg" width=800>
> This discussion is an extension of this [post](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/275094) by @phalanx 

## Notebook:
* CNN + Regression:
    * train: [[TF] PetFinder: Image [TPU][Train] 🐶](https://www.kaggle.com/awsaf49/tf-petfinder-image-tpu-train)
    * infer: [[TF] PetFinder: Image [TPU][Infer] 🐶](https://www.kaggle.com/awsaf49/tf-petfinder-image-tpu-infer)
* Transformer + Classification:
    * train: [[TF] PetFinder: ViT+Cls [TPU][Train] 😺](https://www.kaggle.com/awsaf49/tf-petfinder-vit-cls-tpu-train/)
    * infer: [[TF] PetFinder: ViT+Cls [TPU][Infer] 😺](https://www.kaggle.com/awsaf49/tf-petfinder-vit-cls-tpu-infer)



## Experiments:
### 1. Regression Vs Classification:
Normalize `Pawpularity` from `[0,100]` range to `[0,1]`. Use `BinaryCrossentropy` loss intead of `RootMeanSquaredError`. Calculate competition metric after **denormalizing** the prediction.
* **Regression**: 
    * Model: **EfficientnetB2**
    * Loss: `RMSE`
    * CV: 19.058
    * LB: 18.766
* **Classification**:
    * Model: **EfficientnetB2**
    * Loss: `BCE`
    * CV: 18.681

**Result** : Classification wins ✅


### 2. CNN Vs Transformer:
Use a Transformer model like **ViT** instead of **CNN** based model **EfficientNet**.
* **CNN**: 
    * Model: **EfficientnetB2**
    * Loss: `BCE`
    * CV: 18.681
* **Transformer**:
    * Model: **Vision-Transformer**
    * Loss: `BCE`
    * CV: 18.035
    * LB: 18.245

**Result** : Transformer wins ✅

## Analysis:
Let's look at the **Grad-CAM** by **CNN** and **Attention-MAP** by **Transformer**.
![](/assets/animation/cvt-smaller.gif)
> It seems that **CNN** is focusing more on facial features due to its `local_receptive_field` whereas **Transformer** is focusing on both facial features and surrounding features due to its `global_receptive_fields`. That could be the reason why **Transformer** is outperforming **CNN**.

**Notes:**
* Classification with `MAE` loss may not perform as good as Classification with `BCE` loss.
* Using **Swin-Transformer** may change the result.
* For **ViT** you can use this amazing repo, https://github.com/faustomorales/vit-keras.


