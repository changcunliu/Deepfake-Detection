# Context-Aware Deepfake Detection for Securing AI-Driven Financial Transactions
<b> Authors: Changcun Liu</a>, Guisheng Zhang</a>, Siyou Guo</a>, Qilei Li</a>, Gwanggil Jeon</a>, Mingliang Gao*</a>  </b>


> News:
> OUR LATEST WORK:  We have proposed a network model aimed at addressing the issue of high-risk transactions involving deepfake-generated faces in facial payment systems.  We will soon release all the codes implemented by the DeepfakeBench codebase.


### Training 

<a href="#top">[Back to top]</a>

To run the training code, you should first download the pretrained weights for the corresponding **backbones** (These pre-trained weights are from ImageNet). You can download them from [Link](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip). After downloading, you need to put all the weights files into the folder `./training/pretrained`.

Then, you should go to the `./training/config/detector/` folder and then Choose the detector to be trained. For instance, you can adjust the parameters in [`xception.yaml`](./training/config/detector/xception.yaml) to specify the parameters, *e.g.,* training and testing datasets, epoch, frame_num, *etc*.

After setting the parameters, you can run with the following to train the Xception detector:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml
```

You can also adjust the training and testing datasets using the command line, for example:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml  \
--train_dataset "FF++" \
--test_dataset "Celeb-DF-v1" "Celeb-DF-v2"
```

By default, the checkpoints and features will be saved during the training process. If you do not want to save them, run with the following:

```
python training/train.py \
--detector_path ./training/config/detector/xception.yaml \
--train_dataset "FF++" \
--test_dataset "Celeb-DF-v1" \
--no-save_ckpt \
--no-save_feat
```

### Test

If you want to produce the results, you can use the the [`test.py`](./training/test.py) code for evaluation. Here is an example:

```
python3 training/test.py \
--detector_path ./training/config/detector/.yaml \
--test_dataset "FF++" \
--weights_path ./training/weights/ .pth
```
