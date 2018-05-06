# CJS-CNN

This is an open-source project for predicting aesthetic score distribution through cumulative jensen-shannon divergence based on the deep learning-caffe framework, which we completed in the [Victory team of Besti](http://kislab.besti.edu.cn/victory/).


In this paper we investigate the image aesthetics histogram regression problem, aka, automatically predicting an image into aesthetic score histogram, which is quite a challenging problem beyond image recognition. Deep convolutional neural network (DCNN) methods have recently shown promising results for image aesthetics assessment.  Conventional DCNNs which aim to minimize the difference between the predicted scalar numbers or vectors and the ground truth cannot be directly used for the ordinal basic rating distribution. Thus, a novel CNN based on the Cumulative distribution with Jensen-Shannon divergence (CJS-CNN) is presented to predict the aesthetic score distribution of human ratings, with a new reliability-sensitive learning method based on the kurtosis of the score distribution, which eliminates the requirement of the original full data of human ratings (without normalization). Experimental results on large scale aesthetic dataset demonstrate the effectiveness of our introduced CJS-CNN in this task.


**The way of train and test**

If you want to train in AVA dataset, please copy all source code to include and src path and rebuild caffe.

If you want to test your photos, please refer to caffe official tutorial codes.


**Our paper**

Xin Jin, Le Wu, Xiaodong Li, Siyu Chen, Siwei Peng, Jingying Chi, Shiming Ge, Chenggen Song, Geng Zhao. 

**Predicting Aesthetic Score Distribution through Cumulative Jensen-Shannon Divergence.**

Proceedings of the 32th international conference of the America Association for Artificial Intelligence(**AAAI**), 

New Orleans, Louisiana, February 2-7, 2018, 2017.






