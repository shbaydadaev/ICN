# ICN

Impulse Classification Network (ICN) for video Head Impulse Test
--------------------------------------------------------------------

This research proposes the Impulse Classification Network (ICN) using 1D Convolutional Neural Network (1D CNN) that able to detect noisy data and classify human VOR impulses. ICN is a high-performance classification method that works on a patient's video Head Impulse Test (vHIT) impulse data by identifying abnormalities and artifacts. Our ICN method found actual classes of patient’s impulses with 95% accuracy. 


We provide train and test python files. We created our dataset which came from the ICS goggle device.


Lateral canal test type: left side vHIT data with a) normal and b) artifact impulses

a)
![](/images/normal_impulses.png)

b)
![](/images/artifact_impulses.png) 

----------------------------
Four type of classes

1	Normal	1081
2	Abnormal	804
3	Artifact_phase_shift	797
4	Artifact_high_gain	1115

Total	3797
----------------------------
