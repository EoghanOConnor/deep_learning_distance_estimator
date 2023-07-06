# deep_learning_distance_estimator
Monocular Vision for Distance Estimation Using Deep Learning.  

The following is a Deep Learning distance estimator.  
A CNN is used to predict the distances of pedestrians in a live video.  
The video is recorded using monocular vision (Single camera).  
Instead of dual cameras (Stereo vision) as used in industry for distance estimation.  

## Preprocessing.py
Processes label file info from the KITTI dataset.  
The pedestrian's info is then stored locally.  

## Distance_estimator.py
The trained CNN is imported.  
A video of pedestrians is loaded.  
The predicted distance of the pedestrians is displayed on the output video.  









https://github.com/EoghanOConnor/deep_learning_distance_estimator/assets/45408401/9938aa1e-e946-4ef3-81d3-aafe203e9bc6

