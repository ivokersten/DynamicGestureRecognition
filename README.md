# Dynamic Gesture Recognition

This repository contains software to recognize four dynamic hand gestures: 'Horizontal', 'Vertical', 'Clockwise Circle' and 'Counterclockwise Circle' from an input video stream using machine vision. The hand detection algorithm is based on [this](https://github.com/victordibia/handtracking) github repository by Victor Dibia. This software is edited and added upon by me. 

## Files

The files `Drone_Processing.py` and `External_Processing.py` work in conjunction, where `Drone_Processing.py` can run on a low performance computer proving the video stream and `External_Processing.py` can run on a high performance computer to perform the computationally expensive operations and send results back to the low performance computer, as long as both devices are connected to the same Robot Operating System (ROS) network.

The file `Dynamic_Gesture_Recognition.py` combines the above files to run on a powerful computer where the video capture and processing is performed on the same device. ROS is not needed to run this file.


## References
Victor Dibia, HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks, https://github.com/victordibia/handtracking

Apache Licence. See [LICENSE](LICENSE) for details. Copyright (c) 2020 Ivo Kersten.
