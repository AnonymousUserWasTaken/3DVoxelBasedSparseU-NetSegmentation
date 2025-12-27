# 3DVoxelBasedSparseU-NetSegmentation (H1)
## I want to create a segmentation method that is capable of multi-instancing humans in real-time + deriving pose patterns from 3D data.


The goal was simple, Create a basic pattern recognition architecture that could detect any bit of training data from arbituary but limited sets. After feeding it only 2 meshes a female and a male mesh as well as a lot of background information (Planes, Shapes, Meshes, anything NOT human) it was able to determine the location of humans in under a second flat 

<img width="1476" height="497" alt="image" src="https://github.com/user-attachments/assets/e95315e4-0b0c-45aa-a464-c8795df5a483" />

I gave it very sparse and few data yet it was even able to detect a human it's never even seen before in training, the posed human (SHOWN BELOW) was never present in any of the training data nor was it even close to relevant size. After doing some Point Sampling (where we take the nearest denser points, I know right...) we are able to accurately determine the most plausible candidates location (as well as a bounding box around the points designated) and print them to the screen in 0.541s. However given that most of the Performance is bogged down by the actual wireframe we're building to PRINT to said screen the time is actually significantly less, I'll know more when I finish the implementation.
---
<img width="1461" height="51" alt="image" src="https://github.com/user-attachments/assets/3009efea-def4-4045-8514-45651902eac1" />

The image clearly shows a finish time of 0.541s as does the code but that doesn't take into account the recorded time it takes to actually build the wireframe. Meaning the actual time that it would take to FORWARD data using a temporal like model frame by frame is 0.204s or 204 ms. Not bad taking into account that we'd be reading point cloud data and looking for plausible candidates frame by frame, this can be signicantly reduced in future implementations.
---

<img width="1192" height="577" alt="image" src="https://github.com/user-attachments/assets/ac94a44f-2322-4bb9-a893-79a05fc1851a" />



Finally had time to push all of my work, Research Notes coming soon...


To work, simply install UPBGE 0.44 or later __(ensure that the python version aligns with the 0.44 version's python 3.11)__

https://upbge.org/#/download

**For Starters I'd like to state that UPBGE may seem like a lack luster and poor environment to run ANY training material, however I've come to learn that with a bit of knowledge with the bpy API and a few small data tweaks with numpy library we can run torch training within this environment... on top of that the coding environment meshes really well with the 3D viewport environment, meaning we can test ALOT of 3D applications as well as 2D in the near future...**

```
Dependencies: once you've done that all you need to do is install a CUDA version of Torch (CPU should work but you may get a few errors depending on the training scripts and have lack luster performance when training), if not training CUDA shouldn't matter

to Test Everything you'll need :
  -  1) Install Numpy consistent with Python Version 3.11, Install Torch, TorchVision consistent with a CUDA version >= 12.x
  -  2) Install a version of UPBGE WITHHHHHHHHHHHHHHHHHH PYTHON VERSION 3.11!!! if not ensure you have a version of numpy and torch that runs well with the inference script4
  -  3) Download the Blender Project to run the code. you can actually just download the blender project itself and run it like that. 
```


Want to try out the weights? 

Run the inference scripts as indicated within textblocks on the blender project on checkout some of the DEMO checkpoints within the link below, training for this repo will probably never be completed nor do I plan on it. This was a simple demo to test out the constraints of an architecture i've been eyeballing for some time now.


<img width="308" height="162" alt="image" src="https://github.com/user-attachments/assets/c4b5b522-8c50-430e-9ecc-d9285678ef5a" /> 

<img width="833" height="992" alt="image" src="https://github.com/user-attachments/assets/161ef6ca-2254-46b4-be68-58ddc9144da8" />

<img width="345" height="141" alt="image" src="https://github.com/user-attachments/assets/6bb0ba7e-e525-40c3-a3fc-09605f949cc1" />

<img width="1583" height="112" alt="image" src="https://github.com/user-attachments/assets/361a0266-9617-47ba-b657-062646887145" />

<img width="651" height="314" alt="image" src="https://github.com/user-attachments/assets/7eba69c0-e1e0-4dd4-b866-35482109d1a2" />

OR Alternatively you can drag the mesh into the geometry nodes

then you can run the inference script left within the scripting window.




Here are a few more demo images

<img width="593" height="446" alt="image" src="https://github.com/user-attachments/assets/417726c1-2582-4b37-9ff3-4bca9b2a0ab1" />

<img width="342" height="419" alt="image" src="https://github.com/user-attachments/assets/3f4bb96b-a8a3-4b64-a275-3b3fa13aed8e" />

<img width="824" height="717" alt="image" src="https://github.com/user-attachments/assets/1d7b3e99-3c1e-4e2b-a9d5-489cce725fb4" />

<img width="834" height="716" alt="image" src="https://github.com/user-attachments/assets/8b86221b-533d-459b-975b-8bd5cdbc5e34" />



****
