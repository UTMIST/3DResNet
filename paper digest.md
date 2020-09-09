(View in Raw)
Reference:

K. Hara, H. Kataoka and Y. Satoh, "Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition," 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), Venice, 2017, pp. 3154-3160, doi: 10.1109/ICCVW.2017.373.


Related MI Concepts:
1. Neural Network
	a) Overfitting:
		- too many learnable parameters, insufficient amount of training examples, resulting in failure to generalize from the training set
	b) Shallow vs. Deep:
		- adding learnable parameters in depth (deeper) is exponentially more effective than adding learnable parameters in layers (wider), in theory
		- in practice, classic deep neural networks suffer from increasing depth during training due to gradient diminishing, and often have increasing training error
		- ResNets, consisted of Residual Blocks, are specialized deep neural networks that solve the issue of gradient diminishing
2. Convolutional Neural Network
	a) Kernel/Filter:
		- 2D kernel is a 3D tensor, with 2 customizable sides along the height and width, and 1 determined side that has the depth(e.g. 3 for RGB channels; multi-frame optical flow)
		- 3D kernel is a 4D tensor, with 3 customizable sides along the height, width, and “time”, and 1 determined side that has the depth
	b) ResNet:
		- 3D convolution, 3D pooling
		- stride
		- down-sampling with conv3_1, conv4_1, conv5_1
		- identity shortcuts
		- zero-padding
3. NN Action Recognition
	a) (popular) Two-stream 2D ConvNet Approach
		- 1st stream:		RGB channels as depth
		- 2nd stream:		stacked Optical Flow frames as depth
		- multiple papers for further improvement 
		- ResNets have been found effective for Two-stream 2D ConvNet Approach
	b) 3D ConvNets
		- extract spatio-temporal features from videos
		- C3D model with Sports-1M:
        3x3x3 kernel (best performance),			increased temporal length (increased performance)
		- Optical Flow as inputs:
				(better performance than RGB inputs),		combined Optical Flow and RGB (best performance)
				=> 3D ConvNet trained from Kinetics is competitive to pretrained 2D ConvNet(from ImageNet)
				=> 3D ConvNet trained from HMDB51 and UCF101 is, however, inferior to 2D ConvNet
		- Inception Architecture (22 layers):
				=> 3D ConvNet achieved state-of-the-art performance
		- ResNet:
				=> (outperform?)

Interests:
1. Lack of Exploration in Deep 3D ConvNets for Action Recognition
	a) Main Training Difficulty:
		- large learnable parameters
	b) Better Performance than 2D ConvNets:
		- From recent availability of large video datasets
2. Residual 3D ConvNet
	a) Pre-trained Model:
		- available at https://github.com/kenshohara/3D-ResNets-PyTorch 
	b) Specific Datasets for Training and Testing:
		- ActivityNet    (small datasets)
		- Kinetics    (~300,000 examples, 400 categories)
		- HMDB51, UCF101    (successful in 2D ConvNet action recognition, but prone to overfitting with 3D ConvNet)
		- Sports-1M, YouTube-8M    (large database, but noisy)

Architecture:
1. Based on original ResNets
	a) Difference:
		- dimensions (inputs, kernels, pooling)
	b) Inputs:
		- 16 frames RGB clips		(no optical flow*)
	b) Hyperparameters (similar to C3D):
		- stride = 1 in depth (temporal);	stride = 2 in height and width
		- 3x3x3 kernel

Implementation:
1. Training
	a) Stochastic Gradient Descent (SGD):
		- mini-batch size = 256    (4 NVIDIA TITAN X)
		- with Momentum (0.9)
		- Learning Rate (0.1, weight decay = 0.001 once validation loss saturates)
		- Large Batch Size and Learning Rate are important to achieve good performance
	b) Training Set:
		- processing:
			resize to 360-pixel height (keeping aspect ratio)
			uniform sampling of 16 frames (loop if out of range) from each video as the stack of inputs
			mean subtractions
		- data augmentation:		https://arxiv.org/pdf/1507.02159.pdf
			select a spatial position (4 corners, center)
			select spatial scale for multi-scale cropping
			flipping
2. Validating and Testing
	a) Similar to Training Set:
		- each set of 16 images are cropped around a center position with max scale
		- average class probabilities from different sets of the same video

Experiments:
1. Dataset
	a) ActivityNet (v1.3):
		- 200 classes,		137 untrimmed videos per class,		849 hours
		- training: 50%,	validation: 25%,	testing: 25%
	b) Kinetics:
		- 400 classes,		400 trimmed videos per class,		~1000 hours
		- training: 80%,	validation: ~6%,	testing: ~13%
		- 10x activity instance of ActivityNet
2. Results
	a) Early Test on ActivityNet:
		- 18-layer 3D ResNet		=> overfitted
		- compare with pretrained Sports-1M C3D (shallower)	=> better accuracy
	b) Kinetics:
		- 34-layer 3D ResNet		=> does not overfit, competitive performance
		- compare with pretrained Sports-1M C3D		=> underfitted
