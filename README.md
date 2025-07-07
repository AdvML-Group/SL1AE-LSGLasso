# SL1AE-LSGLasso
[**\[IJCNN 2025\] "Deep Robust Data Reconstruction via Smoothed L1-Autoencoder and Layerwise Sparse Group Lasso", Junyang Zhang, Wen Yang, Di Ming*.**]([https://github.com/advml-group](https://github.com/AdvML-Group/SL1AE-LSGLasso)) 


# Getting Started
## Dependencies
1. Install [Pytorch](https://pytorch.org/). This repo is tested with pytorch=1.13.0, python=3.9.13.  
2. Create an Anaconda Python environment on other computers using the environment.yml file, as follows,  
```
conda env create -f environment.yml
```
The execution will create an environment identical to that during export, including all the dependent packages and their versions.  
Or use Notepad to open the environment.yml file to check the environment version.  

## Dataset
The dataset is located in the "data" folder of the SL1AE_Lasso file, or refer to  
- [ATNT] F. Samaria and A. Harter, "Parameterisation of a stochastic model for" human face identification,” in Winter Conference on Applications of Computer Vision, 1994, pp. 138–142.  
- [AR] A. M. Martínez and A. C. Kak, "PCA versus LDA," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 23, no. 2, pp. 228–233, 2001.   
- [YaleB] A. S. Georghiades, P. N. Belhumeur, and D. J. Kriegman, “From few to many: Illumination cone models for face recognition under variable lighting and pose,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 23, no. 6, pp. 643–660, 2001.  
- [Face_GT] A. V. Nefian and M. H. H. III, “An embedded hmm-based approach for face detection and recognition,” in International Conference on Acoustics, Speech and Signal Processing, 1999, pp. 3553–3556.  
## Training
Run the following command. Train using the SL1AE_Lasso model. You can also modify the hyperparameters or change the model to match the experimental setup in our paper.
```
python run_demo_SL1AE_Lasso.py
```
The other settings for the hyperparameters are defined in the previous run_demo_SL1AE_Lasso.py.
## Testing
Evaluate our model based on the experimental results.
## Citation
```
@InProceedings{IJCNN2025_SL1AE_Lasso,
    author    = {Junyang Zhang, Wen Yang, Di Ming},
    title     = {Deep Robust Data Reconstruction via Smoothed L1-Autoencoder and Layerwise Sparse Group Lasso},
    booktitle = {2025 International Joint Conference on Neural Networks (IJCNN)},
    month     = {June},
    year      = {2025}
}
```
