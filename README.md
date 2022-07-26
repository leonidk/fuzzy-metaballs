# Fuzzy Metaballs Differentiable Renderer
# [Project Page](https://leonidk.github.io/fuzzy-metaballs/)
Here is a simplified version of our code base, without all the specific code to generate our plots, visualizations and run experiments on our custom dataset. Likewise we only feature the linear intersection rule, as it is the main method used in our paper. 

# Shape From Silhouette
This sample builds on the PyTorch3D model reconstruction tutorial, except the method uses only silhouettes and uses Fuzzy Metaballs. Should run in about 1 minute on a laptop.

# Pose Estimation
This sample builds on the PyTorch3D camera pose optimization tutorial, except the method uses depth and silhouette along with pose perturbation. 
