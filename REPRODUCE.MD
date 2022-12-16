# Reproducing our experiments

In the following, we describe which scripts and notebooks to run in which order to reproduce
the different figures from our paper.
Note that you may have to adjust the `save_dir` parameters in the individual config files (this is where the experimental results get stored).  
You have to manually create these directories, they are not automatically created by the program itself.  
You will also need to adjust the slurm configuration parameters at the top of each file to match your cluster configuration (partition names etc.).

## Main figure (Fig. 1)
Run `plotting/explainy_figure.ipynb`

## Inverse certificates

### Fig. 2, 8
These are the inverse certificate for adversarial scaling.
```
seml invariance_eval_inverse_parallel add seml/configs/eval_inverse_parallel.yaml start
```
Then run `plotting/eval_inverse/eval_inverse_parallel.ipynb`.

### Fig. 3, 4, 9, 10, 11
These are the heatmap figures in which we evaluate the certificate for 2D rotation invariance for various parameter combinations.
```
seml invariance_eval_inverse_multiple_inner_cross add seml/configs/eval_inverse_parallel.yaml start
```
Then run `plotting/eval_inverse/eval_inverse_multiple_inner_cross`.

## Smoothed accuracy (Fig. 5)
Here we evaluate the test set accuracy of the different trained models.
```
seml invariance_train add seml/configs/train/train_pointnet.yaml start
seml invariance_train add seml/configs/train/train_attention.yaml start
seml invariance_train add seml/configs/train/train_dgcnn.yaml start
seml invariance_sample_pred add seml/configs/sample_pred/sample_pred_pointnet.yaml start
seml invariance_sample_pred add seml/configs/sample_pred/sample_pred_attention.yaml start
seml invariance_sample_pred add seml/configs/sample_pred/sample_pred_dgcnn.yaml start
```
Then run `plotting/smoothed_accuracy.ipynb`.

## Certificates applied to point cloud classification datasets
For these experiments, you first need to train the models and take Monte Carlo samples:
```
seml invariance_train add seml/configs/train/train_pointnet.yaml start
seml invariance_train add seml/configs/train/train_attention.yaml start
seml invariance_train add seml/configs/train/train_dgcnn.yaml start
seml invariance_sample_cert add seml/configs/sample_cert/sample_cert_pointnet.yaml start
seml invariance_sample_cert add seml/configs/sample_cert/sample_cert_attention.yaml start
seml invariance_sample_cert add seml/configs/sample_cert/sample_cert_dgcnn.yaml start
```

### Fig. 6, 12
These are the certificates evaluated on adversarial scaling of MNIST.
```
seml invariance_eval_forward_parallel add seml/configs/eval_forward/eval_forward_mnist_parallel.yaml start
```
Then run `plotting/eval_forward/eval_forward_mnist_parallel`.

### Fig. 7a, 13
These are the certificates applied to perturbations with random rotation components on MNIST.
```
seml invariance_eval_forward_rotation_2d add seml/configs/eval_forward/eval_forward_mnist_rotation_baselines.yaml start
seml invariance_eval_forward_rotation_2d add seml/configs/eval_forward/eval_forward_mnist_rotation_tight.yaml start
```
Then run `plotting/eval_forward/eval_forward_rotation_2d.ipynb`.

### Fig. 7b, 14, 15, 16
These are the certificates applied to perturbations with random rotation components on ModelNet40.
```
seml invariance_eval_forward_rotation_2d add seml/configs/eval_forward/eval_forward_modelnet_rotation_arbitrary_3d_baselines.yaml start
seml invariance_eval_forward_rotation_2d add seml/configs/eval_forward/eval_forward_modelnet_rotation_arbitrary_3d_tight.yaml start
```
Then run `plotting/eval_forward_rotation_3d.ipynb`.