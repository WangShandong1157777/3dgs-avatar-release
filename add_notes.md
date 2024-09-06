## install
```bash
conda create -n 3dgs-avatar --clone splatting
cd submodules/diff-gaussian-rasterization
pip install .
cd ../submodules/simple-knn
pip install .
pip install scikit-image
pip install hydra-core==1.3.2
pip install torchmetrics==0.11.4
pip install wandb==0.15.11
wandb login
```

## data prepare
* add 'body_models' folder
* add 'data' folder
* add 'preprocess_dataset' folder



## Training
```python
# PeopleSnapshot
python train.py dataset=ps_female_3 option=iter30k pose_correction=none 
python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ps_female_3
```


## Error solving
* wandb: ERROR Error while calling W&B API: permission denied (<Response [403]>)  
A: wandb.init里面去掉entity, 设置project为空
```python
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='',
        # project='3dgs-avatar-release',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )
```

## 实验记录：
* female-3-casual数据集，作者官方代码，数据加载修改 line85 in dataset/people_snapshot.py，实验结果保存在exp_log/female-3-casual/train-ori.log
```python
            # img_files = img_files[frame_slice]
            # mask_files = mask_files[frame_slice]
            img_files = [os.path.join(subject_dir, f'images/{frame:06d}.png') for frame in frames]
            mask_files = [os.path.join(subject_dir, f'masks/{frame:06d}.png') for frame in frames]
```
* 注释掉adptive-desify-prune,line229 in train.py, 保持高斯数量不变，实验结果保存在exp_log/female-3-casual/train-remove-densify-prune.log
```python
            # # Densification
            # if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            #
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt, scene, size_threshold)
            #
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()
```
* 上一步实验结果基础上，去掉non-rigid模块，修改代码line52 in models/deformer/non_rigid.py
```python
        if iteration < 30000:#self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature", torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}
```