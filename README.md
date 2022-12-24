## ProFPN: Progressive Feature Pyramid Network with Soft Proposal Assignment for Object Detection

#### This is the official repo of ProFPN and soft proposal assignment.

![image-20221221180156099](https://user-images.githubusercontent.com/37873318/208881184-0cad72b6-4d46-4ccb-935a-fe3b183f0f6b.png)


#### Updates

$\bullet$ (21/12/2022) Core code is released.

$\bullet$ (22/12/2022) Old version code and inference model is released.
#### Performance

##### $\bullet$ Baseline Performance (CNN and Transformer)

![image-20221221180820219](https://user-images.githubusercontent.com/37873318/208881234-e41f547e-b52a-4a8d-bb3c-49d40f86d97e.png)
##### $\bullet$ SOTA Performance



![image-20221221180850282](https://user-images.githubusercontent.com/37873318/208881275-f30c22a8-c9d4-4f63-a098-4b8f6ef6edb0.png)
#### Model Evaluation

Old version code and model pth uploaded!!! The old version code and pth is paired, which are equivalent to the new version.You can inference based on old version code, I have comment each part in the old version code.

The pth link has been upload to https://pan.baidu.com/s/1HATOK4Wx6swjmnW4hwIK9Q?pwd=pi8e extract_code: pi8e.

Upload Soon!!!

#### Getting Started

The installation instruction and usage are in [MMdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md).
If you want to tran adn evaluate ProFPN, please add profpn.py to mmdet/neck and add register in mmdet/neck/__init__.py. Then replace mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py with our SPA. Finally, rename soft_proposal_assignment.py with single_level_roi_extractor.py.

#### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/GingerCohle/ProFPN/blob/main/LICENSE.md) for details.
