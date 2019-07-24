# TP-GAN

data中5pt文件为人脸关键点部位,用mtcnn模型识别,五行分别是左右眼嘴鼻的x和y坐标,代码能在tensorflow0.12版本下跑,不想修改兼容其他版本了
tensorflow0.12要求CUDA版本Cuda 8.0 and CudNN 5.1?
mtcnn的模型不能在tf0.12下跑,用5pt文件做下中转
没用到face_db_align_single_custom.m
