"""
车牌四边形检测模型训练代码

模型架构：
    输入图像 (B, H, W, 3)
    ↓
    Backbone: EfficientNetV2B0
    ├── C3 (stride=8): block3b_add
    ├── C4 (stride=16): block5e_add
    └── C5 (stride=32): block6h_add
    ↓
    Neck: CSPPAnet (PANet变体)
    ├── SPPF模块 (在C5上应用)
    ├── 上采样路径 (Top-Down): C5→C4→C3
    └── 下采样路径 (Bottom-Up): C3→C4→C5
    输出: P3, P4, P5 (多尺度特征)
    ↓
    Head: 三个检测头 (每个特征层都有)
    ├── P1: 边界框回归头 (输出: 4 * reg_max = 512维)
    ├── P2: 分类头 (输出: 3类)
    └── P3: 关键点预测头 (输出: 8维，4个角点的x,y坐标)
    ↓
    最终输出: [分类(3) + 边界框(512) + 关键点(8)]

关键特点：
    - 无锚框（Anchor-free）：基于中心点检测
    - 分布焦点损失（DFL）：用于边界框回归
    - 四边形检测：直接预测4个角点坐标 (key1x,key1y, key2x,key2y, key3x,key3y, key4x,key4y)
    - 多尺度融合：PANet结构增强特征表达
    - TAL (Task Alignment Learning): 正负样本分配策略
"""

import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# 指定GPU设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf_record_filename_list = (
                           "plate_detect_v1.tfrecords",
                           "plate_detect_v2.tfrecords", 
                           "plate_detect_v3.tfrecords", 
                           "plate_detect_v4.tfrecords",
                           "plate_detect_v5.tfrecords",
                           "plate_detect_v6.tfrecords",
                           "plate_detect_v7.tfrecords",
                           "plate_detect_label-31-CRPD-double-obj-people-clear.tfrecords",
                           "plate_detect_v8.tfrecords",
                           "plate_detect_v9.tfrecords",
                           "plate_detect_v10.tfrecords",
                           "plate_detect_v11.tfrecords",
                            "plate_detect_label-32-CRPD-double-obj-people-clear.tfrecords",
                           "plate_detect_v12.tfrecords",
                            "plate_detect_double_v13.tfrecords",
                            "plate_detect_single_yellow_v14.tfrecords",
                            "plate_detect_v15.tfrecords",
                            "plate_detect_v21_multiobj.tfrecords",
                            "plate_detect_v22_multiobj.tfrecords",
                            "plate_detect_label-34-black.tfrecords",
                            "plate_detect_label-35-special.tfrecords",
                            "plate_detect_label-33-CRPD-double-obj-people-clear.tfrecords",
                            "plate_detect_label-37-real-2025-9-17.tfrecords",
                            "plate_detect_label-38-real-single-yellow-2025-9-17.tfrecords",
                            "plate_detect_label-39-real-2025-9-18.tfrecords",
                            "plate_detect_label-40-real-2025-9-19.tfrecords",
                            "plate_detect_label-41-real-2025-9-20.tfrecords",
                            "plate_detect_label-42-real-2025-9-22.tfrecords",
                          )
obj_dataset = tf.data.TFRecordDataset(tf_record_filename_list)

# TFRecord数据格式描述
# 包含图像、边界框和4个关键点（四边形角点）的标注信息
obj_feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], dtype = tf.string),  # JPEG编码的图像
    "image/format": tf.io.FixedLenFeature([], dtype = tf.string),
    "image/height": tf.io.FixedLenFeature([], dtype = tf.int64),
    "image/width": tf.io.FixedLenFeature([], dtype = tf.int64),
    'image/object/class/text': tf.io.FixedLenSequenceFeature([], dtype = tf.string, allow_missing=True),
    'image/object/class/id': tf.io.FixedLenSequenceFeature([], dtype = tf.int64, allow_missing=True),
    # 边界框坐标 (归一化)
    'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    # 4个关键点坐标（四边形角点，归一化）
    # key1: 左上角, key2: 右上角, key3: 右下角, key4: 左下角
    'image/object/keypoint/key1x': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/keypoint/key1y': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/keypoint/key2x': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/keypoint/key2y': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/keypoint/key3x': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/keypoint/key3y': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/keypoint/key4x': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
    'image/object/keypoint/key4y': tf.io.FixedLenSequenceFeature([], dtype = tf.float32, allow_missing=True),
}

def _parse_obj_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, obj_feature_description)

rng = tf.random.Generator.from_seed(123, alg='philox')

def preprocess_data(obj_features):
    """
    数据预处理函数
    
    功能：
        1. 解码JPEG图像
        2. 随机多尺度训练（320x320 到 704x704，步长32）
        3. 保持宽高比resize并padding到目标尺寸
        4. 同步调整边界框和4个关键点坐标
    
    返回：
        padded_image: 预处理后的图像 (H, W, 3)
        bbox: 边界框和关键点 [xmin, ymin, xmax, ymax, key1x, key1y, key2x, key2y, key3x, key3y, key4x, key4y]
        class_id: 类别ID
    """
    
    image_str = obj_features['image/encoded']
    image_height = obj_features['image/height']
    image_width  = obj_features['image/width']
    
    # 边界框坐标（归一化）
    xmin         = obj_features['image/object/bbox/xmin']
    ymin         = obj_features['image/object/bbox/ymin']
    xmax         = obj_features['image/object/bbox/xmax']
    ymax         = obj_features['image/object/bbox/ymax']
    # 4个关键点坐标（四边形角点，归一化）
    key1x        = obj_features['image/object/keypoint/key1x']
    key1y        = obj_features['image/object/keypoint/key1y']
    key2x        = obj_features['image/object/keypoint/key2x']
    key2y        = obj_features['image/object/keypoint/key2y']
    key3x        = obj_features['image/object/keypoint/key3x']
    key3y        = obj_features['image/object/keypoint/key3y']
    key4x        = obj_features['image/object/keypoint/key4x']
    key4y        = obj_features['image/object/keypoint/key4y']
    
    class_name   = obj_features['image/object/class/text']
    class_id     = obj_features['image/object/class/id']
    
    image = tf.io.decode_jpeg(contents = image_str, channels = 3)
    
    image = tf.cast(image, dtype = tf.float32)
    
    image = tf.reshape(image, (image_height, image_width, 3))
    
    class_id     = tf.cast(class_id, dtype = tf.int32)


    # 随机采样，用于多尺度训练
    sample = rng.uniform(shape = (1,), minval = 0, maxval = 1, name = "random_resize", dtype = tf.float32)

    image_height_side  = tf.cast(image_height, dtype = tf.float32)
    image_width_side   = tf.cast(image_width,  dtype = tf.float32)

    # 多尺度训练：随机选择目标尺寸 (320x320 到 704x704，步长32)
    # 用于增强模型对不同尺寸目标的鲁棒性
    padding_image_min_side = tf.constant(32 * 10, dtype = tf.int32)  # 默认320
    padding_image_max_side = tf.constant(32 * 10 , dtype = tf.int32)  # 默认320
    
    if sample < 0.05:
        padding_image_min_side = tf.constant(32 * 10, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 10 * 1, dtype = tf.int32)
    if sample >= 0.05 and sample < 0.1:
        padding_image_min_side = tf.constant(32 * 11, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 11 * 1, dtype = tf.int32)
    elif sample >= 0.1 and sample < 0.15:
        padding_image_min_side = tf.constant(32 * 12, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 12 * 1, dtype = tf.int32)
    elif sample >= 0.15 and sample < 0.2:
        padding_image_min_side = tf.constant(32 * 13, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 13 * 1, dtype = tf.int32)
    elif sample >= 0.2 and sample < 0.25:
        padding_image_min_side = tf.constant(32 * 14, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 14 * 1, dtype = tf.int32)
    elif sample >= 0.25 and sample < 0.3:
        padding_image_min_side = tf.constant(32 * 15, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 15 * 1, dtype = tf.int32)
    elif sample >= 0.3 and sample < 0.35:
        padding_image_min_side = tf.constant(32 * 16, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 16 * 1, dtype = tf.int32)
    elif sample >= 0.35 and sample < 0.4:
        padding_image_min_side = tf.constant(32 * 17, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 17 * 1, dtype = tf.int32)        
    elif sample >= 0.45 and sample < 0.5:
        padding_image_min_side = tf.constant(32 * 18, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 18 * 1, dtype = tf.int32)        
    elif sample >= 0.5 and sample < 0.55:
        padding_image_min_side = tf.constant(32 * 19, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 19 * 1, dtype = tf.int32) 
    elif sample >= 0.55 and sample < 0.7:    
        padding_image_min_side = tf.constant(32 * 20, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 20 * 1, dtype = tf.int32) 
    elif sample >= 0.7 and sample < 0.75:    
        padding_image_min_side = tf.constant(32 * 21, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 21 * 1, dtype = tf.int32) 
    elif sample >= 0.75 and sample < 0.8:    
        padding_image_min_side = tf.constant(32 * 21, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 21 * 1, dtype = tf.int32) 
    elif sample >= 0.8 and sample < 0.85:    
        padding_image_min_side = tf.constant(32 * 21, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 21 * 1, dtype = tf.int32) 
    elif sample >= 0.85 and sample < 0.95:    
        padding_image_min_side = tf.constant(32 * 22, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 22 * 1, dtype = tf.int32) 
    else:   
        padding_image_min_side = tf.constant(32 * 22, dtype = tf.int32)
        padding_image_max_side = tf.constant(32 * 22 * 1, dtype = tf.int32) 


    target_width  = tf.cast(padding_image_max_side, dtype = tf.int32)
    target_height = tf.cast(padding_image_min_side, dtype = tf.int32)
    original_height = tf.cast(image_height_side, dtype = tf.float32)
    original_width  = tf.cast(image_width_side,  dtype = tf.float32)
        
    scale = tf.minimum(tf.cast(target_width, dtype = tf.float32) / original_width, tf.cast(target_height, dtype = tf.float32) / original_height)

    # Resize image while preserving aspect ratio
    new_width = tf.cast(original_width * scale, tf.int32)
    new_height = tf.cast(original_height * scale, tf.int32)
    resized_image = tf.image.resize(image, [new_height, new_width])

    # Calculate padding
    pad_height = target_height - new_height
    pad_width = target_width - new_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # 应用padding（零填充）
    padded_image = tf.pad(resized_image, 
                          [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], 
                          mode='CONSTANT', 
                          constant_values=0)

    # 将归一化的边界框坐标转换为绝对坐标（考虑resize和padding）
    xmin    = xmin  * image_width_side  * scale + tf.cast(pad_left, dtype = tf.float32)
    ymin    = ymin  * image_height_side * scale + tf.cast(pad_top,  dtype = tf.float32) 
    
    xmax    = xmax  * image_width_side  * scale + tf.cast(pad_left, dtype = tf.float32)
    ymax    = ymax  * image_height_side * scale + tf.cast(pad_top,  dtype = tf.float32)

    # 将归一化的关键点坐标转换为绝对坐标（考虑resize和padding）
    # 4个关键点：key1(左上), key2(右上), key3(右下), key4(左下)
    key1x   = key1x * image_width_side  * scale + tf.cast(pad_left, dtype = tf.float32)
    key1y   = key1y * image_height_side  * scale + tf.cast(pad_top,  dtype = tf.float32) 

    key2x   = key2x * image_width_side  * scale + tf.cast(pad_left, dtype = tf.float32)
    key2y   = key2y * image_height_side  * scale + tf.cast(pad_top,  dtype = tf.float32) 

    key3x   = key3x * image_width_side  * scale + tf.cast(pad_left, dtype = tf.float32)
    key3y   = key3y * image_height_side  * scale + tf.cast(pad_top,  dtype = tf.float32) 

    key4x   = key4x * image_width_side  * scale + tf.cast(pad_left, dtype = tf.float32)
    key4y   = key4y * image_height_side  * scale + tf.cast(pad_top,  dtype = tf.float32) 
    

    bbox = tf.stack([
            xmin,
            ymin,
            xmax,
            ymax,
            key1x, 
            key1y,
            key2x,
            key2y,
            key3x,
            key3y,
            key4x,
            key4y
        ], axis = -1)

    print("bbox.shape", bbox.shape)
    
    
    #随机亮度变换
    # if sample >= 0.5 and sample < 0.6:
    #     image = tf.image.random_brightness(image, max_delta=0.1)
        
    padded_image = tf.clip_by_value(padded_image, 0, 255)
        
    return padded_image, bbox, class_id

def batch_encode(batch_images, gts_box, gts_cls):
    
    batch_size = tf.shape(batch_images)[0]
    
    labels = tf.TensorArray(dtype=tf.float32, size = batch_size, dynamic_size = True)
        
    for i in range(batch_size):
        
        gt_box = gts_box[i]
        gt_cls = gts_cls[i]
        
        images = batch_images[i]
        
        #转换图片文件，TODO
        
        gt_cls = tf.cast(gt_cls, dtype = tf.float32)
        gt_cls = tf.expand_dims(gt_cls, axis = -1)
        label  = tf.concat([gt_box, gt_cls], axis = -1)
        
        labels = labels.write(i, label)
            
    return (batch_images, labels.stack())
    

train_parsed_obj_dataset = obj_dataset.map(_parse_obj_function, tf.data.AUTOTUNE)

#缓存dataset, 避免文件读取
# train_parsed_obj_dataset = train_parsed_obj_dataset.cache()

train_parsed_obj_dataset = train_parsed_obj_dataset.map(preprocess_data, tf.data.AUTOTUNE)

#总的数据集大小
batch_size    = 16
train_dataset = train_parsed_obj_dataset

# train_dataset = train_dataset.shuffle(32 * batch_size, tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(128 * batch_size)

train_dataset = train_dataset.padded_batch(batch_size = batch_size, padding_values = (0.0, 1e-8, -1), drop_remainder = False)

train_dataset = train_dataset.map(batch_encode, tf.data.AUTOTUNE)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


def visualize_detection( image, gts_bbox, figsize = (16,16), linewidth = 1.5, color = [1, 0, 0] ):
    
    feature_level = 5
    
    image = np.array(image, dtype = np.uint8)
    
    plt.figure(figsize = figsize)
    # plt.grid(visible = False, axis = 'both', linestyle = '-.')
    
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    print("image_width", image_width)
    print("image_height", image_height)

    yticks_num = np.math.ceil(image_height/ 2 ** feature_level)
    xticks_num = np.math.ceil(image_width / 2 ** feature_level)
#     print("xticks_num", xticks_num)
#     print("yticks_num", yticks_num)
    
    xticks = np.linspace(0, image_width, xticks_num)
    yticks = np.linspace(0, image_height, yticks_num)
    
    plt.xticks(xticks, rotation = 'horizontal')
    plt.yticks(yticks, rotation = 'vertical')
    
    plt.imshow(image)
    ax = plt.gca()
    
    for bbox in gts_bbox:
        
        xmin, ymin, xmax, ymax, key1x, key1y, key2x, key2y, key3x, key3y, key4x, key4y = bbox

        # # print("compute rbox xc", rbox[0])
        # # print("compute rbox yc", rbox[1])
        # # print("compute rbox width", rbox[2])
        # # print("compute rbox height", rbox[3])
        # # print("compute rbox angle",  rbox[4])

        # patch = plt.Polygon(
            # [xy1, xy2, xy3, xy4], closed=True, edgecolor='r', alpha = 0.2, fill = True, linewidth=6
        # )

        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0 
        width = xmax - xmin 
        height = ymax - ymin 
        patch = plt.Rectangle(
            (xmin, ymin), width, height,  fill=True, color = [1,0,0], alpha=0.5
        )

        ax.add_patch(patch)
        
        patch2 = plt.Circle(
            [key1x, key1y], radius = 5, fill=True, color = [0,1,0]
        )
        ax.add_patch(patch2)

        patch3 = plt.Circle(
            [key2x, key2y], radius = 5, fill=True, color = [0,1,0]
        )
        ax.add_patch(patch3)

        patch4 = plt.Circle(
            [key3x, key3y], radius = 5, fill=True, color = [0,1,0]
        )
        ax.add_patch(patch4)    

        patch5 = plt.Circle(
            [key4x, key4y], radius = 5, fill=True, color = [0,1,0]
        )
        ax.add_patch(patch5) 
    
    plt.show()
    
    return ax


count = 0
for image, label in train_dataset.take(10):
    #data_format HWC
    image_height = tf.cast(tf.shape(image)[1], dtype = tf.float32)
    image_width  = tf.cast(tf.shape(image)[2], dtype = tf.float32)
    count = count + 1
    # print("image_height %f " % image_height)
    # print("image_width %f " % image_width)
    #bbox.shape (batch_size, nums_gt, 4)
    bbox = label[:, :, :12]
    label = label[:, :, 12]
    print("label", label)
    # print(label)

    # print("bbox.shape")
    # print(bbox.shape)
#     print("label", label)
    # visualize_detection(image[0], bbox[0])
    
    visualize_detection(image[0], bbox[0])

print("count", count)

backboneV3 = tf.keras.applications.MobileNetV3Small(
    include_top=False, input_shape=[None, None, 3],
    minimalistic=False,
    include_preprocessing = True,
    weights="model/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5",
)


P3, P4, P5 = [
     backboneV3.get_layer(layer_name).output for layer_name in ['expanded_conv_2/Add','expanded_conv_7/Add','expanded_conv_10/Add']
 ]

mobileV3Small = tf.keras.Model(
    inputs = [backboneV3.inputs], outputs = [P3, P4, P5]
)


backboneV4 = tf.keras.applications.MobileNetV3Large(
    include_top=False, input_shape=[None, None, 3],
    minimalistic=False,
    include_preprocessing = True,
    weights="model/weights_mobilenet_v3_large_224_1.0_float_no_top_v2.h5",
)


P3_V4, P4_V4, P5_V4 = [
     backboneV4.get_layer(layer_name).output for layer_name in ['expanded_conv_5/Add','expanded_conv_11/Add','expanded_conv_14/Add']
 ]

mobileV3Large = tf.keras.Model(
    inputs = [backboneV4.inputs], outputs = [P3_V4, P4_V4, P5_V4]
)

backboneV6 = tf.keras.applications.MobileNetV2(
    include_top = False,
    input_shape = (None, None, 3),
    pooling = "avg",
    weights = "model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
)

P3_V6, P4_V6, P5_V6 = [
    backboneV6.get_layer(layer_name).output for layer_name in ['block_5_add', 'block_12_add', 'block_15_add']
]

mobileV2 = tf.keras.Model(
    inputs = [backboneV6.inputs], outputs = [P3_V6, P4_V6, P5_V6]
)

# ============================================================================
# Backbone: EfficientNetV2B0 主干网络
# ============================================================================
# 提取多尺度特征：
#   - C3 (stride=8): block3b_add  - 用于检测小目标
#   - C4 (stride=16): block5e_add - 用于检测中等目标
#   - C5 (stride=32): block6h_add - 用于检测大目标
backboneV5 = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    input_shape= (None, None, 3),
    include_preprocessing = True,
    pooling="avg",
    weights = "model/efficientnetv2-b0_notop.h5",
)

P3_V5, P4_V5, P5_V5 = [
     backboneV5.get_layer(layer_name).output for layer_name in ['block3b_add','block5e_add','block6h_add']
]

efficientNetV2B0 = tf.keras.Model(
    inputs = [backboneV5.inputs], outputs = [P3_V5, P4_V5, P5_V5]
)

# ============================================================================
# 激活函数和基础层
# ============================================================================

class HardSwish(tf.keras.layers.Layer):
    """
    HardSwish激活函数
    公式: x * relu6(x + 3) / 6
    用于MobileNetV3和EfficientNet，比ReLU更平滑
    """
    
    def __init__(self):
        super(HardSwish, self).__init__()
    
    def call(self, x, training = True):
        r = x * (tf.nn.relu6(x + 3.0) / 6.0)
        return r

class Conv(tf.keras.layers.Layer):
    """
    标准卷积层：Conv2D + BatchNorm + HardSwish
    支持深度可分离卷积（用于减少参数量）
    """
    
    def __init__(self, name = "Conv", 
                         out_channel = 128, 
                         kernel_size = 3,
                         strides = 1,
                         use_separable = False, **kwargs):
        
        super(Conv, self).__init__(name = name, **kwargs)
        
        self.out_channel = out_channel
        
        self.kernel_size = kernel_size
        
        self.use_separable = use_separable
        
        self.strides = strides
        
    def build(self, input_shape):
        
        if self.use_separable:
        #使用深度可分离卷积
            self.conv = tf.keras.layers.SeparableConv2D(filters = self.out_channel,
                                                        kernel_size = self.kernel_size,
                                                        strides      = self.strides,
                                                        padding = 'same'
                                                        )
        else:
            #是正常的卷积
            self.conv = tf.keras.layers.Conv2D(filters = self.out_channel,
                                                kernel_size = self.kernel_size,
                                                strides     = self.strides,
                                                padding = 'same'
                                              )
        self.bn   = tf.keras.layers.BatchNormalization()
        
        self.act  = HardSwish()

        # self.act = tf.keras.layers.Activation('relu')
        
    def call(self, x, training = True):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x

#Fast SPP (Spatial Pyramid Pooling - Fast)
class SPPF(tf.keras.layers.Layer):
    """
    快速空间金字塔池化模块
    通过多个不同尺度的最大池化操作，增强特征的感受野
    用于提取多尺度上下文信息
    """
    
    def __init__(self, name = "SPPF",
                 out_channel = 96,
                 **kwargs):
        
        super(SPPF, self).__init__(name = name, **kwargs)
        
        self.out_channel = 96
    
        self.mid_channel = int(self.out_channel // 2)
        
        self.first_conv = Conv(out_channel = self.mid_channel,
                               kernel_size = 1)
        
        self.final_conv = Conv(out_channel = self.out_channel,
                               kernel_size = 1)
        
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size = 5,
                                                      strides = 1,
                                                      padding = 'same')
     
    def call(self, x, training = True):
        # 通过级联的maxpool获得不同尺度的特征
        x = self.first_conv(x)
        
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        
        # 拼接多尺度特征
        x  = tf.concat([x, y1, y2, y3], axis = -1)
        
        return self.final_conv(x)


class CustomUpsampling2D(tf.keras.layers.Layer):

    def __init__(self, name = "CustomUpsampling2D", size=(2, 2), data_format=None, interpolation="nearest", **kwargs):
        super(CustomUpsampling2D, self).__init__(name = name, **kwargs)
        self.size = size
        self.data_format = data_format
        
    def call(self, x, training = True):
        
        if self.data_format == "channels_first":
            x = tf.transpose(x, [0, 2, 3, 1])

        # width = tf.shape(x)[1]
        # height = tf.shape(x)[2]

        # new_shape = (width * self.size[0], height * self.size[1])

        # x = tf.image.resize(x, new_shape,  method = "nearest")
        x = tf.repeat(x, self.size[0], axis = 1)
        x = tf.repeat(x, self.size[1], axis = 2)
        
        if self.data_format == "channels_first":
            x = tf.transpose(x, [0, 3, 1, 2])
        return x
        

class CSPPAnet(tf.keras.layers.Layer):
    """
    CSP-PANet: 特征融合网络（Neck）
    
    结构：
        1. Backbone提取多尺度特征 (C3, C4, C5)
        2. SPPF模块增强C5特征
        3. 上采样路径 (Top-Down): C5→C4→C3，融合高层语义信息
        4. 下采样路径 (Bottom-Up): C3→C4→C5，融合低层细节信息
    
    输出：P3, P4, P5 (多尺度融合后的特征)
    """
    
    def __init__(self, name = "CSPPAnet", **kwargs):
        super(CSPPAnet, self).__init__(name = name, **kwargs)
        
        # 使用EfficientNetV2B0作为主干网络
        self.backbone = efficientNetV2B0
        
        # SPPF模块：在C5特征上应用，增强感受野
        self.spp = SPPF(out_channel = 512)
        
        self.out_channel = 128
        
        self.conv_c3_1x1_inp = Conv(out_channel = self.out_channel,
                                    kernel_size = 1)
    
        self.conv_c4_1x1_inp = Conv(out_channel = self.out_channel, 
                                    kernel_size = 1)
    
        self.conv_c5_1x1_inp = Conv(out_channel = self.out_channel,
                                    kernel_size = 1)
        
        self.up_sample_2x2    = tf.keras.layers.UpSampling2D(size=(2, 2))
        # self.up_sample_2x2   = CustomUpsampling2D(size=(2,2))
        self.up_sample_c5_2x2    = tf.keras.layers.Conv2DTranspose(
                                        kernel_size = 3,
                                        filters= 96,
                                        strides= 2, 
                                        padding='same')

        self.up_sample_c4_2x2    = tf.keras.layers.Conv2DTranspose(
                                        kernel_size = 3,
                                        filters= 112,
                                        strides= 2,
                                        padding='same')
        
        self.down_sample_c3_2x2 = Conv(out_channel = self.out_channel, 
                                       kernel_size = 3, 
                                       strides = 2, 
                                       use_separable = False)
        
        self.down_sample_c4_2x2 = Conv(out_channel = self.out_channel, 
                                       kernel_size = 3, 
                                       strides = 2, 
                                       use_separable = False)
        
        self.csp_pan_3 = Conv(out_channel = self.out_channel,
                              kernel_size = 3)
        
        self.csp_pan_4_1 = Conv(out_channel = self.out_channel,
                              kernel_size = 3)
        
        self.csp_pan_4_2 = Conv(out_channel = self.out_channel,
                              kernel_size = 3)
        
        self.csp_pan_5   = Conv(out_channel = self.out_channel,
                              kernel_size = 3)
        
    def call(self, image, training = True):
        """
        前向传播
        
        流程：
            1. Backbone提取特征: C3(stride=8), C4(stride=16), C5(stride=32)
            2. SPPF增强C5特征
            3. 上采样路径 (Top-Down): 融合高层语义到低层
            4. 下采样路径 (Bottom-Up): 融合低层细节到高层
        """
        
        # 提取多尺度特征
        c3, c4, c5 = self.backbone(image, training)

        # SPPF模块增强C5特征（增加感受野）
        c5 = self.spp(c5)
        
        c3_inp = c3  # stride=8的特征
        c4_inp = c4  # stride=16的特征
        c5_inp = c5  # stride=32的特征（经过SPPF）

        print("c5_inp.shape", c5_inp.shape)
        print("c4_inp.shape", c4_inp.shape)
        
        # ========== 上采样路径 (Top-Down) ==========
        # C5 → C4: 上采样并融合
        c4_td  = tf.concat([c4_inp, self.up_sample_2x2(c5_inp)], axis = -1)
        c4_td  = self.csp_pan_4_1(c4_td)
    
        # C4 → C3: 上采样并融合
        c3_out = tf.concat([c3_inp, self.up_sample_2x2(c4_td)], axis = -1)
        c3_out = self.csp_pan_3(c3_out)
        
        # ========== 下采样路径 (Bottom-Up) ==========
        # C3 → C4: 下采样并融合
        c4_out = tf.concat([c4_td, self.down_sample_c3_2x2(c3_out)], axis = -1)
        c4_out = self.csp_pan_4_2(c4_out)
        
        # C4 → C5: 下采样并融合
        c5_out = tf.concat([c5_inp, self.down_sample_c4_2x2(c4_out)], axis = -1)
        c5_out = self.csp_pan_5(c5_out)
        
        # 输出多尺度特征
        p3_out = c3_out  # stride=8，用于检测小目标
        p4_out = c4_out  # stride=16，用于检测中等目标
        p5_out = c5_out  # stride=32，用于检测大目标
        
        # # # print("p3_out.shape", p3_out.shape)
        # # # print("p4_out.shape", p4_out.shape)
        # # # print("p5_out.shape", p5_out.shape)
        
        return (p3_out, p4_out, p5_out)

#默认的stride = (8.0, 16.0, 32.0)
def generate_multi_level_anchor_center(image_height, image_width, stride_list = (8, 16, 32)):
    
    anchor_centers = []
    anchor_strides = []
    
    for stride in stride_list:
        feature_height = tf.cast(image_height / stride, dtype = tf.int32)
        feature_width =  tf.cast(image_width /  stride, dtype = tf.int32)
        cx = tf.range(feature_height, dtype = tf.float32) + 0.5
        cy = tf.range(feature_width,  dtype = tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(cy, cx),  axis = - 1)
        #centers.shape (feature_height, feature_width, 2)
        centers = tf.expand_dims(centers, axis = 0)
        #centers.shape (1, feature_height, feature_width, 2)
        centers = tf.tile(centers, (1, 1, 1, 1))
        strides = tf.fill(value = tf.cast(stride, dtype = tf.float32),
                          dims = (1, feature_height, feature_width, 1))
        
        centers = tf.reshape(centers, (-1, feature_height * feature_width, 2))
        strides = tf.reshape(strides, (-1, feature_height * feature_width, 1))
        
        anchor_centers.append(centers)
        anchor_strides.append(strides)
        
    anchor_centers        = tf.concat(anchor_centers, axis = 1)
    anchor_strides        = tf.concat(anchor_strides, axis = 1) 
    
    #anchor_centers (batch_size, anchors_num, 2)
    #anchor_strides (batch_size, anchors_num, 1)
    return (anchor_centers, anchor_strides)


#独立生成anchor_centers
#ratio = image_width / image_height
def generate_anchor_centers(batch_size, anchors_num, ratio = 1.0):
    #get strides list
    ratio = 1
    strides = (8.0, 16.0, 32.0)
    #通过anchors_num 计算image_height和image_width
    #假设 image_height == image_width
    a = 1 # a 表示stride = 8
    b = 1 # b 表示stride = 16
    c = 1 # c 表示stride = 32
    # 假设 image_height == image_width 
    denominator = 16 * a + 4 * b + c
    #默认 image_height == image_width
#     image_height = image_width = tf.stop_gradient(32 * tf.math.sqrt(anchors_num / denominator))
    # image_height * 2 == image_width,  height:width = 1:2
    
    anchors_num = tf.cast(anchors_num, dtype = tf.float32)
    
    image_height = tf.stop_gradient(32 * tf.math.sqrt(anchors_num /(ratio * denominator)))
    image_width  = image_height * ratio
    
    print("image_height", image_height)
    print("image_width",  image_width)
    anchor_centers = []
    anchor_strides = []
    
    for stride in strides:
        feature_height = tf.cast(image_height / stride, dtype = tf.int32)
        feature_width =  tf.cast(image_width /  stride, dtype = tf.int32)
        cx = tf.range(feature_height, dtype = tf.float32) + 0.5
        cy = tf.range(feature_width,  dtype = tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(cy, cx),  axis = - 1)
        #centers.shape (feature_height, feature_width, 2)
        centers = tf.expand_dims(centers, axis = 0)
        #centers.shape (1, feature_height, feature_width, 2)
        centers = tf.tile(centers, (batch_size, 1, 1, 1))
        strides = tf.fill(value = stride, dims = (batch_size, feature_height, feature_width, 1))
        
#         centers = centers * strides

        #centers.shape (batch_size, feature_height * feature_width, 2)
        #strides.shape (batch_size, feature_height * feature_width, 1)
        centers = tf.reshape(centers, (-1, feature_height * feature_width, 2))
        strides = tf.reshape(strides, (-1, feature_height * feature_width, 1))
        
        anchor_centers.append(centers)
        anchor_strides.append(strides)
        
    anchor_centers        = tf.concat(anchor_centers, axis = 1)
    anchor_strides        = tf.concat(anchor_strides, axis = 1) 
    
    #anchor_centers (batch_size, anchors_num, 2)
    #anchor_strides (batch_size, anchors_num, 1)
    return (anchor_centers, anchor_strides)

def bbox_decode(anchors_center, pred_dist):
    """
    边界框解码函数
    
    将DFL分布转换为实际的距离值，然后计算边界框坐标
    
    流程：
        1. 对距离分布应用softmax，得到概率分布
        2. 通过加权求和将分布转换为实际距离: delta = sum(prob * index)
        3. 根据anchor中心点和距离计算边界框: [cx-delta_l, cy-delta_t, cx+delta_r, cy+delta_b]
    
    参数：
        anchors_center: (B, N, 2) - anchor中心点坐标
        pred_dist: (B, N, 4*reg_max) - 距离分布预测
    
    返回：
        bbox: (B, N, 4) - [x1, y1, x2, y2]
    """
    reg_max = 128  # 距离分布的最大值 [0, 1, 2, ..., 127]
    
    # 投影向量: [0, 1, 2, ..., reg_max-1]
    proj = tf.range(0, reg_max, dtype = tf.float32)
    proj = proj[:, None]  # (reg_max, 1)
    
    c = tf.shape(pred_dist)[-1]
    anchors_num = tf.shape(pred_dist)[1]
    
    # 转换为概率分布
    pred_dist = tf.nn.softmax(pred_dist, axis = -1)
    
    # Reshape: (B, N, 4*reg_max) -> (B, N, 4, reg_max)
    pred_dist = tf.reshape(pred_dist, (-1,anchors_num, 4, c // 4))
    
    print("pred_dist.shape", pred_dist.shape)
    
    # 通过加权求和将分布转换为实际距离
    # delta.shape (batch_size, anchors_num, 4) - [left, top, right, bottom]
    delta = tf.matmul(pred_dist, proj)
    delta = tf.squeeze(delta, axis = -1)
    print("anchors_center.shape", anchors_center.shape)
    print("delta.shape", delta.shape)
    
    # 确保距离非负
    tf.debugging.assert_non_negative(delta, message = "delta (l,t,r,b) >= 0")
    
    # 计算边界框坐标
    # anchors_center.shape (batch_size, anchors_num, 2) - [cx, cy]
    # bbox (x0, y0, x1, y1)
    return tf.stack([
        anchors_center[..., 0] - delta[..., 0],  # x1 = cx - left
        anchors_center[..., 1] - delta[..., 1],  # y1 = cy - top
        anchors_center[..., 0] + delta[..., 2],  # x2 = cx + right
        anchors_center[..., 1] + delta[..., 3]   # y2 = cy + bottom
    ], axis = -1)


def keypoint_decode(anchors_center, pred_dist):
    """
    关键点解码函数
    
    将相对坐标转换为绝对坐标
    
    参数：
        anchors_center: (B, N, 2) - anchor中心点坐标
        pred_dist: (B, N, 8) - 关键点相对坐标预测 [key1x,key1y, key2x,key2y, key3x,key3y, key4x,key4y]
    
    返回：
        keypoints: (B, N, 8) - 绝对坐标
    """
    # 将anchor中心点复制4次，对应4个关键点
    anchors_center = tf.tile(anchors_center, (1, 1, 4))  # (B, N, 8)
    print("anchors_center.shape", anchors_center.shape)
    
    # 关键点坐标 = anchor中心点 + 相对偏移
    return pred_dist + anchors_center



class TalAssign:
    """
    TAL (Task Alignment Learning) 正负样本分配策略
    
    功能：
        1. 计算anchor与gt的对齐分数（结合分类分数和IoU）
        2. 选择gt框内的anchors作为候选
        3. 每个gt选择top-k (k=13) 个最佳匹配的anchors作为正样本
        4. 其他anchors分配为负样本（背景）
    
    对齐分数计算：
        metrics = (classification_score)^alpha * (IoU)^beta
        - alpha: 分类分数权重（默认0.5）
        - beta: IoU权重（默认6.0）
    
    特点：
        - 动态匹配：根据预测质量动态分配正负样本
        - 任务对齐：同时考虑分类和定位任务的对齐程度
    """
    
    def __init__(self, alpha = 0.5 , beta = 6.0):
        self.alpha = alpha  # 分类分数权重
        self.beta  = beta   # IoU权重
        
    #计算 anchors_box 与 gts_box 的iou_matrix
    def __iou_matrix(self, predicts_box, gts_box):
        #anchors_box (B, N, 4)
        #gts_box     (B, M, 4)
        predict_num = tf.shape(predicts_box)[1]
        gt_num     = tf.shape(gts_box)[1]
        #生成最终的matrix (B, M, N)
        predicts_box = tf.expand_dims(predicts_box, axis = 1)
        #anchors_box.shape (B, 1, N, 4)
        predicts_box = tf.tile(predicts_box, (1, gt_num, 1, 1))
    
        gts_box     = tf.expand_dims(gts_box, axis = 2)
        #gts_box.shape (B, M, 1, 4)
        gts_box     = tf.tile(gts_box, (1, 1, predict_num, 1))
        #gts_box.shape (B, M, N, 4)
    
#         print("predicts_box.shape", predicts_box.shape)
#         print("gts_box.shape", gts_box.shape)
        
        boxes1_x1, boxes1_y1,  boxes1_x2,  boxes1_y2 = tf.unstack(predicts_box, axis = -1)
        boxes2_x1, boxes2_y1,  boxes2_x2,  boxes2_y2 = tf.unstack(gts_box, axis = -1)
    
        lu_x1 = tf.maximum(boxes1_x1, boxes2_x1)
        lu_y1 = tf.maximum(boxes1_y1, boxes2_y1)
        rd_x2 = tf.minimum(boxes1_x2, boxes2_x2)
        rd_y2 = tf.minimum(boxes1_y2, boxes2_y2)
        
        intersection_w = tf.maximum(0.0, rd_x2 - lu_x1)
        intersection_h = tf.maximum(0.0, rd_y2 - lu_y1)
            
        intersection_area = intersection_w * intersection_h
    
        area1 = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
        area2 = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
        union_area = area1 + area2 - intersection_area
    
        iou = tf.clip_by_value(intersection_area / (union_area + 1e-8), 0.0, 1.0)
        #iou.shape (B, gt_num, anchor_num)
        return iou
    
    def __score_matrix(self, anchors_score, gts_score):
        #anchors_box (B, N, nums_class)
        #gts_box     (B, M, nums_class)
        anchor_num = tf.shape(anchors_score)[1]
        gt_num     = tf.shape(gts_score)[1]
        #生成最终的matrix (B, M, N)
        anchors_score = tf.expand_dims(anchors_score, axis = 1)
        #anchors_box.shape (B, 1, N, 4)
        anchors_score = tf.tile(anchors_score, (1, gt_num, 1, 1))
    
        gts_score     = tf.expand_dims(gts_score, axis = 2)
        #gts_box.shape (B, M, 1, 4)
        #gts_box.shape (B, M, N, 4)
        gts_score     = tf.tile(gts_score, (1, 1, anchor_num, 1))
        
        #计算anchors_score 和 gts_score 的交叉熵
        #anchors_score.shape (B, N, M. nums_class)
        #gts_score.shape     (B, N, M, nums_class)
        epsilon = tf.constant(1e-6, dtype = tf.float32)
        anchors_score = tf.clip_by_value(anchors_score, epsilon, 1.0 - epsilon)
        negative_bce    = gts_score * tf.math.log(anchors_score + epsilon)
        negative_bce    += (1 - gts_score) * tf.math.log(1 - anchors_score + epsilon)
        bce  = -negative_bce
        #bce.shape (B, M, N, nums_class)
        #return bce.shape(B, M, N)
        return tf.reduce_mean(bce, axis = -1)
    
    #计算对齐分数
    def compute_anchor_alignment_metrics(self, anchors_score, gts_score, anchors_box, gts_box, anchors_stride, anchors_obj):
        #anchors_score.shape (B, anchors_num, nums_class)
        #anchors_box         (B, anchors_num, 4)
        #gts_box             (B, gts_num, 4)
        #gts_score           (B, gts_num, nums_class)
        gts_label_ind  = tf.argmax(gts_score, axis = -1)
        
        
        #还原真实的box预测, bug....
        predicts_box = anchors_box * anchors_stride
        #gts_label.shape     (B, gts_num)
        ious_matrix = self.__iou_matrix(predicts_box, gts_box)
        #iou_matrix.shape (B, gts_num, anchors_num)
        #anchors_score.shape (B, anchors_num, nums_class)
        #anchors_obj.shape (B, anchors_num, )
#         anchors_score = anchors_score * anchors_obj[:, :, None]
        anchors_score = tf.transpose(anchors_score, (0, 2, 1))
        #anchors_score.shape (B, nums_class, anchors_num)
        #获得gts_label关联的anchor_scores
        scores_matrix = tf.gather(anchors_score, gts_label_ind, batch_dims = 1)
        #bbox_score.shape (B, gts_num, anchors_num)
        #iou_matrix (B, gts_num, anchors_num)
        #加入一个小值,以前top 13 
        metrics_matrix = tf.math.pow(scores_matrix, self.alpha) * tf.math.pow(ious_matrix, self.beta)
#         metrics_matrix = scores_matrix + ious_matrix 
        return (metrics_matrix, ious_matrix)
    
    #判断anchors_point 是否在gts_box 中
    def __in_box(self, anchors_center, anchors_stride, gts_box):
        #anchors_box.shape (B, anchor_num, 4)
        #gts_box.shape (B, gt_num, 4)
        anchor_num = tf.shape(anchors_center)[1]
        gt_num     = tf.shape(gts_box)[1]
        #gts_box.shape (batch_size, num_gt, 4)
        #增加一维
        gts_box = tf.expand_dims(gts_box, axis = 2)
        #gts_box.shape (batch_size, num_gt, 1, 4)
        #平铺
        gts_box = tf.tile(gts_box, (1, 1, anchor_num, 1))
        #gts_box.shape (batch_size, num_gt, num_anchor, 4)
        # x1, y1, x2, y2
        
        #得到anchor的中心点
        #还原真实的anchors_center
        anchors_center = anchors_center * anchors_stride
        anchors_center = tf.expand_dims(anchors_center, axis = 1)
        anchors_center = tf.tile(anchors_center, (1, gt_num, 1, 1))
        #anchors_center.shape (batch_size, num_gt, num_anchor, 2)
        
        in_box = tf.concat([
            #(c_x, c_y) (x1, y1)
            anchors_center - gts_box[..., :2],
            #(c_x, c_y) (x2, y2)
            gts_box[..., 2:] - anchors_center
        ], axis = -1)
        
        return tf.greater(tf.reduce_min(in_box, axis = -1), 0)
    
     #判断anchors_point 是否在gts_box 中 3x3 中心
    def __in_center(self, anchors_center, anchors_stride, gts_box):
        #anchors_box.shape (B, anchor_num, 4)
        #gts_box.shape (B, gt_num, 4)
        batch_size = tf.shape(gts_box)[0]
        anchor_num = tf.shape(anchors_center)[1]
        gt_num     = tf.shape(gts_box)[1]
        
        #gts_box.shape (batch_size, num_gt, 4)
        #增加一维
        gts_box = tf.expand_dims(gts_box, axis = 2)
        #gts_box.shape (batch_size, num_gt, 1, 4)
        #平铺
        gts_box = tf.tile(gts_box, (1, 1, anchor_num, 1))
        
        #gts_box.shape (batch_size, num_gt, num_anchor, 4)
        # x1, y1, x2, y2
        gt_x1, gt_y1, gt_x2, gt_y2 = tf.unstack(gts_box, axis = -1)
        gt_c_x = (gt_x2 + gt_x1) / 2
        gt_c_y = (gt_y2 + gt_y1) / 2
        
        #还原真实的anchors_center
        anchors_center = anchors_center * anchors_stride
        
        #anchors_stride.shape (batch_size, 1, anchor_num, 1)
        anchors_stride = tf.expand_dims(anchors_stride, axis = 1)
        #平铺
        anchors_stride = tf.tile(anchors_stride, (1, gt_num, 1, 1))
        #stride.shape (batch_size, num_gt, num_anchor)
        
        #注意，可能出现，每一层的点都不在当前中心框中
        # 使用 2x2的网格
        radius = 2.5 * anchors_stride[:,:,:,0]
#         print("radius", radius[:, 0, :])
        #将bounds_box_xy 限制在 3x3 的网格中
        bounds_box_xy = tf.stack([
            #left-x x1
            tf.maximum(gt_c_x - radius, gt_x1),
            #top-y y1
            tf.maximum(gt_c_y - radius, gt_y1),
            #right-x x2
            tf.minimum(gt_c_x + radius, gt_x2),
            #bottom-y y2
            tf.minimum(gt_c_y + radius, gt_y2),
        ], axis = -1)
        
        #anchors_center (batch_size, num_anchor, 2)
        anchors_center = tf.expand_dims(anchors_center, axis = 1)
        anchors_center = tf.tile(anchors_center, (1, gt_num, 1, 1))
        
        in_center = tf.concat([
            #(c_x, c_y) (x1, y1)
            anchors_center - bounds_box_xy[...,:2],
            #(c_x, c_y) (x2, y2)
            bounds_box_xy[...,2:] - anchors_center
        ], axis = -1)
        
        return tf.greater(tf.reduce_min(in_center, axis = -1), 0) 
    
    def solve(self, anchors_score, gts_score, anchors_box, gts_box, anchors_keypoint, gts_keypoint, anchors_center, anchors_stride, anchors_obj):
        
        
#         print("anchors_box.shape", anchors_box.shape)
#         print("anchors_stride.shape", anchors_stride.shape)
        
        batch_size = tf.shape(anchors_score)[0]
        
        gt_num     = tf.shape(gts_score)[1]
        
        anchors_num = tf.shape(anchors_score)[1]
        
        class_nums = tf.shape(gts_score)[-1]
        
#         print("gts_label", tf.argmax(gts_score, axis = -1))
        
        #背景分类
        bg_score = tf.fill(dims = (batch_size, 1, class_nums), value = 0.0)
        #背景box
        bg_box = tf.fill(dims = (batch_size, 1, 4), value = 0.0)
        #背景keypoint
        bg_keypoint = tf.fill(dims = (batch_size, 1, 8), value = 0.0)
        
        #in_box_mask.shape (B, gts_num, anchors_num)
        in_center_mask = self.__in_center(anchors_center, anchors_stride, gts_box)
        in_box_mask  = self.__in_box(anchors_center, anchors_stride, gts_box)
        #in_box_mask.shape (B, num_gt, num_anchor)
#         print("in_center_mask", in_center_mask[0])
        #metrics_matrix.shape (B, gts_num, anchors_num)
        metrics_matrix, ious_matrix = self.compute_anchor_alignment_metrics(anchors_score, gts_score, anchors_box, gts_box, anchors_stride, anchors_obj)
        
#         print("metrics_matrix min", tf.reduce_min(metrics_matrix, axis = -1)[0])
#         print("metrics_matrix max", tf.reduce_max(metrics_matrix, axis = -1)[0])
#         print("ious_matrix min", tf.reduce_min(ious_matrix, axis = -1)[0])
#         print("ious_matrix max", tf.reduce_max(ious_matrix, axis = -1)[0])
        #只选择gt框内的anchors
        fg_metrics_cost = -metrics_matrix + 4e8 * (1 - tf.cast(in_box_mask, dtype = tf.float32))
#         fg_metrics_cost = -metrics_matrix + 4e8 * (1 - tf.cast(in_center_mask, dtype = tf.float32))
        #在gt框中，选择前13个最大的anchors
        #Finds values and indices of the k largest entries for the last dimension.
        values , indices = tf.math.top_k(-fg_metrics_cost, k = 13)
        
#         print("fg_metrics_cost top_k = 13", values[0])
#         print("fg_metrics_cost indices_k = 13", indices[0])
        #indices.shape (B, gts_num, 13)
        temp = tf.one_hot(indices, depth = tf.shape(fg_metrics_cost)[-1])
        #temp.shape (B, gts_num, 13, anchors_num)
        topk_mask = tf.cast(tf.reduce_sum(temp, axis = -2), dtype = tf.bool)
#         print("temp.shape", temp.shape)
        #temp.shape (B, gts_num, anchors_num)
        #fg_cost.shape (B, gts_num, anchors_num)
        #选择这些选出来的anchors
        fg_metrics_cost = fg_metrics_cost + 3e8 * (1 - tf.cast(topk_mask, dtype = tf.float32))
        
#         print("fg_metrics_cost min", tf.reduce_min(fg_metrics_cost, axis = -1))
#         print("fg_metrics_cost max", tf.reduce_max(fg_metrics_cost, axis = -1))
        
#         print("in_box_mask stat", tf.reduce_sum(tf.cast(in_box_mask, dtype = tf.int32), axis = -1))
        
#         print("fg_metrics_cost gt_box min", tf.reduce_min(fg_metrics_cost, axis = -1))
#         print("fg_metrics_cost gt_box max", tf.reduce_max(fg_metrics_cost, axis = -1))
        
        fg_ious_cost   =  ious_matrix 
        #增加一个bg gt
        bg_metrics_cost = tf.ones_like(fg_metrics_cost[..., 0:1, :]) * 2e8
        bg_ious_cost     = tf.ones_like(fg_ious_cost[..., 0:1, :]) * 2e8
        #包含bg框的cost 矩阵
        metrics_cost = tf.concat([fg_metrics_cost, bg_metrics_cost], axis = -2)
        ious_cost = tf.concat([fg_ious_cost, bg_ious_cost], axis = -2)
        
        
#         print("metrics_cost min", tf.reduce_min(metrics_cost, axis = -1))
#         print("metrics_cost max", tf.reduce_max(metrics_cost, axis = -1))
        
#         print("metrics_cost min_per_anchor", tf.reduce_min(metrics_cost, axis = -2)[0])
        #bg_cost.shape (B, gts_num, anchors_num)
        #anchor只选择最大映射的gt
        matched_gts_idx = tf.argmin(metrics_cost, axis = -2, output_type = tf.int32)
        #matched_gts_idx.shape (B, anchors_num)
        matched_metrics_val = tf.reduce_min(metrics_cost, axis = -2)
        #变成正值
        matched_metrics_val = -matched_metrics_val
        #matched_metrics_val.shape (B, anchors_num)
#         print("matched_metrics_val min", tf.reduce_min(matched_metrics_val, axis = -1)[0])
#         print("matched_metrics_val max", tf.reduce_max(matched_metrics_val, axis = -1)[0])
        
        matched_metrics_val_mask =  tf.not_equal(matched_gts_idx, gt_num)
        
        matched_metrics_val_v1   = matched_metrics_val[matched_metrics_val_mask]
        
#         print("matched_metrics_val_v1 min", tf.reduce_min(matched_metrics_val_v1, axis = -1))
#         print("matched_metrics_val_v1 max", tf.reduce_max(matched_metrics_val_v1, axis = -1))
        
        #matched_metrics_val.shape(B, anchors_num)
        #前景加背景框
        gts_all_box = tf.concat([gts_box, bg_box], axis = -2)
        #前景加背景分类
        gts_all_score = tf.concat([gts_score, bg_score], axis = -2)

        #前景加背景关键点
        gts_all_keypoint = tf.concat([gts_keypoint, bg_keypoint], axis = -2)
    
        #norm matched_metrics_val
        #in_box_mask.shape (B, gts_num, anchors_num)
        mask_in = tf.cast(tf.math.logical_and(in_box_mask, topk_mask), dtype = tf.float32)
#         mask_in = tf.cast(tf.math.logical_and(in_center_mask, topk_mask), dtype = tf.float32)
        #mask_in.shape (B, gts_num, anchors_num)
        mask_in = tf.concat([mask_in, tf.zeros_like(mask_in[:, 0:1, :])], axis = -2)
        #mask_in.shape (B, gts_num + 1, anchors_num)
        mask_metrics_matrix = tf.concat([metrics_matrix, tf.ones_like(metrics_matrix[:, 0:1, :])], axis = -2)
        mask_metrics_matrix = mask_metrics_matrix * mask_in
        mask_ious_matrix    = tf.concat([ious_matrix, tf.ones_like(ious_matrix[:, 0:1, :])], axis = -2)
        mask_ious_matrix    = mask_ious_matrix   * mask_in
        #mask_metrics_matrix.shape (B, gts_num + 1, anchors_num)
        max_metric_per_gts = tf.reduce_max(mask_metrics_matrix, axis = -1, keepdims = True)
        max_iou_per_gts    = tf.reduce_max(mask_ious_matrix,    axis = -1, keepdims = True)
        #max_iou_per_gts.shape (B, gts_num + 1, 1)
        #max_metric_per_gts.shape (B, gts_num + 1, 1)
        #归一化
        norm_mask_metrics_matrix = tf.math.divide_no_nan(
                                        mask_metrics_matrix,
                                        max_metric_per_gts) * \
                                     max_iou_per_gts
        #norm_mask_metrics_per_gts.shape (B, gts_num + 1, 1)
        #norm_mask_metrics_maxtrix.shape (B, gts_num + 1, 1)
        anchors_ind = tf.range(anchors_num)
        anchors_ind = tf.expand_dims(anchors_ind, axis = 0)
        anchors_ind = tf.tile(anchors_ind, (batch_size, 1))
        
        gather_ind = tf.stack([matched_gts_idx, anchors_ind], axis = -1)
        
        norm_matched_metrics_val = tf.gather_nd(norm_mask_metrics_matrix, gather_ind, batch_dims = 1)
        
        print("norm_matched_metrics_val.shape", norm_matched_metrics_val.shape)
        
        matched_gts_box      = tf.gather(gts_all_box,   indices = matched_gts_idx, batch_dims = 1, axis = 1)
        matched_gts_score    = tf.gather(gts_all_score, indices = matched_gts_idx, batch_dims = 1, axis = 1)
        matched_gts_keypoint = tf.gather(gts_all_keypoint, indices = matched_gts_idx, batch_dims = 1, axis = 1) 
        
        #正样本
        positive_mask = tf.not_equal(matched_gts_idx, gt_num)
        
#         print("reduce_positive_mask", tf.reduce_any(positive_mask, axis = -1))
        #box-norm
        matched_gts_box = matched_gts_box / anchors_stride

        matched_gts_keypoint = matched_gts_keypoint / anchors_stride
        
        matched_gts_obj = tf.reduce_any(in_center_mask, axis = -2)
        #anchor匹配的gts_box和gts_cls, 和正样本
        return (
                tf.stop_gradient(matched_gts_box),
                tf.stop_gradient(matched_gts_score),
                tf.stop_gradient(matched_gts_keypoint),
                tf.stop_gradient(norm_matched_metrics_val),
                tf.stop_gradient(positive_mask),
                tf.stop_gradient(matched_gts_obj),
            )


class P1(tf.keras.Model):
    """
    边界框回归头 (P1)
    
    功能：预测边界框的4个方向的距离分布
    输出：4 * reg_max 维 (每个方向预测reg_max个距离分布值)
    使用DFL (Distribution Focal Loss) 进行训练
    
    输出格式：
        - 每个anchor预测4个方向的距离: [left, top, right, bottom]
        - 每个方向预测reg_max个分布值 (默认128)
        - 总输出维度: 4 * reg_max = 512
    """
    def __init__(self, name = "P1",
                 max_reg = 16,
                 mid_channel = 256,
                 **kwargs):
        super(P1, self).__init__(name = name, **kwargs)
        
        self._bias_init = tf.keras.initializers.Zeros()
    
        self._mid_channel = mid_channel
        self._max_reg    = max_reg  # 距离分布的最大值 (默认128)
        self._final_channel = 4 * self._max_reg  # 4个方向 × reg_max
        
        self.block1_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)
        
        self.block2_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)

        self.block3_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)
        #1x1 pointwise conv
        self.box_delta_predict_conv = tf.keras.layers.Conv2D(
                                filters = self._final_channel,
                                kernel_size = (1,1),
                                padding = 'valid',
                                bias_initializer = self._bias_init,
        )
        
        
    def call(self, x, training = True):
        
        x = self.block1_conv(x)
        x = self.block2_conv(x)
        # x = self.block3_conv(x)
        
        x = self.box_delta_predict_conv(x)
        
        return x

class P2(tf.keras.Model):
    """
    分类头 (P2)
    
    功能：预测目标类别
    输出：num_classes 维 (每个类别一个分数)
    使用Sigmoid激活，支持多标签分类
    
    输出格式：
        - 每个anchor预测num_classes个类别分数
        - 默认3类：单层车牌、双层车牌、背景等
    """
    def __init__(self, num_classes, mid_channel = 256, name = "P2", **kwargs):
        super(P2, self).__init__(name = name, **kwargs)
        self._num_classes = num_classes
        self._mid_channel = mid_channel
        self._final_channel = 1 * num_classes
        
        self.block1_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)
    
        self.block2_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)

        self.block3_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)

        #1x1 pointwise conv bias_initializer 默认为零
        self.block5_conv = tf.keras.layers.Conv2D(
                                filters = self._final_channel,
                                kernel_size = (1,1),
                                padding = 'valid',
        )
        
        self.block5_act = tf.keras.layers.Activation("sigmoid")
        
    def call(self, x, training = True):
        
        x = self.block1_conv(x)
        x = self.block2_conv(x)
        # x = self.block3_conv(x)
        
        x = self.block5_conv(x)
        x = self.block5_act(x)
        
        #x.shape (batch_size, image_height / 32, image_width / 32, 1)
        return x

class P3(tf.keras.Model):
    """
    关键点预测头 (P3) - 四边形角点检测
    
    功能：预测车牌的4个角点坐标
    输出：8维 (4个角点 × 2坐标 = 8)
    
    输出格式：
        [key1x, key1y, key2x, key2y, key3x, key3y, key4x, key4y]
        key1: 左上角, key2: 右上角, key3: 右下角, key4: 左下角
    
    用于四边形车牌检测，可以处理倾斜、变形的车牌
    """
    def __init__(self, name = "P3",
                 mid_channel = 256,
                 **kwargs):
        super(P3, self).__init__(name = name, **kwargs)
        
        self._bias_init = tf.keras.initializers.Zeros()
    
        self._mid_channel = mid_channel
        self._max_reg    = 1  # 关键点直接预测坐标，不需要分布
        self._final_channel = 8 * self._max_reg  # 8维输出
        
        self.block1_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)
        
        self.block2_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)

        self.block3_conv = Conv(out_channel = self._mid_channel,
                                kernel_size = 3)
        #1x1 pointwise conv
        self.keypoint_delta_predict_conv = tf.keras.layers.Conv2D(
                                filters = self._final_channel,
                                kernel_size = (1,1),
                                padding = 'valid',
                                bias_initializer = self._bias_init,
        )
        
        
    def call(self, x, training = True):
        
        x = self.block1_conv(x)
        x = self.block2_conv(x)
        
        x = self.keypoint_delta_predict_conv(x)
        
        return x
        
class Detector(tf.keras.Model):
    """
    主检测模型
    
    架构：
        Backbone (EfficientNetV2B0) → Neck (CSPPAnet) → Head (P1, P2, P3)
    
    输出：
        - 分类: 3类 (单层车牌、双层车牌、背景等)
        - 边界框: 4 * reg_max = 512维 (DFL分布)
        - 关键点: 8维 (4个角点坐标)
    
    特点：
        - 无锚框 (Anchor-free)
        - 多尺度检测 (stride=8,16,32)
        - 四边形检测 (直接预测4个角点)
    """
    
    def __init__(self, name = "Detector", **kwargs):
        super(Detector, self).__init__(name = name, **kwargs)
        
        # 特征融合网络 (Neck)
        self.bifpn = CSPPAnet()
        
        # 多尺度stride
        self.strides = (8.0, 16.0, 32.0)
        
        # 类别数
        self.nums_class = 3
        
        # DFL距离分布的最大值
        self.reg_max = 128  # [0, 1, 2, ..., 127]
        
#         self.head = THead(num_classes = self.nums_class)

        self.p1_list = (
            P1(mid_channel = int(256 // 4),max_reg = self.reg_max), #P3 Detect Head
            P1(mid_channel = int(512 // 4),max_reg = self.reg_max), #P4 Detect Head
            P1(mid_channel = int(512 // 4),max_reg = self.reg_max)  #P5 Detect Head
        )
        
        self.p2_list = (
            P2(mid_channel = int(256 // 4), num_classes = self.nums_class), #P3 Detect Head
            P2(mid_channel = int(512 // 4), num_classes = self.nums_class), #P4 Detect Head
            P2(mid_channel = int(512 // 4), num_classes = self.nums_class)  #P5 Detect Head
        )

        self.p3_list = (
            P3(mid_channel = int(256 // 4)), #P3 Detect Head
            P3(mid_channel = int(512 // 4)), #P4 Detect Head
            P3(mid_channel = int(512 // 4))  #P5 Detect Head
        )
        
    def call(self, inputs, training = None):
        """
        前向传播
        
        输入: (B, H, W, 3) - 图像批次
        输出: (B, N, 3 + 512 + 8) - [分类, 边界框分布, 关键点]
            - 分类: 3维
            - 边界框: 512维 (4 * reg_max)
            - 关键点: 8维 (4个角点)
        """
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        image_height = input_shape[1]
        image_width  = input_shape[2]
        
        print("Detector image_height", image_height)
        print("Detector image_width",  image_width)
        
        # 特征融合网络提取多尺度特征
        feat_list = self.bifpn(inputs)  # [P3, P4, P5]
        
        cls_pred_list = []
        dist_pred_list = []
        keypoint_pred_list = []
        
        # 对每个特征层应用检测头
        for idx, feat in enumerate(feat_list):
            feat_height = tf.shape(feat)[1]
            feat_width  = tf.shape(feat)[2]
            
            # 分类头: 预测类别
            cls_pred = self.p2_list[idx](feat)
            # 边界框回归头: 预测距离分布
            dist_pred = self.p1_list[idx](feat)
            # 关键点预测头: 预测4个角点坐标
            keypoint_pred = self.p3_list[idx](feat)
            
            # Reshape: (B, H, W, C) -> (B, H*W, C)
            cls_pred = tf.reshape(cls_pred, shape = (-1, feat_height * feat_width, self.nums_class))
            dist_pred = tf.reshape(dist_pred, shape = (-1, feat_height * feat_width, 4 * self.reg_max))
            keypoint_pred = tf.reshape(keypoint_pred, shape = (-1, feat_height * feat_width, 8))
            
            cls_pred_list.append(cls_pred)
            dist_pred_list.append(dist_pred)
            keypoint_pred_list.append(keypoint_pred)
       
        # 拼接多尺度预测结果
        cls_pred_list      = tf.concat(cls_pred_list, axis = 1)  # (B, N, 3)
        dist_pred_list     = tf.concat(dist_pred_list, axis = 1)  # (B, N, 512)
        keypoint_pred_list = tf.concat(keypoint_pred_list, axis = 1)  # (B, N, 8)
        
        # 最终输出: [分类, 边界框分布, 关键点]
        # 形状: (B, N, 3 + 512 + 8) = (B, N, 523)
        return tf.concat([cls_pred_list, dist_pred_list, keypoint_pred_list], axis = -1)

class TooDNetDflLoss(tf.losses.Loss):
    """
    分布焦点损失 (Distribution Focal Loss, DFL)
    
    用于边界框回归，将距离预测转化为分布预测
    通过softmax将距离分布转换为实际距离值
    
    原理：
        - 真实距离可能不是整数，使用左右两个整数位置的加权和
        - loss = weight_left * CE(left) + weight_right * CE(right)
    """
    
    def __init__(self, reg_max = 16):
        super(TooDNetDflLoss, self).__init__(reduction = 'none', name = 'TooDNetDflLoss')
        
        self.__reg_max = reg_max  # 距离分布的最大值
        
    def call(self, y_true, y_pred):
        
        #将y_true 限制在[0, reg_max - 1]中
        
        y_true = tf.clip_by_value(y_true, 0, self.__reg_max - 1 - 0.01)
        
        
        print("y_true.shape", y_true.shape)
        print("y_pred.shape", y_pred.shape)
        
        dist_left = tf.math.floor(y_true)
        dist_right = dist_left + 1
        
        weight_left = y_true - dist_left
        weight_right = dist_right - y_true
        
        dist_left = tf.cast(dist_left, dtype = tf.int32)
        dist_right = tf.cast(dist_right, dtype = tf.int32)
        
        loss = weight_left * tf.nn.sparse_softmax_cross_entropy_with_logits(dist_left, y_pred) + \
               weight_right * tf.nn.sparse_softmax_cross_entropy_with_logits(dist_right, y_pred)
        
        #loss.shape (batch_size, anchors_num, 4)
        
        #return loss.shape (batch_size, anchors_num)
        return tf.reduce_mean(loss, axis = -1)
        
class TooDNetScoreLoss(tf.losses.Loss):
    """
    分类损失 (Binary Cross Entropy)
    
    用于多标签分类，每个类别独立计算BCE损失
    支持多标签场景（一个目标可能属于多个类别）
    """
    
    def __init__(self):
        super(TooDNetScoreLoss, self).__init__(
            reduction = "none", name = "TooDNetScoreLoss"
        )
    def call(self, y_true, y_pred):
        epsilon = tf.constant(1e-6, dtype = tf.float32)
        
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary Cross Entropy
        negative_cross_entropy = y_true * tf.math.log(y_pred + epsilon) + \
                        (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        
        cross_entropy = -negative_cross_entropy
       
        # 对所有类别求和
        loss = tf.reduce_sum(cross_entropy, axis = -1)
        
        return loss


class TooDNetKeypointLoss(tf.losses.Loss):
    """
    关键点损失 (Keypoint Loss)
    
    用于4个角点的回归损失
    使用L2距离，并根据边界框面积进行归一化
    
    损失计算：
        - 计算预测角点与真实角点的L2距离
        - 根据边界框面积归一化（大目标容忍更大的误差）
        - 使用指数函数将距离转换为损失值
    """

    def __init__(self): 
        super(TooDNetKeypointLoss, self).__init__(
            reduction="none", name = "TooDNetKeypointLoss"
        )

        # 关键点损失的超参数（控制损失曲线的陡峭程度）
        self.__sigmas = 0.025

    def call(self, y_true, y_pred):
        """
        y_true: [gts_box(4), gts_keypoint(8)] - (B, N, 12)
        y_pred: 预测的关键点坐标 (B, N, 8)
        """
        print("TooDNetKeypointLoss y_true.shape", y_true.shape)
        print("TooDNetKeypointLoss y_pred.shape", y_pred.shape) 
        
        gts_box      = y_true[:, :, :4]  # 边界框
        gts_keypoint = y_true[:, :, 4:]  # 4个关键点坐标 (8维)

        print("TooDNetKeypointLoss gts_box.shape", gts_box.shape)
        print("TooDNetKeypointLoss gts_keypoint.shape", gts_keypoint.shape)

        # 计算边界框面积（用于归一化）
        gts_box_x1, gts_box_y1, gts_box_x2, gts_box_y2 = tf.unstack(gts_box, axis = -1) 
        area = (gts_box_x2 - gts_box_x1) * (gts_box_y2 - gts_box_y1)  # (B, N)

        # 分离4个关键点
        # y_pred.shape (batch_size, anchors_num, 8) 
        # gts_keypoint.shape (batch_size, anchors_num, 8)
        pred_k1, pred_k2, pred_k3, pred_k4 = tf.split(y_pred, num_or_size_splits=4, axis = -1)
        gts_k1,  gts_k2,  gts_k3,  gts_k4  = tf.split(gts_keypoint, num_or_size_splits=4, axis = -1)
        # pred_k1.shape (batch_size, anchors_num, 2) - 第一个角点的(x,y)
        # gts_k1.shape  (batch_size, anchors_num, 2) 
        print("pred_k1.shape", pred_k1.shape)
        print("pred_k2.shape", pred_k2.shape)
        print("pred_k3.shape", pred_k3.shape)
        print("pred_k4.shape", pred_k4.shape)
        print("gts_k1.shape",  gts_k1.shape)
        print("gts_k2.shape",  gts_k2.shape)
        print("gts_k3.shape",  gts_k3.shape)
        print("gts_k4.shape",  gts_k4.shape)
        
        d1 = tf.reduce_sum((pred_k1 - gts_k1)**2, axis = -1)
        d2 = tf.reduce_sum((pred_k2 - gts_k2)**2, axis = -1)
        d3 = tf.reduce_sum((pred_k3 - gts_k3)**2, axis = -1)
        d4 = tf.reduce_sum((pred_k4 - gts_k4)**2, axis = -1) 
        #d1.shape (batch_size, anchors_num) 
        e1 = d1 / (tf.math.pow(2 * self.__sigmas, 2) * (area + 1e-9) * 2)
        e2 = d2 / (tf.math.pow(2 * self.__sigmas, 2) * (area + 1e-9) * 2)
        e3 = d3 / (tf.math.pow(2 * self.__sigmas, 2) * (area + 1e-9) * 2)
        e4 = d4 / (tf.math.pow(2 * self.__sigmas, 2) * (area + 1e-9) * 2) 
        #e1.shape (batch_size, anchors_num)        
        f1 = 1 - tf.math.exp(-e1)
        f2 = 1 - tf.math.exp(-e2)
        f3 = 1 - tf.math.exp(-e3)
        f4 = 1 - tf.math.exp(-e4)

        return (f1 + f2 + f3 + f4) / 4.0 

class TooDNetBoxLoss(tf.losses.Loss):
    """
    边界框损失 (SIoU Loss)
    
    使用SIoU (Smooth IoU) 损失函数，包含：
        - IoU损失
        - 角度损失 (angle_cost): 考虑边界框的角度对齐
        - 距离损失 (dist_cost): 考虑中心点距离
        - 形状损失 (shape_cost): 考虑宽高比
    
    相比传统IoU，SIoU能更好地处理边界框的角度和形状差异
    """
    
    def __init__(self):
        super(TooDNetBoxLoss, self).__init__(
            reduction = "none", name = "TooDNetBoxLoss"
        )
        
    def call(self, y_true, y_pred):
        """
        y_true.shape (B, anchors_num, 4) - 真实边界框 [x1, y1, x2, y2]
        y_pred.shape (B, anchors_num, 4) - 预测边界框 [x1, y1, x2, y2]
        """
        boxes1_x1, boxes1_y1,  boxes1_x2,  boxes1_y2 = tf.unstack(y_true, axis = -1)
        boxes2_x1, boxes2_y1,  boxes2_x2,  boxes2_y2 = tf.unstack(y_pred, axis = -1)
        
        boxes1_w = boxes1_x2 - boxes1_x1
        boxes1_h = boxes1_y2 - boxes1_y1
        
        boxes2_w = boxes2_x2 - boxes2_x1
        boxes2_h = boxes2_y2 - boxes2_y1
        
        boxes1_center_x = (boxes1_x1 + boxes1_x2) / 2
        boxes1_center_y = (boxes1_y1 + boxes1_y2) / 2
        
        boxes2_center_x = (boxes2_x1 + boxes2_x2) / 2
        boxes2_center_y = (boxes2_y1 + boxes2_y2) / 2
        
        lu_x1 = tf.maximum(boxes1_x1, boxes2_x1)
        lu_y1 = tf.maximum(boxes1_y1, boxes2_y1)
        rd_x2 = tf.minimum(boxes1_x2, boxes2_x2)
        rd_y2 = tf.minimum(boxes1_y2, boxes2_y2)
        
        intersection_w = tf.maximum(0.0, rd_x2 - lu_x1)
        intersection_h = tf.maximum(0.0, rd_y2 - lu_y1)
            
        intersection_area = intersection_w * intersection_h
    
        area1 = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
        area2 = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
        union_area = area1 + area2 - intersection_area
    
        iou = tf.clip_by_value(intersection_area / (union_area + 1e-8), 0.0, 1.0)
        
        lu1_x1 = tf.minimum(boxes1_x1, boxes2_x1)
        lu1_y1 = tf.minimum(boxes1_y1, boxes2_y1)
        rd1_x2 = tf.maximum(boxes1_x2, boxes2_x2)
        rd1_y2 = tf.maximum(boxes1_y2, boxes2_y2)
        enclosing_w = rd1_x2 - lu1_x1
        enclosing_h = rd1_y2 - lu1_y1
        
        #enclosing_w 和 enclosing_h 不参与梯度更新
        enclosing_area = tf.pow(enclosing_w, 2) + tf.pow(enclosing_h, 2)
        
        center_squared_distance = tf.math.pow(boxes1_center_x - boxes2_center_x, 2) + tf.math.pow(boxes1_center_y - boxes2_center_y, 2)

        #TODO SIoU
        center_h = tf.maximum(boxes1_center_y, boxes2_center_y) - tf.minimum(boxes1_center_y, boxes2_center_y)
        center_w = tf.maximum(boxes1_center_x, boxes2_center_x) - tf.minimum(boxes1_center_x, boxes2_center_x)
        
        alpha_1 = center_h / (tf.sqrt(center_squared_distance) + 1e-8)
        
        alpha_2  = center_w / (tf.sqrt(center_squared_distance) + 1e-8)
        
        threshold = 0.702  # 1/sqrt(2)
        
        alpha = tf.where(alpha_1 > threshold, alpha_2, alpha_1)
        
        angle_cost = tf.math.cos(2 * tf.math.asin(alpha) - math.pi / 2)
        
        #dist_cost
        rho_x = ((boxes1_center_x - boxes2_center_x)/ (enclosing_w + 1e-8))**2
        rho_y = ((boxes1_center_y - boxes2_center_y)/ (enclosing_h + 1e-8))**2
        
        gamma = angle_cost - 2.0
        
        dist_cost = (1.0 - tf.math.exp(gamma * rho_x)) + (1.0 - tf.math.exp(gamma * rho_y))
        
        #shape_cost
        theta = 2
        omega_w = tf.math.abs(boxes1_w - boxes2_w) / (tf.maximum(boxes1_w, boxes2_w) + 1e-8)
        omega_h = tf.math.abs(boxes1_h - boxes2_h)/  (tf.maximum(boxes1_h, boxes2_h) + 1e-8)
        
        shape_cost = tf.math.pow(1 - tf.math.exp(-1.0 * omega_w), theta) + \
                     tf.math.pow(1 - tf.math.exp(-1.0 * omega_h), theta)
        
        loss = 1 - iou + (shape_cost + dist_cost) / 2.0
        
        #(B, anchor_nums)
        return loss
        
class TooDNetLoss(tf.losses.Loss):
    """
    总损失函数
    
    组合多个损失：
        1. 分类损失 (Score Loss): BCE
        2. 边界框损失 (Box Loss): SIoU
        3. 分布焦点损失 (DFL Loss): 用于边界框回归
        4. 关键点损失 (Keypoint Loss): 用于4个角点回归
    
    正负样本分配：
        使用TAL (Task Alignment Learning) 策略
        - 选择gt框内的anchors
        - 每个gt选择top-k (k=13) 个最佳匹配的anchors
    
    损失权重（YOLO8默认）:
        - alpha = 0.5: 分类损失权重
        - beta = 7.5: 边界框损失权重
        - gamma = 1.5: DFL损失权重
        - delta = 8.0: 关键点损失权重
    """
    
    def __init__(self, num_classes = 3, alpha = 0.25, gamma = 2.0):
        
        super(TooDNetLoss, self).__init__(reduction = 'auto', name = 'TooDNetLoss')
        
        self.__num_classes = num_classes

        self.__reg_max     = 128  # DFL距离分布最大值
        self.__box_loss = TooDNetBoxLoss()  # SIoU损失
        self.__score_loss = TooDNetScoreLoss()  # 分类损失
        self.__dfl_loss   = TooDNetDflLoss(reg_max = self.__reg_max)  # DFL损失
        self.__keypoint_loss = TooDNetKeypointLoss()  # 关键点损失
        # 正负样本分配策略 (TAL)
        self.__assign = TalAssign()

    def compute_iou(self, boxes1, boxes2):
        
        boxes1_x1, boxes1_y1,  boxes1_x2,  boxes1_y2 = tf.unstack(boxes1, axis = -1)
        boxes2_x1, boxes2_y1,  boxes2_x2,  boxes2_y2 = tf.unstack(boxes2, axis = -1)
        
        lu_x1 = tf.maximum(boxes1_x1, boxes2_x1)
        lu_y1 = tf.maximum(boxes1_y1, boxes2_y1)
        rd_x2 = tf.minimum(boxes1_x2, boxes2_x2)
        rd_y2 = tf.minimum(boxes1_y2, boxes2_y2)
        
        intersection_w = tf.maximum(0.0, rd_x2 - lu_x1)
        intersection_h = tf.maximum(0.0, rd_y2 - lu_y1)
            
        intersection_area = intersection_w * intersection_h
    
        area1 = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
        area2 = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
        union_area = area1 + area2 - intersection_area
        
        iou = tf.clip_by_value(intersection_area / (union_area + 1e-8), 0.0, 1.0)
        
        return iou
        
    def call(self, y_true, y_pred):
        
        y_pred = tf.cast(y_pred, dtype = tf.float32)
        
        batch_size = tf.shape(y_pred)[0]
        anchors_num = tf.shape(y_pred)[1]
        
        gts_box = y_true[:,:,:4]

        gts_keypoint = y_true[:, :, 4:12]
        
        gts_score = tf.one_hot(
            tf.cast(y_true[:,:,12], dtype = tf.int32),
            depth = self.__num_classes,
            dtype = tf.float32,
        )

        anchors_center, anchors_stride = generate_anchor_centers(batch_size, anchors_num, 1.0)
        
        anchors_score = y_pred[:, :, :self.__num_classes]
        
        anchors_distance_dist = y_pred[:, :, self.__num_classes: self.__num_classes + 4 * self.__reg_max]

        anchors_keypoint_dist = y_pred[:, :, self.__num_classes + 4 * self.__reg_max:]

        # print("anchors_center.shape", anchors_center.shape)
        # print("anchors_score.shape", anchors_score.shape)
        # print("anchors_dist.shape", anchors_dist.shape)
        
        anchors_box      = bbox_decode(anchors_center, anchors_distance_dist)

        anchors_keypoint = keypoint_decode(anchors_center, anchors_keypoint_dist)

        
        # print("anchors_box.shape", anchors_box.shape)
        
    
        #TODO 是否引入initial_assign 正负样本匹配机制（ATS）？
        matched_gts_box, matched_gts_score, matched_gts_keypoint, matched_metrics_val, positive_mask, matched_gts_obj = self.__assign.solve(anchors_score, 
                                                                               gts_score,
                                                                               anchors_box,
                                                                               gts_box,
                                                                               anchors_keypoint,
                                                                               gts_keypoint,
                                                                               anchors_center,
                                                                               anchors_stride,
                                                                               None)
        
        hard_matched_gts_score = matched_gts_score
        soft_matched_gts_score = matched_gts_score * matched_metrics_val[:, :, None]
        matched_gts_obj        = tf.cast(matched_gts_obj, dtype = tf.float32)
        
        
        score_loss = self.__score_loss(soft_matched_gts_score, anchors_score)
        box_loss   = self.__box_loss(matched_gts_box,     anchors_box)

        print("matched_gts_box.shape", matched_gts_box.shape)
        print("matched_gts_keypoint.shape", matched_gts_keypoint.shape) 

        matched_tmp = tf.concat([matched_gts_box, matched_gts_keypoint], axis = -1) 
        #matched_gts_box.shape       (batch_size, anchors_num, 4) 
        #matched_gts_keypoint.shape (batch_size, anchors_num, 8)
        print("matched_tmp.shape", matched_tmp.shape)
        
        keypoint_loss = self.__keypoint_loss(matched_tmp, anchors_keypoint) 

        #keypoint_loss.shape (batch_size, anchors_num)
        
        #生成目标dist
        x0y0, x1y1 = tf.split(matched_gts_box, num_or_size_splits = 2, axis = -1)
        targets_distance_dist = tf.concat([anchors_center - x0y0, x1y1 - anchors_center], axis = -1)
        
        c = tf.shape(anchors_distance_dist)[-1]
        #anchors_dist.shape (batch_size, anchors_num, 4, 26)
        anchors_distance_dist = tf.reshape(anchors_distance_dist, (-1, anchors_num, 4, c // 4))
        #targets_dist.shape (batch_size, anchors_num, 4)
        dfl_loss   = self.__dfl_loss(targets_distance_dist, anchors_distance_dist)


        # print("score_loss", score_loss)
        # print("box_loss",   box_loss)
        # print("dfl_loss",   dfl_loss)
        
        positive_anchors_box = anchors_box[positive_mask]
        positive_matched_gts_box = matched_gts_box[positive_mask]
        
        positive_anchors_center = anchors_center[positive_mask]
        positive_anchors_stride = anchors_stride[positive_mask]
        
        positive_anchors_score = anchors_score[positive_mask]
        positive_matched_gts_soft_score = soft_matched_gts_score[positive_mask]
        positive_matched_gts_hard_score = hard_matched_gts_score[positive_mask]
        
        iou = self.compute_iou(positive_anchors_box, positive_matched_gts_box)


        print("=============================================================================")
        print("TooDNetLoss positive_anchors_box.shape", positive_anchors_box.shape)
        print("TooDNetLoss positive_matched_gts_box.shape", positive_matched_gts_box.shape)
        print("TooDNetLoss iou", iou)
        print("=============================================================================")
#         print("positive_anchors_score", positive_anchors_score)
#         print("positive_matched_gts_score", positive_matched_gts_score)
        #positive_matched_gts_score.shape (B, postive_anchors_num, 40)
        print("positive_anchors_score: max-value", tf.reduce_max(positive_anchors_score, axis = 1))
        print("positive_anchors_score: max-index", tf.argmax(positive_anchors_score, axis = 1))
        print("soft positive_matched_gts_score: value", tf.reduce_max(positive_matched_gts_soft_score, axis = 1))
        print("soft positive_matched_gts_score: index", tf.argmax(positive_matched_gts_soft_score, axis = 1))
        print("hard positive_matched_gts_score: value", tf.argmax(positive_matched_gts_hard_score, axis = 1))
        #只关心正样本
        box_loss      = tf.where(positive_mask, box_loss, 0.0)
        dfl_loss      = tf.where(positive_mask, dfl_loss, 0.0)
        keypoint_loss = tf.where(positive_mask, keypoint_loss, 0.0)
#         score_loss = tf.where(positive_mask, score_loss, 0.0)
        #box_loss weight
        box_loss   = box_loss * matched_metrics_val

        dfl_loss   = dfl_loss * matched_metrics_val
        #box_loss.shape (B, anchors_num)
        #score_loss.shape (B, anchors_num)
#         normalizer = tf.reduce_sum(tf.cast(positive_mask, dtype = tf.float32), axis = -1)

        #matched_metrics_val.shape (B, anchors_num)
        normalizer = tf.reduce_sum(tf.where(positive_mask, matched_metrics_val, 0.0), axis = -1)
        normalizer = tf.maximum(normalizer, 1.0)
        
        score_loss = tf.math.divide_no_nan(tf.reduce_sum(score_loss, axis = -1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis = -1), normalizer)
        dfl_loss = tf.math.divide_no_nan(tf.reduce_sum(dfl_loss, axis = -1), normalizer)

        keypoint_loss = tf.reduce_sum(keypoint_loss, axis = -1)
    
        #YOLO8 默认超参数
        alpha = 0.5
        beta = 7.5
        gamma = 1.5

        delta = 8.0

        # print("score_loss", score_loss)
        # print("box_loss",  box_loss)
        # print("dfl_loss",    dfl_loss)
        loss = alpha * score_loss + beta * box_loss + gamma * dfl_loss + delta * keypoint_loss
        
        return loss 

#TODO 是否多GPU 训练
#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():
    #model = Detector()

model = Detector()
#初始化损失函数
loss_fn = TooDNetLoss() 
#初始化optimizer
# optimizer = tf.optimizers.Adam(learning_rate = 1e-3)

#Build the Model (Optional but Recommended): For subclassed models, 
#it's often necessary to call the build method or run a dummy input 
#through the model to create its weights before loading. 
#This ensures the model's layers are initialized and have defined weights to which the loaded weights can be assigned.
dummy_img = np.random.randn(1, 640, 640, 3)
model(dummy_img)

#加载权重
checkpoint_filepath = "checkpoint/obj1/"

checkpoint_weight_filepath = os.path.join(checkpoint_filepath, "weights_plate_detector_mobileNetLarge_640x640_reg128_keypoint_v1.weights.h5")

#加载上一次训练的模型
if False and os.path.exists(checkpoint_filepath):
    model.load_weights(checkpoint_weight_filepath)
    print("load model weight ... " + checkpoint_weight_filepath)

'''
使用adamW 优化器
'''
learning_rate =  0.000001
weight_decay  = 0.006
weight_decay = 5e-4
optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay, #clipnorm=1
    )

model.compile(loss = loss_fn, 
              optimizer = optimizer, 
              run_eagerly = False, 
              # run_eagerly= True,
              jit_compile = False,
             )
np.set_printoptions(precision = 4, threshold = 1e6)


### np.set_printoptions(threshold = 1e40)

warmup_steps = 10
initial_warmup_learning_rate = 0.001
initial_start_learning_rate =  0.000001
decay_steps = 200

epochs = decay_steps + warmup_steps

checkpoint_filepath = "checkpoint/obj1/"

checkpoint_weight_filepath = os.path.join(checkpoint_filepath, "weights_efficientNetV2B0_640x640_multi_PANet_hardswish_v1.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(checkpoint_filepath, "weights" + "_epoch_{epoch}"),
        filepath = os.path.join(checkpoint_filepath, "weights_efficientNetV2B0_640x640_multi_PANet_hardswish_v1.weights.h5"),
        monitor='loss',
        save_best_only=True,
        save_weights_only=True,
        verbose = 1,
)

print("checkpoint_weight_filepath", checkpoint_weight_filepath)
#加载上一次训练的模型
if False and os.path.exists(checkpoint_filepath):
    model.load_weights(checkpoint_weight_filepath)
    print("load model weight ...")
    
def scheduler(epoch, lr):
    #前100epoch 1e-4 到 1e-3
    if epoch < warmup_steps:
        lr = initial_start_learning_rate + (epoch / warmup_steps) * (initial_warmup_learning_rate - initial_start_learning_rate)
        return lr
    #后200epoch CosineDecay 1e-3 
    else:
#         step = min(epoch, decay_steps) - warmup_steps
        step = epoch - warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
#         decayed = (1 - alpha) * cosine_decay + alpha
#         lr = initial_warmup_learning_rate * decayed
#         eta_min = 2e-10
        eta_min =  1e-8
        lr = (initial_warmup_learning_rate - eta_min) * cosine_decay + eta_min
        return lr


learningRateSchedulercallback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)


earlyStopcallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience = 50, verbose = 1, mode = "min")
#加入tensorboard
# logdir="logs/fit/x"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(
    #reshuffle 重排序数据
     x = train_dataset,
     # validation_data = test_dataset,
     epochs = epochs,
#      epochFs = 1,
     verbose = 1,
     # callbacks = [checkpoint_callback, earlyStopcallback, learningRateSchedulercallback ]
     # callbacks = [checkpoint_callback, learningRateSchedulercallback]
    callbacks =  [learningRateSchedulercallback],
)



