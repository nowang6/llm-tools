"""
车牌OCR识别模型训练脚本
功能：
1. 车牌字符识别（OCR）
2. 车牌文字颜色分类
模型架构：CRNN（Convolutional Recurrent Neural Network）
"""

# ==================== GPU配置 ====================
# 指定使用的GPU设备（使用GPU 3）
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少TensorFlow日志输出

# ==================== 导入库 ====================
import logging
import tensorflow as tf
import os
import math
import numpy as np
# import matplotlib.pyplot as plt
logger = tf.get_logger()
logger.setLevel(logging.ERROR)  # 设置日志级别为ERROR，减少输出信息


# ==================== 自定义激活函数 ====================
class HardSwish(tf.keras.layers.Layer):
    """
    HardSwish激活函数
    HardSwish(x) = x * ReLU6(x + 3) / 6
    这是MobileNetV3中使用的激活函数，相比Swish计算效率更高
    """
    
    def __init__(self):
        super(HardSwish, self).__init__()
    
    def call(self, x, training = True):
        # HardSwish实现：x * (ReLU6(x + 3) / 6)
        r = x * (tf.nn.relu6(x + 3.0) / 6.0)
        return r

# ==================== 骨干网络模块 ====================
class FusedMBConv(tf.keras.layers.Layer):
    """
    FusedMBConv模块（Fused Mobile Inverted Bottleneck Convolution）
    这是EfficientNetV2中使用的模块，融合了MBConv的深度可分离卷积和普通卷积
    相比MBConv，FusedMBConv在早期层使用标准3x3卷积，减少计算量
    
    参数：
        in_channel: 输入通道数
        out_channel: 输出通道数
        kernel_size: 卷积核大小（3或5）
        activation: 激活函数类型（'relu'或'hard_swish'）
        se_ratio: SE（Squeeze-and-Excitation）注意力机制的比例（0表示不使用）
        expand_channel: 扩展通道数（通常大于in_channel）
        strides: 步长
    """
    
    def __init__(self, name = "FusedMBConv", 
                       in_channel = 128,
                       out_channel = 128, 
                       kernel_size = 3,
                       activation = 'relu',
                       se_ratio = 0.0,
                       expand_channel = 100,
                       strides = (1,1),
                       **kwargs):
        super(FusedMBConv, self).__init__(name = name, **kwargs)

        self.in_channel   = in_channel
        self.activation   = activation
        self.se_ratio     = se_ratio  # SE注意力机制的比例
        self.expand_channel = expand_channel  # 扩展通道数
        self.se_channel  = max(1, int(self.expand_channel * se_ratio))  # SE模块的通道数
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.strides     = strides

        # 扩展卷积层（fused conv）
        self.conv3x3    = tf.keras.layers.Conv2D(
                                filters = self.expand_channel,
                                kernel_size = self.kernel_size,
                                strides      = self.strides,
                                use_bias =  False,
                                padding = 'same')
        self.bn1        = tf.keras.layers.BatchNormalization()
        self.act1       = HardSwish()  # HardSwish激活
        self.act11      = tf.keras.layers.Activation("relu6")  # ReLU6激活
        
        # SE（Squeeze-and-Excitation）注意力机制模块
        self.se_pool    = tf.keras.layers.GlobalAveragePooling2D(
                                    data_format = 'channels_last',
                                    keepdims    = True)  # Squeeze：全局平均池化
        self.se_conv1x1_1 = tf.keras.layers.Conv2D(
                                filters = self.se_channel,  # 降维
                                kernel_size = 1,
                                strides      = 1,
                                use_bias =  False,
                                padding = 'same')
        self.se_act_1   = tf.keras.layers.Activation("relu6")

        self.se_conv1x1_2 = tf.keras.layers.Conv2D(
                                filters = self.expand_channel,  # 恢复到原始通道数
                                kernel_size = 1,
                                strides      = 1,
                                use_bias =  False,
                                padding = 'same')
        
        self.se_act_2  = tf.keras.layers.Activation("hard_sigmoid")  # 生成权重

        # 输出投影层（1x1卷积降维）
        self.out_conv  = tf.keras.layers.Conv2D(
                                filters = self.out_channel,
                                kernel_size = 1,
                                use_bias = False,
                                padding = 'same')

        self.out_bn   = tf.keras.layers.BatchNormalization()

    def call(self, x, training = True):
        """
        前向传播流程：
        1. Fused卷积 + BN + 激活
        2. SE注意力机制（可选）
        3. 输出投影（1x1卷积降维）+ BN
        4. 残差连接（如果满足条件）
        """
        # Step 1: Fused卷积扩展
        x1 = self.conv3x3(x)
        x1 = self.bn1(x1, training=training)
        # 根据配置选择激活函数
        if self.activation == 'hard_swish':
            x1 = self.act1(x1)
        else:
            x1 = self.act11(x1)
            
        # Step 2: SE注意力机制（如果启用）
        if 0 < self.se_ratio <= 1:
            se = self.se_pool(x1)  # 全局平均池化
            se = self.se_conv1x1_1(se)  # 降维
            se = self.se_act_1(se)  # ReLU6
            se = self.se_conv1x1_2(se)  # 恢复维度
            se = self.se_act_2(se)  # Hard Sigmoid生成权重
            x1  = x1 * se  # 应用注意力权重

        # Step 3: 输出投影
        x3 = self.out_conv(x1)
        x3 = self.out_bn(x3, training=training)

        # Step 4: 残差连接（当步长为1且输入输出通道相同时）
        if self.strides == (1,1) and self.in_channel == self.out_channel:
            x3 = x3 + x

        return x3

# ==================== 骨干网络 ====================
class BackboneLarge(tf.keras.Model):
    """
    大型骨干网络
    基于FusedMBConv模块构建的特征提取网络，用于车牌OCR识别
    图像预处理：将像素值从[0, 255]归一化到[-1, 1]
    """
    
    def __init__(self, name = "BackboneLarge", **kwargs):
         
        super(BackboneLarge, self).__init__(name, **kwargs)

        # 图像预处理：归一化到[-1, 1]范围
        self.preprocess  = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)

        self.conv3x3    = tf.keras.layers.Conv2D(
                                filters = 16,
                                kernel_size = 3,
                                strides      = (2, 2),
                                use_bias =  False,
                                padding = 'same')
        self.bn        = tf.keras.layers.BatchNormalization()
        self.act       = HardSwish()
         
        self.fused_mbConv1 = FusedMBConv(
                        in_channel = 16,
                        expand_channel = 16,
                        out_channel = 16,
                        kernel_size = 3,
                        se_ratio = 0,
                        strides = (1, 1),
                        activation = 'relu',
                       )
         
        self.fused_mbConv2 = FusedMBConv(
                         in_channel = 16,
                         expand_channel = 64,
                         out_channel = 24,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (2, 1),
                         activation = 'relu',
                       )

        self.fused_mbConv3 = FusedMBConv(
                         in_channel = 24,
                         expand_channel = 72,
                         out_channel = 24,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (1, 1),
                         activation = 'relu',
                        )
        self.fused_mbConv4 = FusedMBConv(
                         in_channel = 24,
                         expand_channel = 72,
                         out_channel = 40,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (2, 1),
                         activation = 'relu',
                        )
        
        self.fused_mbConv5 = FusedMBConv(
                         in_channel = 40,
                         expand_channel = 120,
                         out_channel = 40,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'relu',
                        )

        self.fused_mbConv6 = FusedMBConv(
                         in_channel = 40,
                         expand_channel = 120,
                         out_channel = 40,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'relu',
                        )
        
        self.fused_mbConv7 = FusedMBConv(
                         in_channel  = 40, 
                         expand_channel = 240,
                         out_channel = 80,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )
        self.fused_mbConv8 = FusedMBConv(
                         in_channel  = 80, 
                         expand_channel = 200,
                         out_channel = 80,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )
        
        self.fused_mbConv9 = FusedMBConv(
                         in_channel  = 80, 
                         expand_channel = 184,
                         out_channel = 80,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )
        self.fused_mbConv10 = FusedMBConv(
                         in_channel  = 80, 
                         expand_channel = 184,
                         out_channel = 80,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )
        self.fused_mbConv11 = FusedMBConv(
                         in_channel  = 80, 
                         expand_channel = 480,
                         out_channel = 112,
                         kernel_size = 3,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )

        self.fused_mbConv12 = FusedMBConv(
                         in_channel  = 112, 
                         expand_channel = 672,
                         out_channel = 112,
                         kernel_size = 3,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )

        self.fused_mbConv13 = FusedMBConv(
                         in_channel  = 112, 
                         expand_channel = 672,
                         out_channel = 160,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (2, 1),
                         activation = 'hardswish',
                       )    
        
        self.fused_mbConv14 = FusedMBConv(
                         in_channel  = 160, 
                         expand_channel = 960,
                         out_channel = 160,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )   
        
        self.fused_mbConv15 = FusedMBConv(
                         in_channel  = 160, 
                         expand_channel = 960,
                         out_channel = 160,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )   
        # 最后的特征提取层
        self.last_conv1x1 = tf.keras.layers.Conv2D(
                            filters = 512,  # 扩展到512通道
                            kernel_size = 3,
                            strides      = (1, 1),
                            use_bias =  False,
                            padding = 'same')
        self.last_act   = HardSwish()

        self.last_bn     = tf.keras.layers.BatchNormalization()

        # 最大池化层：进一步下采样
        self.las_max_pool  = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))

    def call(self, image, training = True):
        """
        前向传播：提取图像特征
        输入：(batch_size, height, width, 3)
        输出：(p1, x) - p1为中间特征，x为最终特征
        """
        # 图像预处理：归一化
        x = self.preprocess(image)
        # 可选：转置操作（如果需要交换宽高）
        # x = tf.transpose(x, (0, 2, 1, 3))  # (B, W, H, C)
        
        print("image.shape", image.shape)
        # 初始卷积
        x = self.conv3x3(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        print("x.shape", x.shape)
        p1 = x  # 保存中间特征（用于后续多尺度特征融合）

        x = self.fused_mbConv1(x) #
        print("x.shape mbconv1", x.shape)
        x = self.fused_mbConv2(x) # 
        print("x.shape mbconv2", x.shape)
        x = self.fused_mbConv3(x) # /8
        print("x.shape mbconv3", x.shape)
        x = self.fused_mbConv4(x)
        print("x.shape mbconv4", x.shape)
        x = self.fused_mbConv5(x)
        print("x.shape mbconv5", x.shape)
        x = self.fused_mbConv6(x)
        print("x.shape mbconv6", x.shape)
        x = self.fused_mbConv7(x)
        print("x.shape mbconv7", x.shape)
        x = self.fused_mbConv8(x)
        print("x.shape mbconv8", x.shape)
        x = self.fused_mbConv9(x)
        print("x.shape mbconv9", x.shape)
        x = self.fused_mbConv10(x)
        print("x.shape mbconv10", x.shape)
        x = self.fused_mbConv11(x)
        print("x.shape mbconv11", x.shape)
        x = self.fused_mbConv12(x)
        print("x.shape mbconv12", x.shape)
        x = self.fused_mbConv13(x)
        print("x.shape mbconv13", x.shape)
        x = self.fused_mbConv14(x)
        print("x.shape mbconv14", x.shape)
        x = self.fused_mbConv15(x)
        print("x.shape mbconv15", x.shape)

        x = self.last_conv1x1(x)
        x = self.last_bn(x)
        x = self.last_act(x)

        print("x.shape last", x.shape)
        
        x = self.las_max_pool(x)

        print("x.shape max_pool", x.shape)

        return (p1, x)

class Backbone(tf.keras.Model):
    """
    标准骨干网络（当前使用的版本）
    相比BackboneLarge更轻量，适合实际部署
    使用11个FusedMBConv模块进行特征提取
    """
    
    def __init__(self, name = "Backbone", **kwargs):
         
        super(Backbone, self).__init__(name, **kwargs)

        # 图像预处理：归一化到[-1, 1]
        self.preprocess  = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)

        self.conv3x3    = tf.keras.layers.Conv2D(
                                filters = 16,
                                kernel_size = 3,
                                strides      = (2, 2),
                                use_bias =  False,
                                padding = 'same')
        self.bn        = tf.keras.layers.BatchNormalization()
        self.act       = HardSwish()
         
        self.fused_mbConv1 = FusedMBConv(
                        in_channel = 16,
                        expand_channel = 16,
                        out_channel = 16,
                        kernel_size = 3,
                        se_ratio = 0.25,
                        strides = (1, 1),
                        activation = 'relu',
                       )
         
        self.fused_mbConv2 = FusedMBConv(
                         in_channel = 16,
                         expand_channel = 72,
                         out_channel = 24,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (2, 1),
                         activation = 'relu',
                       )
        self.fused_mbConv3 = FusedMBConv(
                         in_channel  = 24, 
                         expand_channel = 88,
                         out_channel = 24,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (1, 1),
                         activation = 'relu',
                       )

        self.fused_mbConv4 = FusedMBConv(
                         in_channel  = 24, 
                         expand_channel = 96,
                         out_channel = 40,
                         kernel_size = 3,
                         se_ratio = 0,
                         strides = (2, 1),
                         activation = 'hardswish',
                       )
        
        self.fused_mbConv5 = FusedMBConv(
                         in_channel  = 40, 
                         expand_channel = 240,
                         out_channel = 40,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )
        
        self.fused_mbConv6 = FusedMBConv(
                         in_channel  = 40, 
                         expand_channel = 240,
                         out_channel = 40,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )

        self.fused_mbConv7 = FusedMBConv(
                         in_channel  = 40, 
                         expand_channel = 120,
                         out_channel = 48,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )

        self.fused_mbConv8 = FusedMBConv(
                         in_channel  = 48, 
                         expand_channel = 144,
                         out_channel = 48,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )    
        
        self.fused_mbConv9 = FusedMBConv(
                         in_channel  = 48, 
                         expand_channel = 288,
                         out_channel = 96,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (2, 1),
                         activation = 'hardswish',
                       )   
        
        self.fused_mbConv10 = FusedMBConv(
                         in_channel  = 96, 
                         expand_channel = 576,
                         out_channel = 96,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )   
        
        self.fused_mbConv11 = FusedMBConv(
                         in_channel  = 96, 
                         expand_channel = 576,
                         out_channel = 96,
                         kernel_size = 5,
                         se_ratio = 0.25,
                         strides = (1, 1),
                         activation = 'hardswish',
                       )  
        self.last_conv1x1 = tf.keras.layers.Conv2D(
                            filters = 576,
                            kernel_size = 3,
                            strides      = (1, 1),
                            use_bias =  False,
                            padding = 'same')
        self.last_act   = HardSwish()

        self.last_bn     = tf.keras.layers.BatchNormalization()

        self.las_max_pool  = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))
            
    def call(self, image, training = True):
        """
        前向传播：通过多层FusedMBConv提取特征
        """
        # 图像预处理
        x = self.preprocess(image)
        
        print("image.shape", image.shape)
        # 初始卷积层
        x = self.conv3x3(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        print("x.shape", x.shape)
        
        # 通过11个FusedMBConv模块提取特征
        # 注意：某些层使用(2,1)步长，只在高度方向下采样，保持宽度信息（适合OCR）
        x = self.fused_mbConv1(x, training=training)  # 第一个MBConv块
        print("x.shape mbconv1", x.shape)
        x = self.fused_mbConv2(x, training=training)  # 步长(2,1)：高度下采样
        print("x.shape mbconv2", x.shape)
        x = self.fused_mbConv3(x, training=training)  # 残差连接
        print("x.shape mbconv3", x.shape)
        x = self.fused_mbConv4(x, training=training)  # 步长(2,1)：高度下采样
        print("x.shape mbconv4", x.shape)
        x = self.fused_mbConv5(x, training=training)  # SE注意力
        print("x.shape mbconv5", x.shape)
        x = self.fused_mbConv6(x, training=training)  # SE注意力
        print("x.shape mbconv6", x.shape)
        x = self.fused_mbConv7(x, training=training)  # 切换到HardSwish激活
        print("x.shape mbconv7", x.shape)
        x = self.fused_mbConv8(x, training=training)
        print("x.shape mbconv8", x.shape)
        x = self.fused_mbConv9(x, training=training)  # 步长(2,1)：最后一次高度下采样
        print("x.shape mbconv9", x.shape)
        x = self.fused_mbConv10(x, training=training)
        print("x.shape mbconv10", x.shape)
        x = self.fused_mbConv11(x, training=training)
        print("x.shape mbconv11", x.shape)

        # 最后的特征提取
        x = self.last_conv1x1(x)
        x = self.last_bn(x, training=training)
        x = self.last_act(x)
        print("x.shape last", x.shape)

        # 最大池化下采样
        x = self.las_max_pool(x)
        print("x.shape max_pool", x.shape)

        return x

class BackboneV2(tf.keras.Model):
    """
    简化版骨干网络（备用版本）
    使用传统的Conv+MaxPool结构，未使用
    """
    
    def __init__(self, name = "BackboneV2", **kwargs):
         
        super(BackboneV2, self).__init__(name, **kwargs)

        # 不同的归一化方式
        self.preprocess  = tf.keras.layers.Rescaling(scale=1.0 / 128.0, offset=-1)
        # 三层卷积+池化结构
        self.conv3x3_1   = tf.keras.layers.Conv2D(
                                filters = 64,
                                kernel_size = 3,
                                strides      = 1,
                                use_bias =  False,
                                padding = 'same')
        self.maxpool_1   = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))
        self.act_1       = tf.keras.layers.Activation("relu")

        self.conv3x3_2   = tf.keras.layers.Conv2D(
                                filters = 128,
                                kernel_size = 3,
                                strides      = 1,
                                use_bias =  False,
                                padding = 'same')
        self.maxpool_2   = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))
        self.act_2       = tf.keras.layers.Activation("relu")

        self.conv3x3_3   = tf.keras.layers.Conv2D(
                                filters = 256,
                                kernel_size = 3,
                                strides      = 1,
                                use_bias =  False,
                                padding = 'same')
        self.maxpool_3   = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))
        self.act_3       = tf.keras.layers.Activation("relu")

    def call(self, image, training = True):
        """简单的三层卷积+池化结构"""
        x = self.preprocess(image)
        x = self.conv3x3_1(x)
        x = self.maxpool_1(x)
        x = self.act_1(x)

        x = self.conv3x3_2(x)
        x = self.maxpool_2(x)
        x = self.act_2(x)

        x = self.conv3x3_3(x)
        x = self.maxpool_3(x)
        x = self.act_3(x)

        return x


# ==================== CRNN模型（多任务） ====================
class CRNN(tf.keras.Model):
    """
    CRNN（Convolutional Recurrent Neural Network）多任务模型
    同时完成两个任务：
    1. 车牌字符识别（OCR）：识别车牌上的文字序列
    2. 车牌颜色分类：识别车牌文字的颜色（10类）
    
    模型结构：
    - Backbone：特征提取网络
    - Color分支：颜色分类（全局平均池化 + 全连接）
    - OCR分支：序列识别（注意力机制 + 全连接）
    """
    def __init__(self, name = "CRNN" , **kwargs):
        super(CRNN, self).__init__(name, **kwargs)

        self.num_colors = 10  # 颜色类别数（如：蓝、白、黑、黄等）
        self.num_classes = 81 + 1  # OCR字符类别数（81个字符 + 1个空白/填充）
        
        # 选择骨干网络（当前使用Backbone）
        # self.backbone = BackboneV2()
        # self.backbone = BackboneLarge()
        self.backbone = Backbone()  # 标准骨干网络

        # ========== 颜色分类分支 ==========
        self.color_last_conv1x1  = tf.keras.layers.Conv2D(
                                    filters = 256,
                                    kernel_size = 1,
                                    strides      = 1,
                                    use_bias =  False,
                                    padding = 'same')  # 1x1卷积降维
        
        self.color_last_bn  = tf.keras.layers.BatchNormalization()
        self.color_last_act = tf.keras.layers.Activation('relu')
        
        # 全局平均池化：将空间特征聚合为全局特征
        self.color_glob_avg = tf.keras.layers.GlobalAveragePooling2D(data_format = 'channels_last', keepdims = False)
        self.color_dropout  = tf.keras.layers.Dropout(rate = 0.2)  # 防止过拟合
        # 颜色分类输出层
        self.color_ffn      = tf.keras.layers.Dense(units = self.num_colors, name='out_color')
                            
        # ========== OCR字符识别分支 ==========
        # 自注意力机制：用于序列建模
        self.att      = tf.keras.layers.Attention(use_scale=False, score_mode="dot")
        self.ffn1     = tf.keras.layers.Dense(units = 256)  # 中间层
        self.ffn1_drop = tf.keras.layers.Dropout(rate = 0.2)
        self.act      = tf.keras.layers.Activation('relu')
        # OCR分类输出层：每个时间步预测一个字符
        self.ffn2     = tf.keras.layers.Dense(units = self.num_classes, name = 'out_plate')

        self.center_compute = CenterLossLayer()  # 中心损失（当前未使用）

    def call(self, image, training = True):
        """
        前向传播
        输入：image (batch_size, height, width, 3)
        输出：字典包含
            - color_output: 颜色分类logits (batch_size, num_colors)
            - plate_output: OCR分类logits (batch_size, seq_len, num_classes)
            - feats_output: 特征和预测的拼接 (batch_size, seq_len, 256+num_classes)
        """
        # 特征提取
        x = self.backbone(image, training=training)  # (B, H, W, C)
        print("x", x)
        
        # ========== 颜色分类分支 ==========
        x1 = self.color_last_conv1x1(x)
        x1 = self.color_last_bn(x1, training=training)
        x1 = self.color_last_act(x1)
        x1 = self.color_glob_avg(x1)  # (B, 256)
        x1 = self.color_dropout(x1, training=training)
        y1 = self.color_ffn(x1)  # (B, num_colors) 颜色分类输出
        
        # ========== OCR字符识别分支 ==========
        # 将特征图转换为序列格式，便于序列建模
        batch       = tf.shape(x)[0]
        height      = tf.shape(x)[1]
        width       = tf.shape(x)[2]
        channels    = tf.shape(x)[3]
        # 转置：(B, H, W, C) -> (B, W, H, C)，将宽度维度作为时间步
        x2 = tf.transpose(x, (0, 2, 1, 3))
        # 重塑：(B, W, H, C) -> (B, W, H*C)，将高度和通道合并
        x2 = tf.reshape(x2, (batch, width, height * channels))
        
        # 自注意力机制：建模序列中字符之间的关系
        x3 = self.att([x2, x2, x2], training=training)  # query, key, value都是x2
        x3 = x3 + x2  # 残差连接
        # 全连接层
        x4 = self.ffn1(x3)  # (B, W, 256)
        x4 = self.act(x4)
        x4 = self.ffn1_drop(x4, training=training)
        y2 = self.ffn2(x4)  # (B, W, num_classes) OCR分类输出

        # 拼接特征和预测（用于CenterLoss，当前可选）
        feats = tf.concat([x4, y2], axis = -1)  # (B, W, 256+num_classes)

        print("x4", x4.shape)
        print("y2", y2.shape)
        print("feats", feats.shape)

        # 返回多任务输出
        return {"color_output": y1, "plate_output": y2, "feats_output": feats}


# ==================== 数据准备 ====================
import numpy as np

# TFRecord数据文件列表（训练数据）
tf_record_filename = [
                      # "plate_ocr_color_combined_filed_v1.tfrecords", 
                      # "plate_ocr_color_combined_filed_v2.tfrecords",
                      "plate_ocr_color_combined_filed_v3.tfrecords",
                      "plate_ocr_color_combined_v7.tfrecords",]
obj_dataset = tf.data.TFRecordDataset(tf_record_filename)

# TFRecord数据格式定义
obj_feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], dtype = tf.string),  # 图像编码（JPEG/PNG）
    "image/format": tf.io.FixedLenFeature([], dtype = tf.string),  # 图像格式
    "image/height": tf.io.FixedLenFeature([], dtype = tf.int64),  # 图像高度
    "image/width": tf.io.FixedLenFeature([], dtype = tf.int64),  # 图像宽度
    "image/ocr/text": tf.io.FixedLenFeature([], dtype = tf.string),  # OCR文本标签
    'image/color/id': tf.io.FixedLenFeature([], dtype=tf.int64)  # 颜色ID标签（0-9）
}

# ==================== 字符词汇表 ====================
# 定义车牌识别支持的字符集合（英文字母、数字、中文字符）
vocab_data = ['A', 'B',  'C', 'D',  'E', 'F', 'G', 'H', 'I', 'J', 'K',
               'L', 'M',  'N', 'O',  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # 26个字母
               '0',  '1', '2',  '3',  '4',  '5',  '6', '7',  '8', '9',  # 10个数字
              '京', '津', '沪', '渝', '蒙', '新', '桂',   # 省份简称
              '宁', '藏', '澳', '港', '黑', '吉', '辽',
              '晋', '冀', '青', '鲁', '豫', '苏', '皖',
              '浙', '闽', '赣', '湘', '鄂', '粤', '琼',
              '甘', '陕', '贵', '云', '川', '警', '使',   # 特殊车牌标识
              '学', '挂','领', '民', '航', '危', '险', 
              '品']

max_len = 10  # 车牌最大字符长度（中国车牌通常7-8个字符）
# 文本向量化层：将字符序列转换为整数序列
vectorize_layer = tf.keras.layers.TextVectorization(
    ngrams=None,  # 不使用n-gram
    standardize=None,  # 不进行标准化
    output_mode='int',  # 输出整数索引
    output_sequence_length=max_len,  # 输出序列长度固定为max_len
    encoding = "utf-8",
    split = "character",  # 按字符分割
    vocabulary=vocab_data)  # 使用预定义的词汇表

print("vob", vectorize_layer.get_vocabulary())
print("len(vob)", len(vectorize_layer.get_vocabulary()))

vocab = vectorize_layer.get_vocabulary()

# ==================== 数据解析和预处理 ====================
def _parse_obj_function(example_proto):
    """
    解析TFRecord中的单个样本
    将序列化的Example协议缓冲区解析为特征字典
    """
    return tf.io.parse_single_example(example_proto, obj_feature_description)

# 随机数生成器（用于数据增强，固定种子保证可复现）
rng = tf.random.Generator.from_seed(123, alg='philox')

def preprocess_data(obj_features):
    """
    数据预处理函数
    功能：
    1. 解码图像
    2. 文本向量化
    3. 图像resize到固定尺寸
    4. 可选的数据增强
    
    返回：(image, ocr_text, color_id)
        - image: (32, 112, 3) 浮点图像
        - ocr_text: (max_len,) 整数序列
        - color_id: 标量整数
    """
    # 从特征字典中提取数据
    image_str = obj_features['image/encoded']
    image_height  = obj_features['image/height']
    image_width   = obj_features['image/width']
    ocr_text      = obj_features['image/ocr/text']  # 字符串格式的OCR文本
    color_id      = obj_features['image/color/id']  # 颜色ID
    
    # 解码图像（JPEG格式，支持截断恢复）
    # image = tf.io.decode_png(contents = image_str, channels = 3)
    image = tf.io.decode_jpeg(contents = image_str, try_recover_truncated = True, channels = 3)
    
    # 转换为浮点数并重塑形状
    image = tf.cast(image, dtype = tf.float32)
    image = tf.reshape(image, (image_height, image_width, 3))

    # 文本向量化：将字符串转换为整数序列
    ocr_text = vectorize_layer(ocr_text)
    ocr_text = tf.cast(ocr_text, dtype = tf.int32)
    
    # 颜色ID转换
    color_id = tf.cast(color_id, dtype = tf.int32)

    # 随机数生成（用于数据增强，当前未使用）
    sample = rng.uniform(shape = (1,), minval = 0, maxval = 1, name = "random_resize", dtype = tf.float32)
    
    # 图像尺寸设置（车牌图像通常是长条形）
    padding_image_min_side = tf.constant(32, dtype = tf.int32)   # 高度
    padding_image_max_side = tf.constant(112, dtype = tf.int32)  # 宽度

    # 图像resize到固定尺寸（32x112）
    image = tf.image.resize(image, (padding_image_min_side, padding_image_max_side))
    
    # 可选的数据增强（当前注释掉）
    # if sample >= 0.5 and sample < 0.6:
    #     image = tf.image.random_brightness(image, max_delta=0.1)

    # 确保像素值在[0, 255]范围内
    image = tf.clip_by_value(image, 0, 255)
    
    return image, ocr_text, color_id

def batch_encode(batch_images, gts_ocr_text, gts_color_id):
    """
    批次编码函数：将OCR文本和颜色ID合并为标签
    将每个样本的(color_id, ocr_text)拼接成一个标签向量
    
    输入：
        batch_images: (batch_size, 32, 112, 3)
        gts_ocr_text: (batch_size, max_len)
        gts_color_id: (batch_size,)
    
    输出：
        (batch_images, labels)
        labels: (batch_size, 1 + max_len) - [color_id, char1, char2, ..., charN]
    """
    batch_size = tf.shape(batch_images)[0]
    
    labels = tf.TensorArray(dtype=tf.float32, size = batch_size, dynamic_size = True)
        
    for i in range(batch_size):
        gt_ocr_text = gts_ocr_text[i]  # (max_len,)
        gt_color = gts_color_id[i]     # 标量
        
        images = batch_images[i]  # 当前未使用
        
        # 将颜色ID扩展为形状(1,)，然后与OCR文本拼接
        gt_color = tf.expand_dims(gt_color, axis = -1)  # (1,)
        gt_ocr_text = tf.cast(gt_ocr_text, dtype=tf.float32)
        gt_color    = tf.cast(gt_color, dtype=tf.float32)
        
        # 拼接：[color_id, char1, char2, ..., charN]
        label  = tf.concat([gt_color, gt_ocr_text], axis = -1)  # (1 + max_len,)
        
        labels = labels.write(i, label)
            
    return (batch_images, labels.stack())  # labels.stack(): (batch_size, 1 + max_len)
    

# ==================== 数据集构建 ====================
# Step 1: 解析TFRecord
train_parsed_obj_dataset = obj_dataset.map(_parse_obj_function, tf.data.AUTOTUNE)

# Step 2: 可选缓存（当前注释掉，大数据集不建议缓存）
# train_parsed_obj_dataset = train_parsed_obj_dataset.cache()

# Step 3: 数据预处理（解码、resize、向量化）
train_parsed_obj_dataset = train_parsed_obj_dataset.map(preprocess_data, tf.data.AUTOTUNE)

# Step 4: 设置批次大小和shuffle
train_obj_cnt = 1  # 当前未使用（用于限制样本数）
batch_size = 1024 * 4  # 批次大小：4096

train_dataset = train_parsed_obj_dataset

# Step 5: 数据打乱（缓冲区大小为10倍批次大小）
train_dataset = train_dataset.shuffle(10 * batch_size)

# Step 6: 批次化（使用填充以适应不同长度的序列）
# padding_values: (image填充值, ocr_text填充值, color_id填充值)
train_dataset = train_dataset.padded_batch(
    batch_size = batch_size, 
    padding_values = (0.0, 0, 0),  # 图像用0.0填充，文本用0填充
    drop_remainder = False  # 保留最后一个不完整的批次
)

# Step 7: 批次编码（合并标签）
train_dataset = train_dataset.map(batch_encode, tf.data.AUTOTUNE)

# Step 8: 预取数据（提高训练效率）
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# ==================== 工具函数 ====================
def reverse_vectorization(vectorized_sequences, text_vectorizer):
    """
    将向量化的序列转换回原始文本
    用于可视化预测结果
    
    参数：
        vectorized_sequences: 整数序列列表
        text_vectorizer: TextVectorization层
    
    返回：
        original_texts: 字符串列表
    """
    vocabulary = text_vectorizer.get_vocabulary()
    reverse_mapping = {idx: word for idx, word in enumerate(vocabulary)}
    
    original_texts = []
    for sequence in vectorized_sequences:
        # 过滤掉0（填充值）和空白字符，转换为原始字符
        words = [reverse_mapping.get(idx, '') for idx in sequence if idx!=0]
        original_texts.append("".join(words))
    return original_texts


def visualize_detection(image, gts_box, figsize = (2,4), linewidth = 1.5, color = [1, 0, 0] ):
    
    feature_level = 5
    
    image = np.array(image, dtype = np.uint8)
    
    plt.figure(figsize = figsize)
    # plt.grid(visible = False, axis = 'both', linestyle = '-.')
    
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    yticks_num = np.math.ceil(image_height/ 2 ** feature_level)
    xticks_num = np.math.ceil(image_width / 2 ** feature_level)
#     print("xticks_num", xticks_num)
#     print("yticks_num", yticks_num)
    
    xticks = np.linspace(0, image_width, xticks_num)
    yticks = np.linspace(0, image_height, yticks_num)
    
    plt.xticks(xticks, rotation = 'horizontal')
    plt.yticks(yticks)
    
    plt.imshow(image)
    ax = plt.gca()
    
    for box in gts_box:
        
        x1, y1, x2, y2 = box
        
        w = x2 - x1
        h = y2 - y1
        c_x = (x1 + x2) / 2
        c_y = (y1 + y2) / 2
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor = [0,0,1], linewidth = 4
        )
#         patch2 = plt.Circle(
#             [c_x, c_y], radius = 5, fill=True, color = [1,0,0]
#         )
#         ax.add_patch(patch2)
        ax.add_patch(patch)
    
    plt.show()
    
    return ax

class_recs = {}
image_idx = 0

det_data = []

for image, label in train_dataset.take(1):
    #data_format HWC
    image_height = tf.cast(tf.shape(image)[1], dtype = tf.float32)
    image_width  = tf.cast(tf.shape(image)[2], dtype = tf.float32)
    #bbox.shape (batch_size, nums_gt, 4)
    ocr_text = label[:, 1:]
    color_id = label[:, 0:1]
    ocr_text = tf.cast(ocr_text, dtype = tf.int32)
    label = reverse_vectorization(ocr_text.numpy(), vectorize_layer)
    print("label", label)
#     print("label", label)
    # visualize_detection(image[0], [])
    # visualize_detection(image[0], bbox[0])

# ==================== 损失函数 ====================
class OCRLoss(tf.losses.Loss):
    """
    OCR字符识别损失函数
    使用CTC（Connectionist Temporal Classification）损失
    CTC适用于序列标注任务，不需要对齐输入和输出序列
    
    注意：y_true的第一列是color_id，需要跳过
    y_true: (batch_size, 1 + max_len) - [color_id, char1, char2, ...]
    y_pred: (batch_size, seq_len, num_classes) - OCR分类logits
    """
    
    def __init__(self):
        super(OCRLoss, self).__init__(reduction = 'sum_over_batch_size', name = 'OCRLoss')
        
    def call(self, y_true, y_pred):
        # 跳过第一列（color_id），提取OCR文本标签
        target = y_true[:, 1:]  # (batch_size, max_len)
        target = tf.cast(target, dtype = tf.int32)
        output = tf.cast(y_pred, dtype = tf.float32)  # (batch_size, seq_len, num_classes)

        print("target", target[0:1])
        print("output", tf.math.argmax(output[0:1], axis = -2))

        # 计算序列长度（用于CTC）
        batch_length  = tf.cast(tf.shape(target)[0], dtype="int32")
        output_length = tf.cast(tf.shape(output)[1], dtype="int32")  # 预测序列长度
        target_length = tf.cast(tf.shape(target)[1], dtype="int32")  # 标签序列长度
        
        # 扩展到批次大小
        output_length = output_length * tf.ones((batch_length, ), dtype="int32")
        target_length = target_length * tf.ones((batch_length,), dtype="int32")
        
        # CTC损失计算
        loss = tf.nn.ctc_loss(
                    labels = target,  # 标签序列
                    logits = output,  # 预测logits
                    label_length = target_length,  # 标签实际长度
                    logit_length = output_length,  # logits实际长度
                    logits_time_major = False,  # logits不是时间主序（batch在第一维）
                    blank_index = -1,  # 空白字符索引
        )

        print("loss.shape", loss.shape)
        return loss


class ColorFocalLoss(tf.losses.Loss):
    """
    颜色分类的Focal Loss（当前未使用）
    Focal Loss用于解决类别不平衡问题，专注于难分类样本
    
    注意：代码中有个bug - weighting_factort应该是weighting_factor
    """

    def __init__(self):
        super(ColorFocalLoss, self).__init__(reduction = 'sum_over_batch_size', name = 'ColorFocalLoss')

        self.num_colors = 10
        self.gamma      = 2.0  # 聚焦参数，控制难易样本权重
        self.alpha      = 0.25  # 类别权重平衡参数

        self.label_smoothing = 0.0

    def call(self, y_true, y_pred):
        # 将颜色ID转换为one-hot编码
        gts_color = tf.one_hot(
            tf.cast(y_true[:, 0], dtype = tf.int32),
            depth = self.num_colors,
            dtype = tf.float32,
        )

        # 可选：标签平滑
        if self.label_smoothing:
            gts_color = gts_color * (1.0 - self.label_smoothing) + (self.label_smoothing / self.num_colors)

        print("gts_color[0]", gts_color[0:1,:])

        y_pred = tf.nn.softmax(y_pred, axis = -1)
        
        epsilon = tf.constant(1e-7, dtype = tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # 交叉熵
        cce = -gts_color * tf.math.log(y_pred)

        # Focal Loss权重：降低易分类样本的权重
        modulating_factor = tf.math.pow(1.0 - y_pred, self.gamma)
        weighting_factor  = modulating_factor * self.alpha

        focal_cce = weighting_factor * cce  # 修复：weighting_factort -> weighting_factor

        print("focal_cce.shape", focal_cce.shape)

        focal_loss = tf.reduce_sum(focal_cce, axis = -1)

        print("focal_loss.shape", focal_loss.shape)
        return focal_loss

class ColorLoss(tf.losses.Loss):
    """
    颜色分类损失函数（当前使用）
    使用带标签平滑的交叉熵损失
    """
    
    def __init__(self):
        super(ColorLoss, self).__init__(reduction = 'sum_over_batch_size', name = 'ColorLoss')

        self.num_colors = 10
        self.label_smoothing = 0.1  # 标签平滑系数，防止过拟合
        
    def call(self, y_true, y_pred):
        """
        y_true: (batch_size, 1 + max_len) - 第一列是color_id
        y_pred: (batch_size, num_colors) - 颜色分类logits
        """
        gt_color = y_true[:, 0]  # 提取颜色ID
        print("gt_color.shape", gt_color)

        # 转换为one-hot编码
        gts_color = tf.one_hot(
            tf.cast(y_true[:, 0], dtype = tf.int32),
            depth = self.num_colors,
            dtype = tf.float32,
        )

        # 标签平滑：将硬标签转换为软标签
        if self.label_smoothing:
            gts_color = gts_color * (1.0 - self.label_smoothing) + (self.label_smoothing / self.num_colors)

        print("gts_color.shape", gts_color.shape)
        print("y_pred.shape",    y_pred.shape)
        
        # 计算交叉熵损失
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = gts_color, logits = y_pred)
        print("loss.shape", loss.shape)
        return loss 

class CenterLoss(tf.losses.Loss):
    """
    中心损失（Center Loss）- 当前未使用
    用于增强类内紧凑性，提高特征区分度
    通过拉近同类样本特征与类中心的距离来优化特征空间
    """

    def __init__(self):
        super(CenterLoss, self).__init__(reduction = 'sum_over_batch_size', name = "CenterLoss")
        self.feature_len = 256  # 特征维度
        self.num_classes = 81 + 1  # 字符类别数

        # 加载预计算的类中心
        import pickle 
        center_file_path = "train_center.pkl"
        shape = (self.num_classes, self.feature_len)
        print("shape", shape)
        centers = np.zeros(shape, np.float32)
        try:
            with open(center_file_path, "rb") as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    centers[key] = char_dict[key]
        except:
            print("center_file_path not found: ", center_file_path)

        self.centers = tf.convert_to_tensor(centers)

    def call(self, y_true, y_pred):
        """
        y_pred: (batch_size, seq_len, 256 + num_classes) - 特征和预测的拼接
        前256维是特征，后面是分类logits
        """
        print("CenterLoss y_pred", y_pred)

        # 分离特征和预测
        features = y_pred[:, :, :256]  # (B, seq, 256)
        predicts = y_pred[:, :, 256:]  # (B, seq, num_classes)

        print("features.shape", features.shape)
        print("predicts.shape",  predicts.shape)
        
        # 重塑为2D：将批次和时间步合并
        features = tf.reshape(features, (-1, tf.shape(features)[-1]))  # (B * seq, 256)
        print("features.shape", features.shape)
        predicts = tf.reshape(predicts, (-1, tf.shape(predicts)[-1]))  # (B * seq, num_classes)
        print("predicts.shape", predicts.shape)
        
        # 从预测中获取标签（当前时间步预测的类别）
        labels = tf.argmax(predicts, axis = -1)  # (B * seq,)
        print("labels.shape", labels.shape)
        
        # 转换为one-hot编码
        one_hot_labels = tf.one_hot(
                            labels,
                            depth = self.num_classes)  # (B * seq, num_classes)
        print("one_hot_labels.shape", one_hot_labels.shape)
        
        # 根据标签选择对应的类中心
        # centers.shape: (num_classes, feature_len)
        matched_centers_features = tf.matmul(one_hot_labels, self.centers)  # (B * seq, feature_len)

        # 计算特征与类中心的距离
        center_loss = tf.math.square(features - matched_centers_features)  # (B * seq, feature_len)
        center_loss = tf.reduce_sum(center_loss, axis = -1)  # (B * seq,) - 每个样本的损失

        print("center_loss.shape", center_loss.shape)
        return center_loss

# ==================== 模型初始化和配置 ====================
# 创建CRNN模型实例
recognition_model = CRNN()

# 构建模型（对于子类化模型，需要运行一次前向传播来初始化权重）
# 这是加载权重前的必要步骤，确保所有层都已初始化
dummy_img = np.random.randn(1, 32, 112, 3)  # 创建虚拟输入
recognition_model(dummy_img)

# 加载预训练权重（可选）
checkpoint_filepath = "checkpoint/obj1/"
checkpoint_weight_filepath = os.path.join(checkpoint_filepath, "CRNN_48x168_mobileSmallRecFeatureMapResolution_v1-best-3.weights.h5")

# 加载上一次训练的模型权重（当前关闭）
if False and os.path.exists(checkpoint_filepath):
    recognition_model.load_weights(checkpoint_weight_filepath)
    print("load model weight ..." + checkpoint_weight_filepath)

# ==================== 损失函数和优化器 ====================
# 初始化损失函数
ocr_loss_fn = OCRLoss()    # OCR识别损失
color_loss_fn = ColorLoss()  # 颜色分类损失
center_loss_fn = CenterLoss()  # 中心损失（当前未使用）

# 优化器配置：使用AdamW（带权重衰减的Adam）
# AdamW相比Adam能更好地处理权重衰减，通常训练效果更好
learning_rate = 0.00001  # 初始学习率
weight_decay = 1e-5      # 权重衰减系数（L2正则化）
optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        # clipnorm=1  # 可选：梯度裁剪
    )

# loss = {'color_output':color_loss_fn, 'plate_output': ocr_loss_fn}
# loss_weights = {'color_output': 1, 'plate_output': 20}

###########################################################################################################
# loss = {'color_output':color_loss_fn, 'plate_output': ocr_loss_fn, 'feats_output': center_loss_fn}
# loss_weights = {'color_output': 1, 'plate_output': 20, 'feats_output': 1}
############################################################################################################
loss = {'color_output':color_loss_fn, 'plate_output': ocr_loss_fn}
loss_weights = {'color_output': 1, 'plate_output': 20}
#############################################################################################################
# loss = {'color_output':color_loss_fn, 'plate_output': ocr_loss_fn, "plate_center": zero_loss_fn}
# loss_weights = {'color_output': 1, 'plate_output': 20, 'plate_center': 1}
# recognition_model.compile(loss = ocr_loss_fn, 
recognition_model.compile(loss = loss,
              optimizer = optimizer, 
              run_eagerly = False, 
              # run_eagerly= True,
              jit_compile = True,
              loss_weights= loss_weights,
             )
np.set_printoptions(precision = 4, threshold = 1e6)

print(tf.__version__)

### np.set_printoptions(threshold = 1e40)

# epochs = 5 + 5

warmup_steps = 10
initial_warmup_learning_rate = 0.001
initial_start_learning_rate =  0.00001
decay_steps = 100

epochs = warmup_steps + decay_steps

checkpoint_filepath = "checkpoint/obj1/"

checkpoint_weight_filepath = os.path.join(checkpoint_filepath, "CRNN_48x168_mobileSmallRecFeatureMapResolution_v2.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath=os.path.join(checkpoint_filepath, "weights" + "_epoch_{epoch}"),
        filepath = os.path.join(checkpoint_filepath, "CRNN_48x168_mobileSmallRecFeatureMapResolution_v2.weights.h5"),
        monitor='loss',
        save_best_only=True,
        save_weights_only=True,
        verbose = 1,
)

    
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
# 

earlyStopcallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience = 50, verbose = 1, mode = "min")
#加入tensorboard
# logdir="logs/fit/x"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = recognition_model.fit(
    #reshuffle 重排序数据
     x = train_dataset,
     validation_data = val_dataset,
     epochs = epochs,
#      epochFs = 1,
     verbose = 1,
     # callbacks = [checkpoint_callback,  earlyStopcallback, learningRateSchedulercallback ]
     callbacks = [checkpoint_callback, learningRateSchedulercallback]
)

