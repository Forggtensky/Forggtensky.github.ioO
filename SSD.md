# SSD

​		**Created by Hanyz@2021/1/4**



**整体特点：**

- 从YOLO中继承了将detection转化为regression的思路，一次完成目标定位与分类
- 基于Faster RCNN中的Anchor，提出了相似的Prior box；
- 加入基于特征金字塔（Pyramidal Feature Hierarchy）的检测方式，即在不同感受野的feature map上预测目标



## 1、整体网络结构：

下图是Backbone采用resnet50：

![image-20210104102110322](image-20210104102110322.png)



### 特征金字塔的思想：

【实现多尺度的识别与分类】

与同为Single shot方式的SSD/YOLO区别：

- YOLO在卷积层后接全连接层，即检测时只利用了最高层Feature maps（包括Faster RCNN也是如此）
- SSD采用金字塔结构，即利用了conv4-3/conv-7/conv6-2/conv7-2/conv8_2/conv9_2这些大小不同的feature maps，在多个feature maps上同时进行softmax分类和位置回归
- SSD还加入了Prior box

<img src="image-20210104102206033.png" alt="image-20210104102206033" style="zoom: 50%;" />



## 2、Prior Box 

​		在SSD300中引入了Prior Box，实际上与Faster RCNN Anchor非常类似，就是一些目标的预选框，后续通过classification+bounding box regression获得真实目标的位置。

<img src="image-20210104103008605.png" alt="image-20210104103008605" style="zoom: 33%;" />

可见：**SSD使用感受野小的feature map(8x8)检测小目标，使用感受野大的feature map(4x4)检测更大目标**

<img src="image-20210104103332623.png" alt="image-20210104103332623" style="zoom: 33%;" />

在SSD中，先验框的尺寸都是**人工预设**的，不过在yoloV2中，先验框的尺寸是由初始的标签，经过**kmeans聚类**，得到一系列尺寸。



Prior box的使用：

<img src="image-20210104103732786.png" alt="image-20210104103732786" style="zoom: 50%;" />

以conv4_3为例，在此分为了3条线路：

1. 经过一次batch norm+一次卷积后，生成了**[1, num_class\*num_priorbox, layer_height, layer_width]**大小的feature用于softmax分类目标和非目标（其中num_class是目标类别，SSD300中num_class = 21，即20个类别+1个背景)
2. 经过一次batch norm+一次卷积后，生成了**[1, 4\*num_priorbox, layer_height, layer_width]**大小的feature用于bounding box regression（即每个点一组[dxmin，dymin，dxmax，dymax]，参考[Faster R-CNN](https://zhuanlan.zhihu.com/p/31426458) 2.5节）
3. 生成了**[1, 2, 4\*num_priorbox\*layer_height\*layer_width]**大小的prior box blob，其中2个channel分别存储prior box的4个点坐标(x1, y1, x2, y2)和对应的4个参数variance

后续通过softmax分类判定Prior box是否包含目标，然后再通过bounding box regression即可可获取目标的精确位置。

还有一个细节就是上面prototxt中的4个 **variance**，这实际上是一种bounding regression中的**权重**。



prior box的位置回归代码（box_utils.cpp的void DecodeBBox()）：

```python
decode_bbox->set_xmin(
     prior_bbox.xmin() + prior_variance[0] * bbox.xmin() * prior_width);
 decode_bbox->set_ymin(
     prior_bbox.ymin() + prior_variance[1] * bbox.ymin() * prior_height);
 decode_bbox->set_xmax(
     prior_bbox.xmax() + prior_variance[2] * bbox.xmax() * prior_width);
 decode_bbox->set_ymax(
     prior_bbox.ymax() + prior_variance[3] * bbox.ymax() * prior_height);
```



## 3、Data flow

上一节以conv4_3 feature map分析了如何检测到目标的真实位置，但是SSD 300是使用包括conv4_3在内的共计6个feature maps一同检测出最终目标的。在网络运行的时候显然不能像图6一样：一个feature map单独计算一次multiclass softmax socre+box regression（虽然原理如此，但是不能如此实现）。



那么多个feature maps如何协同工作？这就用到Permute，Flatten和Concat这3种层了。

==Permute(交换数据维度)==：

​	**bottom blob = [batch_num, channel, height, width]**，经过conv4_3_norm_mbox_conf_perm后的caffe blob为：**top blob = [batch_num, height, width, channel]**

<img src="image-20210104110631140.png" alt="image-20210104110631140" style="zoom: 50%;" />

​		以conv4_3和fc7为例分析SSD是如何将不同size的feature map组合在一起进行prediction。图7展示了conv4_3和fc7合并在一起的过程中caffe blob shape变化。

<img src="image-20210104112708175.png" alt="image-20210104112708175" style="zoom: 50%;" />

SSD一次判断priorbox到底是背景 or 是20种目标类别之一，相当于将Faster R-CNN的RPN与后续proposal再分类进行了整合。

![image-20210104133535552](image-20210104133535552.png)



## 4、SSD的优缺点

1、优点：精度可与FasterRcnn对比，速度可与Yolo对比，可以检测多尺度

2、缺点：

​		· 需要**人工设置prior box的min_size，max_size和aspect_ratio值**。网络中prior box的基础大小和形状不能直接通过学习获得，而是需要手工设置。而网络中每一层feature使用的prior box大小和形状恰好都不一样，导致调试过程非常依赖经验

​		· 虽然采用了pyramdial feature hierarchy的思路，但是对小目标的recall依然一般，并没有达到碾压Faster RCNN的级别。作者认为，这是由于SSD使用conv4_3低级feature去检测小目标，而低级特征卷积层数少，存在特征提取不充分的问题



## 5、训练过程

<img src="image-20210104134056487.png" alt="image-20210104134056487" style="zoom:50%;" />

对于SSD，虽然paper中指出采用了所谓的“multibox loss”，但是依然可以清晰看到SSD loss分为了**confidence loss**和location loss(**bouding box regression loss**)两部分，其中N是match到GT（Ground Truth）的prior box数量；

而α参数用于调整confidence loss和location loss之间的比例，**默认α=1**。

SSD中的confidence loss是典型的softmax loss：

<img src="image-20210104134114624.png" alt="image-20210104134114624" style="zoom: 50%;" />

其中，Xij = {1，0}

<img src="image-20210104134820779.png" alt="image-20210104134820779" style="zoom:50%;" />

**匹配策略：**

在训练时，groundtruth boxes 与 default boxes（就是prior boxes） 按照如下方式进行配对：

- 首先，寻找与每一个ground truth box有最大的IoU的default box，这样就能保证每一个groundtruth box与唯一的一个default box对应起来。
- SSD之后又将剩余还没有配对的default box与任意一个groundtruth box尝试配对，只要两者之间的IoU大于阈值，就认为match（SSD 300 阈值为0.5）。
- 显然配对到GT的default box就是positive，没有配对到GT的default box就是negative。



**正负样本：**

值得注意的是，一般情况下negative default boxes数量>>positive default boxes数量，直接训练会导致网络过于重视负样本，从而loss不稳定。所以需要采取：

- 所以SSD在训练时会依据confidience score排序default box，挑选其中confidence高的box进行训练，控制Positive ：Negative = 1 ：3



**数据增强：**

对每一张image进行如下之一变换获取一个patch进行训练：

- 直接使用原始的图像（即不进行变换）
- 采样一个patch，保证与GT之间最小的IoU为：0.1，0.3，0.5，0.7 或 0.9
- 完全随机的采样一个patch





## 6、代码细节



### 6.1 模型的构建

**backbone**：

采用resnet50中的前4大层。

```python
class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0] # 得到了backbone的CONV4的第0个bottleneck

        # 修改conv4_block1的步距，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    # 相当于执行了ResNet50的前面4大层
    def forward(self, x):
        x = self.feature_extractor(x)
        return x
```



**SSD300整体网络构建**：

```python
class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(SSD300, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")
        self.feature_extractor = backbone # 前4层作为backbone

        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = []
        confidence_extractors = []

        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        default_box = dboxes300_coco()
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)
```



**Backbone之后的5个额外的特征检测层**：

```python
def _build_additional_features(self, input_size):
    """
    为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
    :param ：input_size:
    :return ：self.additional_blocks
    """
    additional_blocks = []
    # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
    middle_channels = [256, 256, 128, 128, 128]
    for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
        padding, stride = (1, 2) if i < 3 else (0, 1)
        layer = nn.Sequential(
            nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(output_ch),
            nn.ReLU(inplace=True),
        )
        additional_blocks.append(layer)
    self.additional_blocks = nn.ModuleList(additional_blocks)
```



**bbox_view**函数：

接在每一个预测特征层，输出要学习的参数。

```python
def bbox_view(self, features, loc_extractor, conf_extractor):
    locs = []
    confs = []
    # f是每个预测特征层输出，l是每一层的location预测器(一个卷积)，c是每一层的confidence预测器(一个卷积)，共6层
    for f, l, c in zip(features, loc_extractor, conf_extractor):
        # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
        # -1为n*feat_size*feat_size，即当前预测特征层的default_boxes总数
        locs.append(l(f).view(f.size(0), 4, -1))
        # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
        confs.append(c(f).view(f.size(0), self.num_classes, -1))
    # 最后得到所有预测特征层的坐标回归参数与置信度回归参数的输出

    locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous() # dim=2拼接(即D_box那个维度)
    # contiguous是将view之后的数据调整为连续存储
    return locs, confs
```



**前向传播过程**：

```python
def forward(self, image, targets=None):
    x = self.feature_extractor(image)

    # 得到6个输出的Feature Map ：38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
    detection_features = torch.jit.annotate(List[Tensor], [])  # [x]
    detection_features.append(x)
    for layer in self.additional_blocks:
        x = layer(x)
        detection_features.append(x)

    # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
    locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

    # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
    # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732

    if self.training:
        if targets is None:
            raise ValueError("In training mode, targets should be passed")
        # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        bboxes_out = targets['boxes']
        bboxes_out = bboxes_out.transpose(1, 2).contiguous()
        # print(bboxes_out.is_contiguous())
        labels_out = targets['labels']
        # print(labels_out.is_contiguous())

        # ploc, plabel, gloc, glabel
        loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
        return {"total_losses": loss}

    # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
    # results = self.encoder.decode_batch(locs, confs)
    results = self.postprocess(locs, confs)
    return results
```



### 6.2 预选框的生成



#### 主要代码

```python
def dboxes300_coco():
    figsize = 300  # 输入网络的图像大小
    feat_size = [38, 19, 10, 5, 3, 1]   # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]   # 每个特征层上的一个cell在原图上的跨度
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]  # 每个特征层上预测的default box的scale
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]  # 每个预测特征层上预测的default box的ratios
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes
```

**DefaultBoxes：**

```python
class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.fig_size = fig_size   # 输入网络的图像大小 300
        # [38, 19, 10, 5, 3, 1]
        self.feat_size = feat_size  # 每个预测层的feature map尺寸

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        # [8, 16, 32, 64, 100, 300]
        self.steps = steps    # 每个特征层上的一个cell在原图上的跨度

        # [21, 45, 99, 153, 207, 261, 315]
        self.scales = scales  # 每个特征层上预测的default box的scale

        fk = fig_size / np.array(steps)     # 计算每层特征层的fk
        # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = aspect_ratios  # 每个预测特征层上预测的default box的ratios

        self.default_boxes = []
        # size of feature and number of feature
        # 遍历每层特征层，计算default box，idx即为层的索引
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size  # scale转为相对值[0-1]
            sk2 = scales[idx + 1] / fig_size  # scale转为相对值[0-1]
            sk3 = sqrt(sk1 * sk2)
            # 先添加两个1:1比例的default box宽和高
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            # 再将该特征层，剩下不同比例的default box宽和高添加到all_sizes中
            for alpha in aspect_ratios[idx]: # aspect_ratios[idx]指该层所有的的ratio
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w)) # 相当于aspect_ratios中的单个value(2)在这对应1:2和2:1

            # 计算当前特征层对应原图上的所有default box
            for w, h in all_sizes: # all_sizes对应每个default box的尺寸
                for i, j in itertools.product(range(sfeat), repeat=2):  # i -> 行（y）， j -> 列（x）
                    # 计算每个default box的中心坐标（范围是在0-1之间）
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx] # 0.5是offset，相当于(j+0.5)*step/fig_size
                    self.default_boxes.append((cx, cy, w, h))

        # 将default_boxes转为tensor格式
        self.dboxes = torch.as_tensor(self.default_boxes, dtype=torch.float32)  # 这里不转类型会报错
        self.dboxes.clamp_(min=0, max=1)  # 将坐标（x, y, w, h）都限制在0-1之间

        # For IoU calculation
        # ltrb is left top coordinate and right bottom coordinate
        # 将(x, y, w, h)转换成(xmin, ymin, xmax, ymax)，方便后续计算IoU(匹配正负样本时)
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]   # xmin
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]   # ymin
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]   # xmax
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]   # ymax
```

生成默认框示意图：

<img src="image-20210105233109078.png" alt="image-20210105233109078" style="zoom: 33%;" />



#### 相关函数

1、生成网格函数：**itertools.product()；np.mgrid[]；torch.meshgrid()**

```python
import itertools
import numpy as np
import torch

feature_map_size = (3, 3)

for i, j in itertools.product(range(3), repeat=2):
    print(i,j) # j代表列，i代表行

y, x  = np.mgrid[0:feature_map_size[0], 0:feature_map_size[1]]
# 生成了两个3*3矩阵，两个矩阵配合(y[i],x[i])，即为网格的idx坐标
print(y)
print(x)

grid_y, grid_x = torch.meshgrid([torch.arange(3), torch.arange(3)])
# 与np.mgrid[]效果相似
print(grid_y)
print(grid_x)
```

代码效果：

```python
#itertools.product()效果：
0 0
0 1
0 2
1 0
1 1
1 2
2 0
2 1
2 2

#np.mgrid[]效果：
[[0 0 0]
 [1 1 1]
 [2 2 2]]
[[0 1 2]
 [0 1 2]
 [0 1 2]]

#torch.meshgrid()效果：
tensor([[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]])
tensor([[0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]])
```



2、torch 的**clamp_(min, max)**函数功能：将坐标（x, y, w, h）都限制在0-1之间

```python
clamp_(min=0, max=1)：
	  | min, if x_i < min
y_i = | x_i, if min <= x_i <= max
      | max, if x_i > max
```



### 6.2 预选框损失

论文中给出的计算损失图：

![image-20210106113623891](image-20210106113623891.png)



代码部分：

```python
class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # dboxes 原来是：tensor(8732,4)
        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]，transpose就是调换维度功能
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0), requires_grad=False)

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
```

其中，scale_xy和scale_wh是可以加速收敛的超参数

设置**loc_loss采用SmoothL1Loss损失函数**；设置**con_loss采用CrossEntropyLoss损失函数**。

**SmoothL1Loss**数学公式：

<img src="image-20210105223031040.png" alt="image-20210105223031040" style="zoom: 50%;" />

​		由公式可见，当预测值和ground truth差别较小的时候（绝对值差小于1），其实使用的是L2 Loss；而当差别大的时候，是L1 Loss的平移。



**预设框与grodtruth的坐标偏移量计算，作为训练时的loc真实标签**

```python
def _location_vec(self, loc):
    # type: (Tensor) -> Tensor
    """
    Generate Location Vectors
    计算ground truth相对anchors的回归参数
    :param loc: anchor匹配到的对应GTBOX Nx4x8732
    :return:
    """
    # 小trick，乘上scale缩放因子后，网络更加快速收敛
    gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # Nx2x8732
    gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
    return torch.cat((gxy, gwh), dim=1).contiguous() # Nx 4 x8732
```

<img src="image-20210106131856772.png" alt="image-20210106131856772" style="zoom: 67%;" />



**前向传播过程：**

```python
def forward(self, ploc, plabel, gloc, glabel): # 备注：gloc和glabel都是数据预处理时弄得
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    """
        ploc, plabel: Nx4x8732, Nxlabel_numx8732
            predicted location and labels

        gloc, glabel: Nx4x8732, Nx8732
            ground truth location and labels
    """
    # 获取正样本的mask  Tensor: [N, 8732], N代表的是batch即多少张图片
    mask = torch.gt(glabel, 0)  # (gt: >)
    # mask1 = torch.nonzero(glabel)
    # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
    pos_num = mask.sum(dim=1)

    # 计算gt的location回归参数 Tensor: [N, 4, 8732]
    vec_gd = self._location_vec(gloc)

    # sum on four coordinates, and mask
    # 在dim1上对Cx,Cy,w,h四个值的损失进行求和得到总损失，即每个default_box的总损失
    loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732]
    # 计算定位损失(只要正样本), 故乘上mask，最终再在dim1即box维度求和，得到每张图片正样本的总损失
    loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]

    # hard negative mining Tenosr: [N, 8732]
    con = self.confidence_loss(plabel, glabel)

    # positive mask will never selected
    # 获取负样本，将正样本都置为0
    con_neg = con.clone()
    con_neg[mask] = 0.0
    # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
    _, con_idx = con_neg.sort(dim=1, descending=True) # 对负样本进行降序
    _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙

    # number of negative three times positive
    # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
    # 但不能超过总样本数8732
    neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
    neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]

    # confidence最终loss使用选取的正样本loss+选取的负样本loss
    con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor [N]

    # avoid no object detected
    # 避免出现图像中没有GTBOX的情况
    total_loss = loc_loss + con_loss
    # eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
    num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
    pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
    ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
    return ret
```

> **1、需要注意的是两个sort部分**

```python
_, con_idx = con_neg.sort(dim=1, descending=True) # 对负样本进行降序
_, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙
```
经过两次sort，得到的是[0,2,4,1,5,7,6,8]每个score值在该batch中的排序rank，rank越小代表score越大

> **2、需要注意，正负样本的配比，原论文中，正：负样本之比为1:3**

```python
neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]
```
也要注意，别超过总预测框数8732

> **3、注意，N是图像中带有正样本的图片总数**

<img src="image-20210106152247302.png" alt="image-20210106152247302" style="zoom:50%;" />

```python
total_loss = loc_loss + con_loss
num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
```





### 6.3 PostProcess 后处理

```python
self.postprocess = PostProcess(default_box)
...
results = self.postprocess(locs, confs)
```



**Class   PostProcess**：

```python
class PostProcess(nn.Module):
    def __init__(self, dboxes):
        super(PostProcess, self).__init__()
        # [num_anchors, 4] -> [1, num_anchors, 4]
        self.dboxes_xywh = nn.Parameter(dboxes(order='xywh').unsqueeze(dim=0),
                                        requires_grad=False)
        self.scale_xy = dboxes.scale_xy  # 0.1
        self.scale_wh = dboxes.scale_wh  # 0.2

        self.criteria = 0.5 # Iou阈值
        self.max_output = 100 #  每张图输出的最大目标个数
```

**forward：**

```python
def forward(self, bboxes_in, scores_in):
    # 通过预测的boxes回归参数得到最终预测坐标, 将预测目标score通过softmax处理
    bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

    outputs = torch.jit.annotate(List[Tuple[Tensor, Tensor, Tensor]], [])
    # 遍历一个batch中的每张image数据
    # bboxes: [batch, 8732, 4]
    for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):  # split_size, split_dim
        bbox = bbox.squeeze(0) # bbox: [1, 8732, 4] -> [8732, 4]
        prob = prob.squeeze(0) # prob: [1, 8732, 21] -> [8732, 21]
        outputs.append(self.decode_single_new(bbox, prob, self.criteria, self.max_output))
    return outputs
```

**Scale_back_batch**：

```python
def scale_back_batch(self, bboxes_in, scores_in):
    ## type: (Tensor, Tensor)
    """
        1）通过预测的boxes回归参数得到最终预测坐标
        2）将box格式从xywh转换回ltrb
        3）将预测目标score通过softmax处理
        Do scale and transform from xywh to ltrb
        suppose input N x 4 x num_bbox | N x label_num x num_bbox

        bboxes_in: [N, 4, 8732]是网络预测的xywh回归参数(坐标偏移量)
        scores_in: [N, label_num, 8732]是预测的每个default box的各目标概率
    """

    # Returns a view of the original tensor with its dimensions permuted.
    # [batch, 4, 8732] -> [batch, 8732, 4]
    bboxes_in = bboxes_in.permute(0, 2, 1)
    # [batch, label_num, 8732] -> [batch, 8732, label_num] label_num：21
    scores_in = scores_in.permute(0, 2, 1)
    # print(bboxes_in.is_contiguous())

    bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]   # 预测的x, y回归参数
    bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]   # 预测的w, h回归参数

    # 将预测的回归参数叠加到default box上得到最终的预测边界框，就是公式的反向推导
    bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
    bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

    # transform format to ltrb
    l = bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2]
    t = bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3]
    r = bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2]
    b = bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3]

    bboxes_in[:, :, 0] = l  # xmin
    bboxes_in[:, :, 1] = t  # ymin
    bboxes_in[:, :, 2] = r  # xmax
    bboxes_in[:, :, 3] = b  # ymax

    # scores_in: [batch, 8732, label_num]
    return bboxes_in, F.softmax(scores_in, dim=-1)
```

**Decode_single_new：**

```python
def decode_single_new(self, bboxes_in, scores_in, criteria, num_output):
    ## type: (Tensor, Tensor, float, int)
    """
    decode:
        input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
        output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
        criteria : IoU threshold of bboexes
        max_output : maximum number of output bboxes
    """
    device = bboxes_in.device
    num_classes = scores_in.shape[-1]

    # 对越界的bbox进行裁剪
    bboxes_in = bboxes_in.clamp(min=0, max=1) # 因为预测的都是相对坐标

    # [8732, 4] -> [8732, 21, 4] # 重复num_classes次
    bboxes_in = bboxes_in.repeat(1, num_classes).reshape(scores_in.shape[0], -1, 4)

    # create labels for each prediction
    labels = torch.arange(num_classes, device=device)
    # [num_classes] -> [8732, num_classes]
    labels = labels.view(1, -1).expand_as(scores_in)

    # remove prediction with the background label
    # 移除归为背景类别的概率信息, 故从1开始取
    bboxes_in = bboxes_in[:, 1:, :]  # [8732, 21, 4] -> [8732, 20, 4]
    scores_in = scores_in[:, 1:]  # [8732, 21] -> [8732, 20]
    labels = labels[:, 1:]  # [8732, 21] -> [8732, 20]

    # batch everything, by making every class prediction be a separate instance
    bboxes_in = bboxes_in.reshape(-1, 4)  # [8732, 20, 4] -> [8732x20, 4]
    scores_in = scores_in.reshape(-1)  # [8732, 20] -> [8732x20]
    labels = labels.reshape(-1)  # [8732, 20] -> [8732x20]

    # remove low scoring boxes
    # 移除低概率目标，self.scores_thresh=0.05
    inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
    # 得到相应索引上的数值
    bboxes_in, scores_in, labels = bboxes_in[inds, :], scores_in[inds], labels[inds]


    # remove empty boxes
    ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
    keep = (ws >= 1 / 300) & (hs >= 1 / 300) # 宽带、高度都要大于等于1个像素(因为是相对值，所以比较的是1/300)
    keep = keep.nonzero(as_tuple=False).squeeze(1) # (8742, 1) -> (8742)
    bboxes_in, scores_in, labels = bboxes_in[keep], scores_in[keep], labels[keep]

    # 传入的bboxes_in是左上角、右下角坐标
    # non-maximum suppression
    keep = batched_nms(bboxes_in, scores_in, labels, iou_threshold=criteria)

    # keep only topk scoring predictions
    keep = keep[:num_output]
    # 获取经过nms处理以及output_max限制的所有bbox的位置、置信分数、标签
    bboxes_out = bboxes_in[keep, :]
    scores_out = scores_in[keep]
    labels_out = labels[keep]

    return bboxes_out, labels_out, scores_out
```

> **注意1：在进入nms前，有两层初步筛选，remove low scoring boxes 和 remove empty boxes**

```python
inds = torch.nonzero(scores_in > 0.05, as_tuple=False).squeeze(1)
...
ws, hs = bboxes_in[:, 2] - bboxes_in[:, 0], bboxes_in[:, 3] - bboxes_in[:, 1]
keep = (ws >= 1 / 300) & (hs >= 1 / 300)
```

**Batched_nms：**

```python
def batched_nms(boxes, scores, idxs, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    # 获取所有boxes中（xmin, ymin, xmax, ymax）的最大的坐标值，可能是box7的ymin
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    # 为每一个类别生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后，保证不同类别之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    # 不同类的框无重叠，方便一次性对所有框进行nms
    keep = nms(boxes_for_nms, scores, iou_threshold)
    # 得到一个排序后的索引列表，列表内容是需要保留的框的索引
    return keep
```

需要注意的是，这里的nms手法参考了fasterrcnn，就是将bboxes_in(8732，4)复制了num_classes份，接着将相乘，8732**num_class相乘，而之后的 score(8732，num_class)也化成8732*num_class，labels为num_classes。

这样，就将每个预选框，都有一个label与之对应，之后再添加一个offset，保证不同类之间没有重叠，所以所有的框一起处理nms，也互相没有影响，摒弃了之前按类别进行nms的低效率方式。



