
# 主要参考pytorch官方文档
# 本草稿只是代码细节，数学细节在学习笔记里

# ==========
# 1.模块引入
# ==========

import torch
from torchvision.models import vgg19,VGG19_Weights
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

# ==========
# 2.相关配置
# ==========

# 设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 路径
project_dir=Path(__file__).resolve().parent
image_dir=project_dir/'images'
output_dir=project_dir/'output'
output_dir.mkdir(parents=True,exist_ok=True)
content_path=image_dir/'content.jpg'
style_path=image_dir/'style.jpg'

# ==========
# 3.加载模型
# ==========

vgg=vgg19(
    weights=VGG19_Weights.DEFAULT
    ).features.to(device).eval()
    # vgg19是一个函数，构建模型用的
    # weights=VGG19_Weights表示使用官方提供的预训练权重
    # .DEFAULT表示使用默认的权重，.IMAGENET1K表示使用ImageNet-1k的那一套
    # .feature是一个属性，里面只有卷积层(包括激活池化什么的)，剩下的全链接层在classifier里面

# ==========
# 4.冻结参数
# ==========

for parm in vgg.parameters():
    parm.requires_grad=False
    # .parameters表示模型里的参数，优化器里面也会用这个
    # 毕竟是风格提取，我们只需要得到传播中的某些特征图，不需要训练模型，所以直接不更新参数了
# print(vgg)

# ==========
# 5.读取与处理图片
# ==========

# 图片大小
image_size=512
# 处理流程
transform=transforms.Compose([
    transforms.Resize(image_size),# 短边变image_size，长宽比例不变
    transforms.ToTensor(),
])
# 读取内容图片
content_pil=Image.open(content_path).convert('RGB')# 转成rgb三通道形式
content_tensor=transform(content_pil).unsqueeze(0).to(device)# 加入批次维度，毕竟输入的时候是有批次维度的
# 读取风格图片
style_pil=Image.open(style_path).convert('RGB')
style_tensor=transform(style_pil).unsqueeze(0).to(device)

# ==========
# 6.标准化
# ==========

normalize_mean=torch.tensor([0.485, 0.456, 0.406],
                            device=device).view(1,3,1,1)
# 原本vgg模型的均值如此，所以按此来
# 后面的view是将之弄成[1,3,1,1]的形式，和图片的[batch,3,hight,width]形式匹配，便于广播
normalize_std=torch.tensor([0.229, 0.224, 0.225],
                           device=device).view(1,3,1,1)
# 标准化函数
def normalize(tensor):
    return (tensor-normalize_mean)/normalize_std

# ==========
# 7.特征提取
# ==========

# pytorch官方选择的特征提取层
content_layer=['conv_10']# 初始为官方的4
style_layers=['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13','conv_16']# 初始为官方的1到5
for layer in vgg:
    if isinstance(layer,nn.ReLU):
        layer.inplace=False
        # 防止池化之后再次覆盖原值，导致字典里的也变了
        # 所以有的也按照激活层当特征提取层
# 特征提取函数
def get_features(model,image,layers):
    x=normalize(image)
    features={}
    conv_count=0
    last_layer=max(
        int(name.split('_')[1]) for name in layers
    )
    for layer in model:
        x=layer(x)
        if isinstance(layer,nn.Conv2d):
            conv_count=conv_count+1
            layer_name=f'conv_{conv_count}'
            if layer_name in layers:
                features[layer_name]=x
            if conv_count==last_layer:
                break
    return features
# 开始提取
with torch.no_grad():
    content_features=get_features(vgg,content_tensor,content_layer)
    style_features=get_features(vgg,style_tensor,style_layers)

# ==========
# 8.gram矩阵
# ==========

def gram_matrix(feature):
    batch_size,channels,height,width=feature.size()
    # 拿到维度形状
    feature_matrix=feature.view(batch_size*channels,width*height)
    # 压平，相比flatten更直观
    # 相当于全部以此排列再按顺序填入新矩阵，恰好是每张特征图压平之后的效果
    gram=torch.mm(feature_matrix,feature_matrix.t())# 计算gram矩阵
    gram=gram/(batch_size*channels*height*width)# 归一化，防止特征里的数字图尺度过大，干扰损失
    return gram

# ==========
# 9.内容损失与风格损失
# ==========

# 保存风格图片的各层gram矩阵
with torch.no_grad():
    style_grams={}
    for layer_name,feature in style_features.items():
        style_grams[layer_name]=gram_matrix(feature)
# 生成图片
generated_tensor=content_tensor.clone().requires_grad_(True).to(device)
# 损失权重
content_weight=1
style_weight=1e6
# 计算总损失
def calculate_loss(generated_tensor,
                   content_layer,
                   content_weight,
                   content_features,
                   style_layers,
                   style_weight,
                   style_grams):
    # 生成图风格特征
    generated_features=get_features(vgg,generated_tensor,style_layers+content_layer)
    # 内容损失
    content_name=content_layer[0]
    content_loss=F.mse_loss(generated_features[content_name],content_features[content_name])
    # 风格损失
    style_loss=0
    for layer_name in style_layers:
        generated_gram=gram_matrix(generated_features[layer_name])
        style_loss=style_loss+F.mse_loss(generated_gram,style_grams[layer_name])
    total_loss=content_weight*content_loss+style_weight*style_loss
    return total_loss,content_loss,style_loss

# ==========
# 10.优化生成图
# ==========

optimizer=torch.optim.LBFGS([generated_tensor])# 选用lbfgs优化器
num_steps=1000# 最多进行多少次更新
run=[0]# 计数器，如果是数字的话需要global
def closure():
    with torch.no_grad():
        generated_tensor.clamp_(0,1)# 压缩到0，1之间，毕竟原始的内容图风格图都被ToTensor压缩到0，1之间
    optimizer.zero_grad()# 清空梯度
    total_loss,content_loss,style_loss=calculate_loss(generated_tensor,
                                                      content_layer,
                                                      content_weight,
                                                      content_features,
                                                      style_layers,
                                                      style_weight,
                                                      style_grams)
    total_loss.backward()# 计算梯度
    run[0]=run[0]+1
    if run[0]%50==0:
        print(
            f'迭代次数:{run[0]},'
            f'内容损失:{content_loss.item():.4f},'
            f'风格损失：{style_loss.item():4f},'
            f'总损失:{total_loss.item():.4f}'
        )
    return total_loss
while run[0]<num_steps:
    optimizer.step(closure)# closure用来传损失
with torch.no_grad():
    generated_tensor.clamp_(0,1)# 保证合法

# ==========
# 11.保存生成图
# ==========

output_tensor=generated_tensor.detach().cpu().squeeze(0)
# detach切断与计算图的联系
# 移动到cpu上并去掉第0维（之前处理内容图和风格图的时候加上的）
output_image=transforms.ToPILImage()(output_tensor)
output_path=output_dir/'stylized.png'
output_image.save(output_path)
output_image.show()
print(f'生成图已保存到：{output_path}')































