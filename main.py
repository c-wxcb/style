
# 最终整理版，代码注释统一放到了draft里面
# 另外，本人是先跑通了draft才开始整理main
# 整体代码量不多，就没有拆分文件

# ==========
# 1.模块引入
# ==========

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

# ===========
# 2.相关配置
# ===========

# 设备
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 路径
project_dir=Path(__file__).resolve().parent
image_dir=project_dir/'images'
output_dir=project_dir/'output'
output_dir.mkdir(parents=True, exist_ok=True)
content_path=image_dir/'content（3000）.jpg'
style_path=image_dir/'style.png'
output_path=output_dir/'stylized.png'
# 模型相关
image_size=1024
content_weight=1
style_weight=1e8
num_steps=1000
tv_weight=1e-6

# ==========
# 3.加载模型
# ==========

vgg=vgg19(
    weights=VGG19_Weights.DEFAULT
).features.to(device).eval()
for layer in vgg:
    if isinstance(layer,nn.ReLU):
        layer.inplace=False

# ==========
# 4.冻结模型
# ==========

for param in vgg.parameters():
    param.requires_grad=False

# ==========
# 5.读取与处理
# ==========

transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])
def load_image(image_path):
    image=Image.open(image_path).convert('RGB')
    tensor=transform(image).unsqueeze(0).to(device)
    return tensor
content_tensor=load_image(content_path)
style_tensor=load_image(style_path)
# 生成图片
generated_tensor=content_tensor.clone().requires_grad_(True).to(device)

# ==========
# 6.标准化
# ==========

normalize_mean=torch.tensor([0.485, 0.456, 0.406],device=device).view(1,3,1,1)
normalize_std=torch.tensor([0.229, 0.224, 0.225],device=device).view(1,3,1,1)
def normalize(tensor):
    return (tensor-normalize_mean)/normalize_std

# ==========
# 7.特征提取
# ==========

content_layer=['conv_10']
style_layer=[
    'conv_1',
    'conv_3',
    'conv_5',
    'conv_9',
    'conv_13',
    'conv_16',
]
def get_features(
        model,
        image,
        layers
):
    x=normalize(image)
    features={}
    conv_count=0
    last_layer=max(int(name.split('_')[1]) for name in layers)
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
with torch.no_grad():
    content_features=get_features(vgg,content_tensor,content_layer)
    style_features=get_features(vgg,style_tensor,style_layer)

# ==========
# 8.gram矩阵
# ==========

def gram_matrix(tensor):
    batch_size,channel,height,width=tensor.size()
    feature_matrix=tensor.view(batch_size*channel,height*width)
    gram=torch.mm(feature_matrix,feature_matrix.t())
    gram=gram/(batch_size*channel*height*width)
    return gram
with torch.no_grad():
    style_grams={}
    for layer_name,feature in style_features.items():
        style_grams[layer_name]=gram_matrix(feature)

# ==========
# 9.内容与风格损失
# ==========

# tvloss
def total_variation_loss(image):
    horizontal=torch.abs(image[:,:,:,1:]-image[:,:,:,:-1])
    vertical=torch.abs(image[:,:,1:,:]-image[:,:,:-1,:])
    tv_loss=horizontal.mean()+vertical.mean()
    return tv_loss

def calculate_loss(
        generated_tensor,
        content_features,
        content_weight,
        content_layer,
        style_grams,
        style_weight,
        style_layers,
        ):
    generated_features=get_features(vgg,generated_tensor,content_layer+style_layers)
    content_name=content_layer[0]
    content_loss=F.mse_loss(generated_features[content_name],content_features[content_name])
    style_loss=0
    for layer_name in style_layers:
        generated_gram=gram_matrix(generated_features[layer_name])
        style_loss=style_loss+F.mse_loss(generated_gram,style_grams[layer_name])
    tv_loss=total_variation_loss(generated_tensor)
    total_loss=content_weight*content_loss + style_weight*style_loss+tv_loss*tv_weight

    return total_loss,content_loss,style_loss,tv_loss

# ==========
# 10.优化器
# ==========

optimizer=torch.optim.LBFGS([generated_tensor])
num_steps=num_steps
runs=[0]
def closure():
    with torch.no_grad():
        generated_tensor.clamp_(0,1)
    optimizer.zero_grad()
    total_loss,content_loss,style_loss,tv_loss=calculate_loss(
        generated_tensor,
        content_features,
        content_weight,
        content_layer,
        style_grams,
        style_weight,
        style_layer
    )
    total_loss.backward()
    runs[0]=runs[0]+1
    if runs[0]%50==0:
        print(
            f'迭代次数:{runs[0]},'
            f'内容损失:{content_loss.item():.4f},'
            f'风格损失:{style_loss.item():.4f},'
            f'tv损失:{tv_loss.item():.4f},'
            f'总损失:{total_loss.item():.4f}'
        )
    return total_loss
while runs[0]<num_steps:
    optimizer.step(closure)
with torch.no_grad():
    generated_tensor.clamp_(0,1)

# ==========
# 11.保存
# ==========

output_tensor=generated_tensor.detach().cpu().squeeze(0)
output_image=transforms.ToPILImage()(output_tensor)
output_image.show()
output_image.save(output_path)
print(f'生成图已保存到：{output_path}')





























































