import torch
import numpy as np
from utils import mse_loss

def train(model, data, optimizer):
    model.train()

    x, y, mask, idx = data
    optimizer.zero_grad()

    output,KLD = model(x)
    output = output.squeeze(0)
    loss = mse_loss(output,y,mask)
    
    # 数值稳定性：检查KLD是否为NaN/Inf
    if torch.isnan(KLD) or torch.isinf(KLD):
        KLD = torch.tensor(0.0, device=KLD.device, requires_grad=True)
    elif not KLD.requires_grad:
        # 如果KLD不需要梯度，创建一个需要梯度的版本
        KLD = torch.tensor(KLD.item(), device=KLD.device, requires_grad=True)
    
    # 对KLD进行clamp，防止过大
    KLD = torch.clamp(KLD, min=0.0, max=1000.0)
    
    loss = loss + KLD/torch.sum(mask)
    
    # 检查最终loss
    if torch.isnan(loss) or torch.isinf(loss):
        # 如果loss是NaN，返回0并跳过这一步
        return output, 0.0, 0.0
    
    # 检查loss是否需要梯度
    if not loss.requires_grad:
        # 如果loss不需要梯度，说明没有可训练参数，跳过backward
        return output, loss.item(), np.sqrt(loss.item())
    
    loss.backward()
    # 只对可训练参数进行梯度裁剪
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)  # 降低梯度裁剪阈值
    optimizer.step()
    mse = loss.item()
    rmse = np.sqrt(mse)
    return output, mse, rmse

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        x, y, mask, idx = data
        output,_ = model(x)
        output = output.squeeze(0)
        loss = mse_loss(output,y,mask)
        mse = loss.item()
        rmse = np.sqrt(mse)

    return output, mse, rmse