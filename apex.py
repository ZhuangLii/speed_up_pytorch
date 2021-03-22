try:
    import apex.optimizers as apex_optim
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


# ------ init ------ #
model.to("cuda")
model, optimizer = amp.initialize(model, optimizer, opt_level='O1') 
# O1 不是 01, 
# opt_level O0 纯FP32训练，可以作为accuracy的baseline；
# O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP1（GEMM, 卷积）还是FP32（Softmax）进行计算; 
# O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算; 
# O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；

for e in epoches:
    for i, (img,label) in enumerate(train_loader):
        optimizer.zero_grad()
        img,label = img.cuda(), label.cuda()
        score = model(img) 
        loss = cal_loss(score, label)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

