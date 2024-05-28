import math

def warmUpLearningRate(max_num_epochs,warm_up_epochs=1,scheduler='cosine'):
    lr_milestones = [20, 40]
    # MultiStepLR without warm up
    finnalScheduler = lambda epoch: 0.1 ** len([m for m in lr_milestones if m <= epoch])
    if scheduler == 'multistep':
        # warm_up_with_multistep_lr
        finnalScheduler = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs else 0.1 ** len(
            [m for m in lr_milestones if m <= epoch])
    elif scheduler == 'step':
        # warm_up_with_step_lr
        gamma = 0.9
        stepsize = 1
        finnalScheduler = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
            else gamma ** (((epoch - warm_up_epochs) / (max_num_epochs - warm_up_epochs)) // stepsize * stepsize)
    elif scheduler == 'cosine':
        # warm_up_with_cosine_lr
        finnalScheduler = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
            else 0.5 * (math.cos((epoch - warm_up_epochs) / (max_num_epochs - warm_up_epochs) * math.pi) + 1)

    return finnalScheduler