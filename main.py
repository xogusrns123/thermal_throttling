"""
This is application for single node multi-gpus

"""
import os
import time
import torch
import csv
# import visdom
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

# for dataset
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# for model
from torchvision.models import vgg11
from torch.nn.parallel import DistributedDataParallel as DDP

# for profile
# from torch.profiler import profile, record_function, ProfilerActivity
from deepspeed.profiling.flops_profiler import FlopsProfiler

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--rank', type=int, default=0) # world rank
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16) # 
    # parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=0) # total number of process in your cluster
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--root', type=str, default='./cifar')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./save')
    parser.add_argument('--save_file_name', type=str, default='vgg_cifar')
    parser.add_argument('--local_rank', type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.") # internal process ID in each node
    parser.add_argument('--dist_url', type=str)
    # usage : --gpu_ids 0, 1, 2, 3
    return parser

def main_worker(rank, opts):
	# 1. argparse (main)
    # 2. init dist
    local_gpu_id = init_for_distributed(rank, opts)

    # 4. data set
    transform_train = tfs.Compose([
        tfs.Resize(256),
        tfs.RandomCrop(224),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2023, 0.1994, 0.2010)),
    ])

    transform_test = tfs.Compose([
                                  tfs.Resize(256),
                                  tfs.CenterCrop(224),
                                  tfs.ToTensor(),
                                  tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                std=(0.2023, 0.1994, 0.2010)),
                                        ])

    # First process downloads the dataset, the others wait for completion
    if opts.rank == 0:
        CIFAR10(root=opts.root, train=True, download=True)
        CIFAR10(root=opts.root, train=False, download=True)
    dist.barrier()  # All processes synchronize here

    train_set = CIFAR10(root=opts.root,
                        train=True,
                        transform=transform_train,
                        download=True)

    test_set = CIFAR10(root=opts.root,
                       train=False,
                       transform=transform_test,
                       download=True)

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    test_sampler = DistributedSampler(dataset=test_set, shuffle=False)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=int(opts.batch_size / opts.world_size),
                              shuffle=False,
                              num_workers=int(opts.num_workers / opts.world_size),
                              sampler=train_sampler,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=int(opts.batch_size / opts.world_size),
                             shuffle=False,
                             num_workers=int(opts.num_workers / opts.world_size),
                             sampler=test_sampler,
                             pin_memory=True)

    # 5. model
    model = vgg11(weights=None)
    model = model.cuda(local_gpu_id)
    # For Profiling
    prof = FlopsProfiler(model)
    model = DDP(module=model,
                device_ids=[local_gpu_id])

    # 6. criterion
    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)

    # 7. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.01,
                                weight_decay=0.0005,
                                momentum=0.9)

    # 8. scheduler
    scheduler = StepLR(optimizer=optimizer,
                       step_size=30,
                       gamma=0.1)

    if opts.start_epoch != 0:

        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'
                                .format(opts.start_epoch - 1),
                                map_location=torch.device('cuda:{}'.format(local_gpu_id)))
        model.load_state_dict(checkpoint['model_state_dict'])  # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # load sched state dict
        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))


    profile_step = 5
    file_path = os.path.join("./profile", f"gpu{str(local_gpu_id)}.csv")

    # CSV 파일의 디렉토리 경로를 추출합니다.
    dir_name = os.path.dirname(file_path)

    # 디렉토리가 존재하지 않는 경우 디렉토리를 생성합니다.
    os.makedirs(dir_name, exist_ok=True)

    with open(file_path, mode='w') as f:
        writer = csv.writer(f)

        # CSV 헤더 작성
        writer.writerow(['epoch', 'iter', 'flops', 'latency'])

    for epoch in range(opts.start_epoch, opts.epoch):

        # 9. train
        tic = time.time()
        model.train()
        train_sampler.set_epoch(epoch)

        for i, (images, labels) in enumerate(train_loader):

            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)

            # start profiling at training step "profile_step"
            if i % profile_step == 0:
                prof.start_profile()

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory= True, with_flops= True) as prof:    
            outputs = model(images)
                
            # end profiling and print ouput
            if i % profile_step == 0:  
                prof.stop_profile()
                flops = prof.get_total_flops(as_string=True).strip()
                latency = prof.get_total_duration(as_string=True).strip()

                # gpu_stats = os.popen(f"nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total \
                #                          --id={idx} --format=csv,noheader").read().split(',')
                                         #['2023/12/06 06:23:51.751', ' 32', ' 0 %', ' 34 %', ' 22.22 W', ' 450.00 W', ' 1 MiB', ' 24564 MiB\n']
                # prof.print_model_profile(profile_step=profile_step, output_file=file_path)

                with open(file_path, 'a') as f:
                    writer = csv.writer(f)

                    writer.writerow([str(epoch), str(i), flops, latency])

                prof.end_profile()

                
            # ----------- update -----------
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(prof.key_averages())

            # get lr
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            # time
            toc = time.time()

            # visualization
            if (i % opts.vis_step == 0 or i == len(train_loader) - 1) and opts.rank == 0:
                print('Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, LR: {5:.5f}, Time: {6:.2f}'.format(epoch,
                                                                                                          opts.epoch, i,
                                                                                                          len(train_loader),
                                                                                                          loss.item(),
                                                                                                          lr,
                                                                                                          toc - tic))


        # # save pth file
        # if opts.rank == 0:
        #     if not os.path.exists(opts.save_path):
        #         os.mkdir(opts.save_path)

        #     checkpoint = {'epoch': epoch,
        #                   'model_state_dict': model.state_dict(),
        #                   'optimizer_state_dict': optimizer.state_dict(),
        #                   'scheduler_state_dict': scheduler.state_dict()}

        #     torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))
        #     print("save pth.tar {} epoch!".format(epoch))

        # # 10. test
        # if opts.rank == 0:
        #     model.eval()

        #     val_avg_loss = 0
        #     correct_top1 = 0
        #     correct_top5 = 0
        #     total = 0

        #     with torch.no_grad():
        #         for i, (images, labels) in enumerate(test_loader):
        #             images = images.to(opts.rank)  # [100, 3, 224, 224]
        #             labels = labels.to(opts.rank)  # [100]
        #             outputs = model(images)
        #             loss = criterion(outputs, labels)
        #             val_avg_loss += loss.item()
        #             # ------------------------------------------------------------------------------
        #             # rank 1
        #             _, pred = torch.max(outputs, 1)
        #             total += labels.size(0)
        #             correct_top1 += (pred == labels).sum().item()

        #             # ------------------------------------------------------------------------------
        #             # rank 5
        #             _, rank5 = outputs.topk(5, 1, True, True)
        #             rank5 = rank5.t()
        #             correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

        #             # ------------------------------------------------------------------------------
        #             for k in range(5):  # 0, 1, 2, 3, 4, 5
        #                 correct_k = correct5[:k+1].reshape(-1).float().sum(0, keepdim=True)
        #             correct_top5 += correct_k.item()

        #     accuracy_top1 = correct_top1 / total
        #     accuracy_top5 = correct_top5 / total

        #     val_avg_loss = val_avg_loss / len(test_loader)  # make mean loss

            # print("top-1 percentage :  {0:0.3f}%".format(correct_top1 / total * 100))
            # print("top-5 percentage :  {0:0.3f}%".format(correct_top5 / total * 100))
            scheduler.step()

    cleanup()
    
    return 0


def init_for_distributed(rank, opts):

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])

    if opts.rank == 0:
        print(f'rank:{opts.rank}')
        print(f'local_gpu_id:{local_gpu_id}')
        print(f'device:{torch.cuda.device_count()}')

    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    dist.init_process_group(backend='gloo',
                            # init_method='tcp://goguma2.kaist.ac.kr:23456',
                            world_size=opts.world_size,
                            rank=opts.rank)
    print("end init process group")
    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    # setup_for_distributed(opts.rank == 0)

    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('vgg11 cifar training', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    # main_worker(opts.rank, opts)
    mp.spawn(main_worker,
             args=(opts, ),
             nprocs=opts.world_size,
             join=True)
