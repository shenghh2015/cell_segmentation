# parser = argparse.ArgumentParser()
# parser.add_argument("--docker", type=str2bool, default = True)
# parser.add_argument("--gpu", type=str, default = '2')
# parser.add_argument("--net_type", type=str, default = 'unet1')  #Unet, Linknet, PSPNet, FPN
# parser.add_argument("--backbone", type=str, default = 'xxxx')
# parser.add_argument("--feat_version", type=int, default = None)
# parser.add_argument("--epoch", type=int, default = 2)
# parser.add_argument("--dim", type=int, default = 512)
# parser.add_argument("--batch_size", type=int, default = 4)
# parser.add_argument("--dataset", type=str, default = 'live_dead')
# parser.add_argument("--ext", type=str2bool, default = False)
# parser.add_argument("--upsample", type=str, default = 'upsampling')
# parser.add_argument("--pyramid_agg", type=str, default = 'sum')
# parser.add_argument("--filters", type=int, default = 16)
# parser.add_argument("--rot", type=float, default = 0)
# parser.add_argument("--lr", type=float, default = 1e-3)
# parser.add_argument("--bk", type=float, default = 0.5)
# parser.add_argument("--focal_weight", type=float, default = 1)
# parser.add_argument("--bn", type=str2bool, default = True)
# parser.add_argument("--train", type=int, default = None)
# parser.add_argument("--loss", type=str, default = 'focal+dice')
# parser.add_argument("--reduce_factor", type=float, default = 0.1)
# args = parser.parse_args()
# python train_unet.py --gpu 1 --epoch 400 --batch_size 6 --dataset cell_cycle_1984_v2 --ext True --rot 20 --lr 5e-4 --loss focal --bn True --reduce_factor 0.9
python single_train_v6.py --net_type AtUnet --backbone efficientnetb0 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 1100 --gpu 5 --loss focal --filters 256 --upsample upsampling --ext True --reduce_factor 1.0 --bk 1.0 --docker False
python single_train_v6.py --net_type AtUnet --backbone efficientnetb1 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 1100 --gpu 6 --loss focal --filters 256 --upsample upsampling --ext True --reduce_factor 1.0 --bk 1.0 --docker False
python single_train_v6.py --net_type AtUnet --backbone efficientnetb2 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 1100 --gpu 0 --loss focal --filters 256 --upsample upsampling --ext True --reduce_factor 1.0 --bk 1.0 --docker False
python single_train_v6.py --net_type AtUnet --backbone efficientnetb3 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 1100 --gpu 1 --loss focal --filters 256 --upsample upsampling --ext True --reduce_factor 1.0 --bk 1.0 --docker False
python single_train_v6.py --net_type Unet --backbone efficientnetb0 --pre_train True --batch_size 4 --dim 512 --epoch 2400 --lr 5e-4 --dataset cell_cycle_1984_v2 --train 1100 --gpu 2 --loss focal --filters 256 --upsample upsampling --ext True --reduce_factor 1.0 --bk 1.0 --docker False