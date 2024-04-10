python3.10 train.py --name 01_ResNetBlockV2 --levels TINY --block ResNetBlockV2 --n_recurrent 0 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 01_EDSRBlock --levels TINY --block EDSRBlock --n_recurrent 0 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 01_ConvNeXtBlock --levels TINY --block ConvNeXtBlock --n_recurrent 0 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 01_ConvNeXtBlockV2 --levels TINY --block ConvNeXtBlockV2 --n_recurrent 0 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 02_EDSRBlock_conv2d --levels TINY --block EDSRBlock --n_recurrent 0 --downsampler conv2d --upsampler convtranspose2d --super_upsampler bicubic --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 02_ConvNeXtBlock_conv2d --levels TINY --block ConvNeXtBlock --n_recurrent 0 --downsampler conv2d --upsampler convtranspose2d --super_upsampler bicubic --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05