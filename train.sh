python3.10 train.py --name 01_ConvNeXtBlock_MaxPool2D_ConvTranspose2D --levels TINY --block ConvNeXtBlock --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 01_ConvNeXtBlockV2_MaxPool2D_ConvTranspose2D --levels TINY --block ConvNeXtBlockV2 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 01_EDSRBlock_MaxPool2D_ConvTranspose2D --levels TINY --block EDSRBlock --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler --n_recurrent 0 bicubic --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 01_ResNetBlockV2_BatchNorm_MaxPool2D_ConvTranspose2D --levels TINY --block ResNetBlockV2 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 01_ResNetBlockV2_LayerNorm_MaxPool2D_ConvTranspose2D --levels TINY --block ResNetBlockV2 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --layer_norm --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 02_ConvNeXtBlock_Conv2D_ConvTranspose2D --levels TINY --block ConvNeXtBlock --downsampler conv2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 02_ResNetBlockV2_Conv2D_ConvTranspose2D --levels TINY --block ResNetBlockV2 --downsampler conv2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 03_ConvNeXtBlock_Conv2D_PixelShuffle --levels TINY --block ConvNeXtBlock --downsampler conv2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 03_ConvNeXtBlock_MaxPool2D_PixelShuffle --levels TINY --block ConvNeXtBlock --downsampler maxpool2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 03_ResNetBlockV2_Conv2D_PixelShuffle --levels TINY --block ResNetBlockV2 --downsampler conv2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 03_ResNetBlockV2_MaxPool2D_PixelShuffle --levels TINY --block ResNetBlockV2 --downsampler maxpool2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 0 --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 04_ConvNeXtBlock_ChannelAttention --levels TINY --block ConvNeXtBlock --downsampler maxpool2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 0 --channel_attention --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 04_ResNetBlockV2_ChannelAttention --levels TINY --block ResNetBlockV2 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --channel_attention --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 05_ConvNeXtBlock_AttentionGate --levels TINY --block ConvNeXtBlock --downsampler maxpool2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 0 --attention_gate --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 05_ResNetBlockV2_AttentionGate --levels TINY --block ResNetBlockV2 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --attention_gate --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 06_ConvNeXtBlock_ChannelAttention_AttentionGate --levels TINY --block ConvNeXtBlock --downsampler maxpool2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 0 --channel_attention --attention_gate --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 06_ResNetBlockV2_ChannelAttention_AttentionGate --levels TINY --block ResNetBlockV2 --downsampler maxpool2d --upsampler convtranspose2d --super_upsampler bicubic --n_recurrent 0 --channel_attention --attention_gate --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 07_ConvNeXtBlock_T1 --levels TINY --block ConvNeXtBlock --downsampler maxpool2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 1 --channel_attention --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
python3.10 train.py --name 07_ConvNeXtBlock_T2 --levels TINY --block ConvNeXtBlock --downsampler maxpool2d --upsampler pixelshuffle --super_upsampler bicubic --n_recurrent 2 --channel_attention --init_weights --alpha 1.0 --beta 0 --gamma 0 --eta 0 --mu 0 --aux_weight 0 --epochs 20 --asym_pct 0.05
