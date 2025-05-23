export CUDA_VISIBLE_DEVICES=4,5

ray start --temp-dir $PWD/ray_temp --head --node-ip-address 127.0.0.1 --num-gpus 2 --num-cpus 16