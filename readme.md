### links

https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file [stylegan2-ada github]
https://github.com/christianversloot/machine-learning-articles/blob/main/stylegan-a-step-by-step-introduction.md [stylegan]
https://arxiv.org/pdf/1812.04948.pdf [stylegan]
https://arxiv.org/pdf/1912.04958.pdf [stylegan2]
https://arxiv.org/pdf/2006.06676.pdf [stylegan2-ada]
https://nn.labml.ai/gan/stylegan/index.html [stylegan code explaned]
https://oscar-guarnizo.medium.com/review-image2stylegan-embedding-an-image-into-stylegan-c7989e345271 [latent space]


### docker 

docker run -it -d --gpus=0 --cpus=23 --shm-size=2048m --name=stylegan -v/home/gpuuser/Denis-Koreshkov/stylegan2-ada-pytorch:/shared cuda:12.3 /bin/bash


### generation 

python generate.py --outdir=out --trunc=0.6 --network=ffhq.pkl --seeds=123
python projector.py --outdir=out --num-steps=300 --target=14hed.png --network=ffhq.pkl

conda env create -f environment.yml
conda deactivate
conda remove -n ldm --all


### add conda env to jupyter kernel

conda install jupyter
conda install ipykernel
python -m ipykernel install --user --name stylegan







