# ✍️ The Stable Signature: Rooting Watermarks in Latent Diffusion Models

Implementation and pretrained models.
For details, see [**the paper**](https://arxiv.org/abs/2303.15435) (or go to ICCV 2023 in Paris 🥐).  

[[`Webpage`](https://pierrefdz.github.io/publications/stablesignature/)]
[[`arXiv`](https://arxiv.org/abs/2303.15435)]
[[`Blog`](https://ai.meta.com/blog/stable-signature-watermarking-generative-ai/)]
[[`Demo`](https://huggingface.co/spaces/imatag/stable-signature-bzh)]

## Setup


### Requirements

First, clone the repository locally and move inside the folder:
```cmd
git clone https://github.com/facebookresearch/stable_signature
cd stable_signature
```
To install the main dependencies, we recommand using conda.
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install -c pytorch torchvision pytorch==1.12.0 cudatoolkit==11.3
```

Install the remaining dependencies with pip:
```cmd
pip install -r requirements.txt
```

This codebase has been developed with python version 3.8, PyTorch version 1.12.0, CUDA 11.3.


### Models and data

#### Data

The paper uses the [COCO](https://cocodataset.org/) dataset to fine-tune the LDM decoder (we filtered images containing people).
All you need is around 500 images for training (preferably over 256x256).

#### Watermark models

The watermark extractor model can be downloaded in the following links.
The `.pth` file has not been whitened, while the `.torchscript.pt` file has been and can be used without any further processing. 
We additionally provide another extractor model, which has been trained with blur and rotations and has better robustness to that kind of attacks, at the cost of a slightly lower image quality (you might need to adjust the perceptual loss weight at your convenience).

| Model | Checkpoint | Torch-Script |
| --- | --- | --- |
| Extractor | [dec_48b.pth](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b.pth) | [dec_48b_whit.torchscript.pt](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt)  |
| Other | [other_dec_48b_whit.pth](https://dl.fbaipublicfiles.com/ssl_watermarking/other_dec_48b.pth) | [other_dec_48b_whit.torchscript.pt](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt) |

The following code automatically downloads the models and put them in the `models` folder:
```cmd
mkdir models
wget https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt -P models/
wget https://dl.fbaipublicfiles.com/ssl_watermarking/other_dec_48b_whit.torchscript.pt -P models/
```

Code to train the watermark models is available in the folder called `hidden/`.

#### Stable Diffusion models

Create LDM configs and checkpoints from the [Hugging Face](https://huggingface.co/stabilityai) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion) repositories.
The code should also work for Stable Diffusion v1 without any change. 
For other models (like old LDMs or VQGANs), you may need to adapt the code to load the checkpoints.

An example of watermarked weights is available at [WM weights of latent decoder](https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth) (the key is the one present in the `decoding.ipynb` file).

#### Perceptual Losses

The perceptual losses are based on [this repo](https://github.com/SteffenCzolbe/PerceptualSimilarity/).
You should download the weights here: https://github.com/SteffenCzolbe/PerceptualSimilarity/tree/master/src/loss/weights, and put them in a folder called `losses` (this is used in [src/loss/loss_provider.py#L22](https://github.com/facebookresearch/stable_signature/blob/main/src/loss/loss_provider.py#L22)).
To do so you can run 
```
git clone https://github.com/SteffenCzolbe/PerceptualSimilarity.git
cp -r PerceptualSimilarity/src/loss/weights src/loss/losses/
rm -r PerceptualSimilarity
```


## Usage

### Watermark pre-training

Please see [hidden/README.md](https://github.com/facebookresearch/stable_signature/tree/main/hidden/README.md) for details on how to train the watermark encoder/extractor.

### Fine-tune LDM decoder

```
python finetune_ldm_decoder.py --num_keys 1 \
    --ldm_config path/to/ldm/config.yaml \
    --ldm_ckpt path/to/ldm/ckpt.pth \
    --msg_decoder_path path/to/msg/decoder/ckpt.torchscript.pt \
    --train_dir path/to/train/dir \
    --val_dir path/to/val/dir
```

This code should generate: 
- *num_keys* checkpoints of the LDM decoder with watermark fine-tuning (checkpoint_000.pth, etc.),
- `keys.txt`: text file containing the keys used for fine-tuning (one key per line),
- `imgs`: folder containing examples of auto-encoded images.

[Params of LDM fine-tuning used in the paper](https://justpaste.it/aw0gj)  
[Logs during LDM fine-tuning](https://justpaste.it/cse0x)

### Generate

#### With Stability AI codebase

Reload weights of the LDM decoder in the Stable Diffusion scripts by appending the following lines after loading the checkpoint 
(for instance, [L220 in the SD repo](https://github.com/Stability-AI/stablediffusion/blob/main/scripts/txt2img.py#L220))
```python
state_dict = torch.load(path/to/ldm/checkpoint_000.pth)['ldm_decoder']
msg = model.first_stage_model.load_state_dict(state_dict, strict=False)
print(f"loaded LDM decoder state_dict with message\n{msg}")
print("you should check that the decoder keys are correctly matched")
```

You should also comment the lines that add the post-hoc watermark of SD: `img = put_watermark(img, wm_encoder)`.

[WM weights of SD2 decoder](https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth). Weights obtained after running [this command](https://justpaste.it/ae93f). 
In this case, the state dict only contains the 'ldm_decoder' key, so you only need to load with `state_dict = torch.load(path/to/ckpt.pth)`

#### With Diffusers

Here is a code snippet that could be used to reload the decoder with the Diffusers library (transformers==4.25.1, diffusers==0.25.1). (Still WIP, this might be updated in the future!)

```python
import torch 
device = torch.device("cuda")

from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from utils_model import load_model_from_config 

ldm_config = "sd/stable-diffusion-2-1-base/v2-inference.yaml"
ldm_ckpt = "sd/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt"

print(f'>>> Building LDM model with config {ldm_config} and weights from {ldm_ckpt}...')
config = OmegaConf.load(f"{ldm_config}")
ldm_ae = load_model_from_config(config, ldm_ckpt)
ldm_aef = ldm_ae.first_stage_model
ldm_aef.eval()

# loading the fine-tuned decoder weights
state_dict = torch.load("sd2_decoder.pth")
unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
print(unexpected_keys)
print("you should check that the decoder keys are correctly matched")

# loading the pipeline, and replacing the decode function of the pipe
model = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model).to(device)
pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))

img = pipe("the cat drinks water.").images[0]
img.save("cat.png")
```

### Decode and Evaluate

The `decode.ipynb` notebook contains a full example of the decoding and associated statistical test.

The `run_eval.py` script can be used to get the robustness and quality metrics on a folder of images.
For instance:
```
python run_eval.py --eval_imgs False --eval_bits True \
    --img_dir path/to/imgs_w \
    --key_str '111010110101000001010111010011010100010000100111'
```
will return a csv file containing bit accuracy for different attacks applied before decoding.

```
python run_eval.py --eval_imgs True --eval_bits False \
    --img_dir path/to/imgs_w --img_dir_nw path/to/imgs_nw 
```
will return a csv file containing image metrics (PSNR, SSIM, LPIPS) between watermarked (`_w`) and non-watermarked (`_nw`) images.



## Acknowledgements

This code is based on the following repositories:

- https://github.com/Stability-AI/stablediffusion
- https://github.com/SteffenCzolbe/PerceptualSimilarity

To train the watermark encoder/extractor, you can also refer to the following repository https://github.com/ando-khachatryan/HiDDeN. 

## License

The majority of Stable Signature is licensed under CC-BY-NC, however portions of the project are available under separate license terms: `src/ldm` and `src/taming` are licensed under the MIT license.

## Citation

If you find this repository useful, please consider giving a star :star: and please cite as:


```
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={ICCV},
  year={2023}
}
```
