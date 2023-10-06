# ✍️ The Stable Signature: Rooting Watermarks in Latent Diffusion Models

Implementation and pretrained models.
For details, see [**the paper**](https://arxiv.org/abs/2303.15435) (or go to ICCV 2023 in Paris 🥐).  

[[`Webpage`](https://pierrefdz.github.io/publications/stablesignature/)]
[[`arXiv`](https://arxiv.org/abs/2303.15435)]

## Setup


### Requirements

First, clone the repository locally and move inside the folder:
```cmd
git clone https://github.com/facebookresearch/stable_signature
cd stable_signature
```
To install the main dependencies, we recommand using conda, and install the remaining dependencies with pip:
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install -c pytorch torchvision pytorch==1.12.0 cudatoolkit=11.3
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
python finetune_ldm_decoder.py --num_keys 1
    --ldm_config path/to/ldm/config.yaml
    --ldm_ckpt path/to/ldm/ckpt.pth
    --msg_decoder_path path/to/msg/decoder/ckpt.torchscript.pt
    --train_dir path/to/train/dir
    --val_dir path/to/val/dir
```

This code should generate: 
- *num_keys* checkpoints of the LDM decoder with watermark fine-tuning (checkpoint_000.pth, etc.),
- `keys.txt`: text file containing the keys used for fine-tuning (one key per line),
- `imgs`: folder containing examples of auto-encoded images.

[Params of LDM fine-tuning used in the paper](https://justpaste.it/aw0gj)  
[Logs during LDM fine-tuning](https://justpaste.it/cse0x)

### Generate

Reload weights of the LDM decoder in the Stable Diffusion scripts by appending the following lines after loading the checkpoint 
(for instance, [L220 in the SD repo](https://github.com/Stability-AI/stablediffusion/blob/main/scripts/txt2img.py#L220))
```python
state_dict = torch.load(path/to/ldm/checkpoint_000.pth)['ldm_decoder']
msg = model.first_stage_model.load_state_dict(state_dict, strict=False)
print(f"loaded LDM decoder state_dict with message\n{msg}")
print("you should check that the decoder keys are correctly matched")
```

You should also comment the lines that add the post-hoc watermark of SD: `img = put_watermark(img, wm_encoder)`.

For instance with: [WM weights of SD2 decoder](https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth), the weights obtained after running [this command](https://justpaste.it/ae93f). In this case, the state dict only contains the 'ldm_decoder' key, so you only need to load with `state_dict = torch.load(path/to/ckpt.pth)`



### Decode

The `decode.ipynb` notebook contains a full example of the decoding and associated statistical test.

## Acknowledgements

This code is based on the following repositories:

- https://github.com/Stability-AI/stablediffusion
- https://github.com/SteffenCzolbe/PerceptualSimilarity

To train the watermark encoder/extractor, you can refer to the following repository https://github.com/ando-khachatryan/HiDDeN. 
Changes were made from this codebase and will be made available soon.

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
