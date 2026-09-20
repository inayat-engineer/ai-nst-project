# AI Neural Style Transfer (AdaIN)

A web application that applies the artistic style of one image (a painting, sketch, etc.) onto the content of another (like a photo), using **Adaptive Instance Normalization (AdaIN)** — a fast, feed-forward neural style transfer method.

Built with **PyTorch** (model) and **Flask** (web interface).

| Content | Style | Output |
|---|---|---|
| ![content](NST_Code/examples/brad_pitt.jpg) | ![style](NST_Code/examples/sketch.png) | ![output](NST_Code/examples/stylized_brad_pitt.jpg) |

## Features

- Upload any content image + any style image, get a stylized result in seconds
- Adjustable **Style Strength (alpha)** slider — control how much style vs. original content is preserved
- **Color-preserving post-processing** — keeps the content's natural skin tones/colors while still applying the style's texture, reducing AdaIN's typical color-blotching artifact
- Fast inference — no per-image optimization, unlike classic (Gatys-style) neural style transfer

## How It Works

The pipeline has three stages:

1. **Encoder (VGG19)** — a frozen, pretrained network that converts both the content and style image into feature maps
2. **AdaIN** — a formula (no learning involved) that replaces the content features' mean/std with the style features' mean/std:



3. **Decoder** — the only trained component. Mirrors the encoder's structure and reconstructs a real image from the mixed features

The **alpha** parameter blends the stylized and original content features:


A color-preserving step (YCbCr luminance/chrominance blending) is applied after stylization to reduce color artifacts on faces.

## Project Structure

ai-nst-project/
├── NST_Code/
│ ├── app.py # Flask web app (inference)
│ ├── train.py # Decoder training script
│ ├── vgg_normalised.pth # Pretrained VGG19 encoder weights
│ ├── utils/
│ │ ├── models.py # Encoder & Decoder architectures
│ │ └── utils.py # AdaIN math, mean/std helpers
│ ├── experiment/final_exp/ # Trained decoder checkpoint
│ ├── content_data/ # Sample content images
│ ├── style_data/ # Sample style images
│ ├── examples/ # Homepage example images
│ ├── templates/index.html # Frontend UI
│ └── requirements.txt
└── README.md



## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/<inayat-engineer>/ai-nst-project.git
cd ai-nst-project/NST_Code

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Then open **http://localhost:5000** in your browser:
1. Upload a **content** image and a **style** image
2. Adjust the **Style Strength** slider (0 = original content, 1 = full style)
3. Click **Transfer Style** and view/download the result

## Training Your Own Decoder

The repo ships with an already-trained decoder (`experiment/final_exp/decoder_final.pth`), but you can train your own on a custom dataset (e.g. on Kaggle/Colab with a free GPU):

```bash
python train.py --content_dir <path_to_content_images> --style_dir <path_to_style_images> --vgg vgg_normalised.pth
```

Training uses two loss functions:
- **Content Loss** — MSE between the re-encoded output and the target mixed features (preserves structure)
- **Style Loss** — MSE of mean/std across 4 VGG layers (matches color & texture)

## Tech Stack

- **PyTorch** & **torchvision** — model & image transforms
- **Flask**, **Flask-WTF** — web server & upload form
- **Pillow (PIL)** — image I/O and the color-preserving fix
- **NumPy**

## Credits

Originally based on [shradha-khapra/ai-nst-project](https://github.com/shradha-khapra/ai-nst-project), implementing the AdaIN method from *Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization" (2017)*.

## License

This project is for educational purposes as part of a university coursework/FYP exploration.