#!/usr/bin/env python3
"""
Combine two images by applying the style of the second image (argv[2]) onto
the content of the first image (argv[1]). The final image has the same size
(width and height) as the first input image (argv[1]).

We also use scikit-learn to generate an "intelligent" output filename by examining
similarities in the base names of the two input files.

Usage:
  python3 combine_side_by_side.py <image1> <image2> [--style_weight FLOAT] [--content_weight FLOAT]

Example:
  python3 combine_side_by_side.py content.jpg style.jpg --style_weight 40000 --content_weight 400

Note:
  - This script uses PyTorch to perform Neural Style Transfer. 
    However, we deliberately only apply style loss to early convolution layers
    (conv_1 and conv_2). This captures mostly color and simple textures from argv[2].
  - We place content loss in a deeper layer (conv_4) and use a stronger content_weight
    plus a weaker style_weight, ensuring the distinct structural features of argv[1]
    remain intact.
  - Both the content image and the style image are forced to the same spatial
    dimensions to avoid tensor shape mismatches during loss calculations.
  - Various safety checks are included to help ensure robust usage.
  - By default, we set style_weight=4e4 and content_weight=400 (this is 100% higher
    than the previous defaults of 2e4 and 200).
    You may adjust these by using the optional command-line arguments.

Requirements:
  - torch, torchvision
  - scikit-learn
  - Pillow
"""

import sys
import os
import argparse
from PIL import Image

# In newer versions of Pillow (9.1.0+), use Image.Resampling.LANCZOS
# instead of Image.ANTIALIAS, which is now deprecated.
try:
    from PIL import ImageResampling
    RESAMPLE_FILTER = ImageResampling.LANCZOS
except ImportError:
    # Fallback if for some reason PIL < 9.1 is used (or different import structure).
    # In older versions, ANTIALIAS is equivalent to LANCZOS in functionality.
    RESAMPLE_FILTER = Image.LANCZOS

# scikit-learn for "intelligent" filename creation
# (pip install scikit-learn if missing)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models


def intelligent_output_name(path1, path2):
    """
    Use scikit-learn (Tfidf) to generate an "intelligent" output name
    based on similarity of the filenames (excluding extensions).

    :param path1: str, file path to the first image
    :param path2: str, file path to the second image
    :return: str, proposed output filename (with .jpg extension)
    """
    base1 = os.path.splitext(os.path.basename(path1))[0]
    base2 = os.path.splitext(os.path.basename(path2))[0]

    # Vectorize the two base names
    vectorizer = TfidfVectorizer()
    docs = [base1, base2]
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Calculate similarity
    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # Generate a combined name reflecting the similarity
    # Example: 'content_AND_style_SIM0.12.jpg'
    combined_name = f"{base1}_AND_{base2}_SIM{sim:.2f}.jpg"
    return combined_name


# ------------
# Style Transfer Helpers
# ------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_to_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor on the selected device.
    """
    transform_to_tensor = transforms.ToTensor()
    tensor = transform_to_tensor(pil_image).unsqueeze(0)
    return tensor.to(device, torch.float)


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor (B=1) back to a PIL image.
    """
    # Clone to avoid modifying the original tensor
    image = tensor.cpu().clone().squeeze(0)
    # Convert back to PIL
    transform_to_pil = transforms.ToPILImage()
    return transform_to_pil(image)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


def gram_matrix(input_tensor):
    """
    Compute the Gram matrix of a given tensor.
    """
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """
    Normalization layer using the mean and std of ImageNet.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # shape [C, 1, 1]
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn,
                               normalization_mean, normalization_std,
                               style_img, content_img):
    """
    Build the style transfer model with content and style losses inserted.

    We deliberately:
      - place content loss at 'conv_4' (capturing deeper structure)
      - place style loss at 'conv_1' and 'conv_2' only (focusing on color/low-level details).
    """
    # We use only these layers for style
    style_layers = ['conv_1', 'conv_2']
    # We place content in conv_4 to preserve deeper structural details
    content_layers = ['conv_4']

    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            name = f'layer_{i}'

        model.add_module(name, layer)

        # Insert content loss in conv_4
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        # Insert style loss in conv_1, conv_2
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model after the last style/content loss
    for idx in range(len(model) - 1, -1, -1):
        if isinstance(model[idx], ContentLoss) or isinstance(model[idx], StyleLoss):
            break
    model = model[:(idx + 1)]

    return model, style_losses, content_losses


def run_style_transfer(
        cnn, 
        normalization_mean, normalization_std,
        content_img, style_img, input_img,
        num_steps=300,
        style_weight=4e4,  # NEW DEFAULT: previously 2e4
        content_weight=400 # NEW DEFAULT: previously 200
    ):
    """
    Run the style transfer optimization.

    - Default style_weight = 4e4  (previously 2e4, now 100% higher)
    - Default content_weight = 400 (previously 200, now 100% higher)
    """
    print("[INFO] Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std,
        style_img, content_img
    )

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print("[INFO] Optimizing..")
    run_step = [0]

    while run_step[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0.0
            content_score = 0.0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            run_step[0] += 1
            if run_step[0] % 50 == 0:
                print(f"[INFO] Step {run_step[0]}/{num_steps}")
                print(f"    Style Loss:   {style_score.item():.4f}")
                print(f"    Content Loss: {content_score.item():.4f}")

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def apply_style_transfer(
    image_content_path, 
    image_style_path, 
    output_path,
    style_weight=4e4, 
    content_weight=400
):
    """
    Apply the style of image_style_path onto the content of image_content_path,
    then save to output_path. Both images are forced to the same dimension.

    We only place style loss on early conv layers, so we mostly transfer color
    while preserving distinct structural features of argv[1].

    :param style_weight: float, how heavily to weight the style (default 4e4)
    :param content_weight: float, how heavily to weight the content (default 400)
    """
    # Safety checks for file existence
    if not os.path.isfile(image_content_path):
        raise FileNotFoundError(f"Content image not found: {image_content_path}")
    if not os.path.isfile(image_style_path):
        raise FileNotFoundError(f"Style image not found: {image_style_path}")

    # Open content as PIL and note size
    content_pil = Image.open(image_content_path)
    if content_pil.mode != "RGB":
        content_pil = content_pil.convert("RGB")
    content_size = content_pil.size  # (width, height)

    # Open style as PIL and force it to the same size
    style_pil = Image.open(image_style_path)
    if style_pil.mode != "RGB":
        style_pil = style_pil.convert("RGB")
    style_pil = style_pil.resize(content_size, RESAMPLE_FILTER)

    # Convert both to PyTorch tensors
    content_img = image_to_tensor(content_pil)
    style_img = image_to_tensor(style_pil)

    # We'll start our optimization from a copy of the content image
    input_img = content_img.clone()

    # Load a pretrained VGG19
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    # Normalization for pretrained VGG
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Run style transfer with user-specified (or new default) style_weight and content_weight
    output_tensor = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=300,
        style_weight=style_weight,
        content_weight=content_weight
    )

    # Convert the output tensor back to PIL
    final_img = tensor_to_image(output_tensor)

    # final_img should already be the same size as the content image
    final_img.save(output_path)
    print(f"[INFO] Style-transferred image saved at: {output_path}")


def main():
    """
    Main entry point: parse arguments (including optional style/content weights),
    generate an intelligent output name using sklearn, and apply style transfer
    from image2 onto image1.
    """
    parser = argparse.ArgumentParser(description="Combine two images by applying style transfer.")
    parser.add_argument("image1", help="Path to the content image.")
    parser.add_argument("image2", help="Path to the style image.")
    parser.add_argument(
        "--style_weight",
        type=float,
        default=4e4,
        help="Adjust style weight. Default is 40000.0 (increased 100% from 20000.0)."
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=400,
        help="Adjust content weight. Default is 400 (increased 100% from 200)."
    )

    args = parser.parse_args()

    image1_path = args.image1  # content image
    image2_path = args.image2  # style image

    # Create an "intelligent" output name
    output_file_path = intelligent_output_name(image1_path, image2_path)

    # Apply style transfer and save
    try:
        apply_style_transfer(
            image1_path, 
            image2_path, 
            output_file_path,
            style_weight=args.style_weight,
            content_weight=args.content_weight
        )
    except Exception as err:
        print(f"[ERROR] {err}")
        sys.exit(1)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()