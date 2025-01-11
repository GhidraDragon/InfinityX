#!/usr/bin/env python3

import sys
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

###############################################################################
# Helper Functions
###############################################################################

def check_image_quality(image_path, min_size=512):
    """
    Checks if the image at image_path meets a minimum size requirement.
    Raises a ValueError if the image is below the min_size in either dimension.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        if width < min_size or height < min_size:
            raise ValueError(
                f"Image '{image_path}' is too small (width={width}, height={height}). "
                f"Minimum dimension size required is {min_size}."
            )

def image_loader(image_name, device, imsize=512):
    """
    Load an image from disk, apply a transformation, and return it as a
    torch.FloatTensor suitable for processing by a neural network.
    """
    loader = transforms.Compose([
        transforms.Resize(imsize),   # scale imported image
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])     # transform it into a torch tensor
    
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def generate_output_filename(content_path, style_path):
    """
    Generate a descriptive output filename using the content and style image names, plus a timestamp.
    Example: output_contentImage_with_styleImage_YYYYMMDD_HHMMSS.jpg
    """
    content_name = os.path.splitext(os.path.basename(content_path))[0]
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output_{content_name}_with_{style_name}_{timestamp}.jpg"

def save_image(tensor, filename='output.jpg'):
    """
    Convert a torch.FloatTensor back into an image and save it.
    """
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(filename)

###############################################################################
# Optional Post-Processing to Enhance Aesthetics
###############################################################################

def postprocess_aesthetics(tensor, brightness_factor=1.1, contrast_factor=1.2):
    """
    Attempts to enhance the final image's subjective appeal by tweaking brightness and contrast.
    Increase or decrease the factors as needed. Remove if you prefer raw output.
    """
    unloader = transforms.ToPILImage()
    pil_img = unloader(tensor.squeeze(0).cpu().clone())

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness_factor)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast_factor)

    # Convert back to tensor
    loader = transforms.ToTensor()
    new_tensor = loader(pil_img).unsqueeze(0)
    return new_tensor.to(tensor.device, torch.float)

###############################################################################
# Content Loss
###############################################################################

class ContentLoss(nn.Module):
    """
    The content loss function compares the target feature map and the input.
    """
    def __init__(self, target, weight=1.0):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.weight = weight
        self.loss = 0

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x * self.weight, self.target * self.weight)
        return x

###############################################################################
# Style Loss
###############################################################################

def gram_matrix(input_tensor):
    """
    Compute the Gram matrix of a given tensor (style feature maps).
    The Gram matrix G is defined as: G[i,j] = sum_over_k( F[i,k] * F[j,k] )
    """
    b, f_map_num, h, w = input_tensor.size()
    features = input_tensor.view(b * f_map_num, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * f_map_num * h * w)

class StyleLoss(nn.Module):
    """
    The style loss function compares the target style Gram matrix to the input's Gram matrix.
    """
    def __init__(self, target_feature, weight=1.0):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.weight = weight
        self.loss = 0

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G * self.weight, self.target * self.weight)
        return x

###############################################################################
# Style Transfer Network
###############################################################################

class Normalization(nn.Module):
    """
    Normalization layer using the ImageNet mean and std for VGG networks.
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view(-1, 1, 1) to match [C x H x W] shape
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std  = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std,
                               style_img, content_img,
                               content_weight=1, style_weight=1000000):
    """
    Construct the style transfer model by inserting content loss and style loss
    modules at appropriate points of the pretrained CNN.
    """
    # Normalization module
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(style_img.device)

    content_losses = []
    style_losses = []

    # We'll create a new 'model' that includes normalization
    model = nn.Sequential(normalization)

    i = 0  # increment for each convolution
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            # replace with non-inplace ReLU
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            name = f'unknown_{i}'

        model.add_module(name, layer)

        # Add content loss
        if name == 'conv_4':
            target = model(content_img).detach()
            content_loss = ContentLoss(target, weight=content_weight)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        # Add style loss
        if name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, weight=style_weight)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model after the last loss
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:j+1]

    return model, style_losses, content_losses

###############################################################################
# Run Style Transfer
###############################################################################

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       num_steps=300, style_weight=1000000, content_weight=1):
    """
    Execute the gradient descent, updating the input image to match the content
    of content_img and style of style_img.
    """
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std,
        style_img, content_img, style_weight=style_weight, content_weight=content_weight
    )

    # Create a PyTorch parameter so we can optimize input_img directly
    input_img.requires_grad_(True)
    optimizer = optim.LBFGS([input_img])

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Iteration {run[0]}/{num_steps}")
                print(f"Style Loss : {style_score.item():4f} "
                      f"Content Loss: {content_score.item():4f}")
                print("-" * 10)

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

###############################################################################
# Main Entry Point
###############################################################################

def main():
    if len(sys.argv) < 3:
        print("Usage: python style_transfer.py <content_image> <style_image>")
        sys.exit(1)

    content_img_path = sys.argv[1]
    style_img_path = sys.argv[2]

    # Check image quality before proceeding
    try:
        check_image_quality(content_img_path, min_size=512)
        check_image_quality(style_img_path, min_size=512)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # You can tweak image size here if your memory is limited
    imsize = 512
    
    # Decide whether to use CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading images...")
    content_img = image_loader(content_img_path, device, imsize=imsize)
    style_img = image_loader(style_img_path, device, imsize=imsize)

    # Initialize the input to the content image (could also use noise)
    input_img = content_img.clone()

    # Use VGG19 pretrained on ImageNet
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    # Standard normalization for VGG19
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Run style transfer
    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=300,
        style_weight=1e6,
        content_weight=1
    )

    # Optional aesthetic enhancement: Slightly increase brightness and contrast
    output = postprocess_aesthetics(output, brightness_factor=1.1, contrast_factor=1.2)

    # Generate a descriptive filename for the final output
    output_filename = generate_output_filename(content_img_path, style_img_path)
    print(f"Saving output image as '{output_filename}'...")
    save_image(output, filename=output_filename)
    print("Style transfer complete!")

if __name__ == "__main__":
    main()