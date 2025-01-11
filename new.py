#!/usr/bin/env python3

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

###############################################################################
# 1. Command-line argument parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstructs an image by matching features at the most impactful layer, "
                    "or multiple layers, in a pretrained VGG19 network."
    )
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument(
        "-p", "--count",
        type=int,
        default=None,
        help="Number of top impactful layers to use for feature matching."
    )
    parser.add_argument(
        "-pa", "--all_layers",
        action="store_true",
        help="If set, use all convolutional layers for feature matching."
    )
    parser.add_argument(
        "--init_content",
        action="store_true",
        help="If set, initialize the target with the content image instead of random noise."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    input_image_path = args.input_image
    layer_count = args.count
    use_all_layers = args.all_layers
    use_content_init = args.init_content

    # Verify the path is valid
    if not os.path.isfile(input_image_path):
        print(f"Error: {input_image_path} is not a valid file path.")
        sys.exit(1)

    # Device selection (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ###########################################################################
    # 2. Set up transformations
    ###########################################################################
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to a manageable size
        transforms.ToTensor(),
        # Normalization stats for ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess input image
    content_image = Image.open(input_image_path).convert('RGB')
    content_tensor = transform(content_image).unsqueeze(0).to(device)

    ###########################################################################
    # 3. Load the VGG19 model and gather feature maps
    #
    #    Using the newer "weights=" argument instead of "pretrained=True" to
    #    avoid deprecation warnings in newer PyTorch versions.
    ###########################################################################
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    # We'll freeze the model parameters since we're not training the network
    for param in vgg.parameters():
        param.requires_grad = False

    ###########################################################################
    # 4. Identify the "most impactful" (or top N) convolutional layers
    #
    #    - If --all_layers is used, we pick all conv layers
    #    - If --count is specified, we pick that many top impactful layers
    #    - Otherwise, we just pick the single most impactful layer
    #
    #    "Impact" is a simple heuristic: run the image through the model
    #    layer-by-layer, track the L2 norm per layer, and rank them by that.
    ###########################################################################
    conv_layers = []
    for idx, layer in enumerate(vgg):
        if isinstance(layer, nn.Conv2d):
            conv_layers.append(idx)

    # Forward pass to compute each conv layer's L2 norm
    norms = []
    x = content_tensor.clone()
    for layer_idx in range(len(vgg)):
        x = vgg[layer_idx](x)
        if layer_idx in conv_layers:
            # Compute the L2 norm of the activation
            layer_norm = x.norm().item()
            norms.append((layer_idx, layer_norm))

    # Sort layers by descending norm
    norms.sort(key=lambda tup: tup[1], reverse=True)

    if use_all_layers:
        # Use all conv layers in descending order of impact
        selected_layer_indices = [lyr[0] for lyr in norms]
        print("Using ALL convolutional layers for feature matching.")
    elif layer_count is not None and layer_count > 0:
        # Use the top N layers
        selected_layer_indices = [lyr[0] for lyr in norms[:layer_count]]
        print(f"Using the top {layer_count} convolutional layers for feature matching.")
    else:
        # Default: pick only the single most impactful layer
        selected_layer_indices = [norms[0][0]]
        print(f"Using only the single most impactful layer index: {selected_layer_indices[0]}")

    print(f"Selected layer indices = {selected_layer_indices}")

    # Create a quick lookup dict for the layer norms
    layer_norm_dict = {layer_idx: norm_val for (layer_idx, norm_val) in norms}

    ###########################################################################
    # 5. Feature extraction function
    #
    #    We will return the activations for all selected layers
    ###########################################################################
    def get_selected_features(tensor, model, layers_to_extract):
        """Return a dict {layer_idx: features} for each layer in layers_to_extract."""
        features = {}
        output = tensor
        for idx, layer in enumerate(model):
            output = layer(output)
            if idx in layers_to_extract:
                features[idx] = output.clone()
            # If we've already passed the highest layer, we can stop early
            if idx >= max(layers_to_extract):
                break
        return features

    ###########################################################################
    # 6. Initialize a “target” image to optimize
    #
    #    By default, use random noise to encourage variety.
    #    If --init_content is set, revert to the old method of using the content image.
    ###########################################################################
    if use_content_init:
        print("Initializing target from content image.")
        target = content_tensor.clone().requires_grad_(True)
    else:
        print("Initializing target from random noise.")
        target = torch.rand_like(content_tensor, device=device, requires_grad=True)

    ###########################################################################
    # 7. Optimization setup
    ###########################################################################
    optimizer = optim.Adam([target], lr=0.03)
    mse_loss = nn.MSELoss()

    # Extract the “reference” feature maps from the original
    with torch.no_grad():
        reference_features = get_selected_features(
            content_tensor, vgg, selected_layer_indices
        )

    num_iterations = 200  # You can tweak this
    print_interval = 50   # Print progress every N iterations

    ###########################################################################
    # 8. Optimization loop
    #
    #    We multiply each layer's MSE loss by the layer’s norm so that the final
    #    results differ more significantly when different subsets of layers are selected.
    ###########################################################################
    for iteration in range(1, num_iterations + 1):
        optimizer.zero_grad()

        current_features = get_selected_features(target, vgg, selected_layer_indices)

        # Sum up MSE losses for all selected layers, weighted by layer impact
        total_loss = 0.0
        for layer_idx in selected_layer_indices:
            impact_weight = layer_norm_dict[layer_idx]
            total_loss += impact_weight * mse_loss(
                current_features[layer_idx],
                reference_features[layer_idx]
            )

        total_loss.backward()
        optimizer.step()

        # Optionally clamp or re-normalize to keep target stable (helps reduce artifacts)
        with torch.no_grad():
            # De-normalization for clamp
            for c, (mean, std) in enumerate([(0.485, 0.229),
                                             (0.456, 0.224),
                                             (0.406, 0.225)]):
                target[:, c] = target[:, c] * std + mean

            target.clamp_(0, 1)  # clamp to valid image range

            # Re-apply normalization
            for c, (mean, std) in enumerate([(0.485, 0.229),
                                             (0.456, 0.224),
                                             (0.406, 0.225)]):
                target[:, c] = (target[:, c] - mean) / std

        if iteration % print_interval == 0 or iteration == num_iterations:
            print(f"Iteration {iteration}, total loss = {total_loss.item():.4f}")

    ###########################################################################
    # 9. Post-processing & save the output
    ###########################################################################
    with torch.no_grad():
        # De-normalize one last time for saving
        final_img = target.clone()
        for c, (mean, std) in enumerate([(0.485, 0.229),
                                         (0.456, 0.224),
                                         (0.406, 0.225)]):
            final_img[:, c] = final_img[:, c] * std + mean

        final_img = final_img.clamp(0, 1)

    # Convert to PIL for saving
    final_img_pil = transforms.ToPILImage()(final_img.squeeze().cpu())

    # Generate a "smart filename" with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(input_image_path)
    name_without_ext, ext = os.path.splitext(base_name)
    output_filename = f"{name_without_ext}_impactful_{timestamp}.png"

    final_img_pil.save(output_filename)
    print(f"Saved reconstructed image as: {output_filename}")


if __name__ == "__main__":
    main()