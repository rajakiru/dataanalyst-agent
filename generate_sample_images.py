"""
Generate synthetic flower images for testing image processing pipeline.
Creates simple colored circles and shapes to simulate flower datasets.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import random


def generate_flower_images(output_dir: str = "data/images", num_images: int = 20):
    """
    Generate synthetic flower images for testing.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define flower types (by color)
    flower_types = {
        "rose": [(255, 0, 0), (255, 100, 100), (200, 0, 0)],          # Red
        "daisy": [(255, 255, 255), (255, 255, 0), (100, 100, 0)],     # White/Yellow
        "tulip": [(255, 0, 100), (200, 50, 150), (255, 100, 150)],    # Pink/Magenta
        "sunflower": [(255, 200, 0), (200, 150, 0), (100, 100, 0)],   # Yellow
        "lavender": [(150, 100, 200), (200, 150, 255), (100, 50, 150)],  # Purple
    }
    
    flower_names = list(flower_types.keys())
    
    print(f"Generating {num_images} synthetic flower images in {output_dir}...")
    
    for i in range(num_images):
        # Pick random flower type
        flower_type = random.choice(flower_names)
        colors = flower_types[flower_type]
        
        # Create image
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw flower with petals
        center_x, center_y = 112, 112
        petal_radius = 30
        petal_distance = 60
        
        # Number of petals
        num_petals = random.randint(5, 8)
        
        for petal_idx in range(num_petals):
            angle = (2 * np.pi * petal_idx) / num_petals + random.uniform(-0.2, 0.2)
            petal_x = center_x + petal_distance * np.cos(angle)
            petal_y = center_y + petal_distance * np.sin(angle)
            
            # Draw petal (oval)
            petal_color = random.choice(colors)
            draw.ellipse(
                (petal_x - petal_radius, petal_y - petal_radius,
                 petal_x + petal_radius, petal_y + petal_radius),
                fill=petal_color,
                outline=(0, 0, 0)
            )
        
        # Draw center (yellow)
        center_radius = 15
        draw.ellipse(
            (center_x - center_radius, center_y - center_radius,
             center_x + center_radius, center_y + center_radius),
            fill=(255, 255, 0),
            outline=(200, 200, 0)
        )
        
        # Add some variation with random noise
        img_array = np.array(img)
        noise = np.random.randint(-10, 10, img_array.shape)
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        image_filename = f"{flower_type}_flower_{i:03d}.png"
        image_path = output_path / image_filename
        img.save(image_path)
        
        if (i + 1) % max(1, num_images // 5) == 0:
            print(f"  Generated {i + 1}/{num_images} images")
    
    print(f"✓ Successfully generated {num_images} synthetic flower images")
    print(f"  Location: {output_path.absolute()}")
    print(f"  Dataset: {', '.join(set([os.path.splitext(f)[0].split('_')[0] for f in os.listdir(output_path) if f.endswith('.png')]))} flowers")


if __name__ == "__main__":
    generate_flower_images(num_images=20)
