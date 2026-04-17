"""
Image Dataset Processor for DataAnalyst-Agent

Handles: image loading → feature extraction → metadata → tabular merge
Architecture: Parallel processing with per-image error budgeting
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

# Try to import torch/torchvision for feature extraction
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("torch/torchvision not available. Image feature extraction will use fallback (color histograms).")

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Process image datasets into tabular format for analysis.
    
    Features:
    - Per-image error handling (independent, non-blocking)
    - Feature extraction (pre-trained CNN or hand-crafted)
    - Metadata extraction (EXIF, file stats)
    - Merge into CSV with coverage metrics
    """
    
    # Configuration
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    MAX_IMAGE_SIZE = (4096, 4096)  # Max dimensions
    MIN_IMAGE_SIZE = (32, 32)      # Min dimensions
    MAX_IMAGES = 10000             # Processing limit
    FEATURE_DIM = 512              # Feature vector dimension (ResNet)
    HISTOGRAM_BINS = 32            # For fallback color histogram
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize processor.
        
        Args:
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained ResNet18 for feature extraction."""
        if not TORCH_AVAILABLE:
            logger.info("Running in fallback mode (color histogram features)")
            return
        
        try:
            self.model = resnet18(pretrained=True)
            # Remove classification layer, keep features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            
            # Image preprocessing for ResNet
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            logger.info("Loaded ResNet18 feature extraction model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Using fallback.")
            self.model = None
    
    def process_images_directory(self, image_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process all images in a directory into a feature table.
        
        Args:
            image_dir: Path to directory containing images
            
        Returns:
            (feature_dataframe, metadata_dict)
            where metadata_dict contains:
                - processed_count: # images successfully processed
                - failed_count: # images that failed
                - coverage: % of images processed
                - feature_count: dimension of feature vectors
                - errors: list of (filename, error) tuples
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # Collect image files
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        image_files = sorted(set(image_files))  # Remove duplicates, sort
        
        if not image_files:
            raise ValueError(f"No valid images found in {image_dir}")
        
        if len(image_files) > self.MAX_IMAGES:
            logger.warning(f"Found {len(image_files)} images, capping at {self.MAX_IMAGES}")
            image_files = image_files[:self.MAX_IMAGES]
        
        logger.info(f"Processing {len(image_files)} images from {image_dir}")
        
        # Process images in parallel-like manner (sequential for simplicity)
        all_features = []
        all_metadata = []
        errors = []
        
        for idx, image_path in enumerate(image_files):
            try:
                # Load and validate image
                image = Image.open(image_path)
                
                # Extract features
                features = self._extract_features(image)
                all_features.append(features)
                
                # Extract metadata
                metadata = self._extract_metadata(image_path, image)
                all_metadata.append(metadata)
                
                if (idx + 1) % max(1, len(image_files) // 10) == 0:
                    logger.info(f"  Processed {idx + 1}/{len(image_files)} images")
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path.name}: {str(e)}")
                errors.append((image_path.name, str(e)))
        
        processed_count = len(all_features)
        failed_count = len(errors)
        total_count = len(image_files)
        coverage = 100.0 * processed_count / total_count if total_count > 0 else 0
        
        logger.info(f"Processing complete: {processed_count}/{total_count} successful ({coverage:.1f}%)")
        
        if processed_count == 0:
            raise ValueError("Failed to process any images from directory")
        
        # Create feature dataframe
        feature_names = [f"feature_{i}" for i in range(len(all_features[0]))]
        feature_df = pd.DataFrame(all_features, columns=feature_names)
        
        # Create metadata dataframe
        metadata_df = pd.DataFrame(all_metadata)
        
        # Merge features + metadata
        merged_df = pd.concat([feature_df, metadata_df], axis=1)
        
        # Add image index
        merged_df.insert(0, "image_id", range(len(merged_df)))
        
        metadata = {
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_count": total_count,
            "coverage_percent": coverage,
            "feature_dimension": len(all_features[0]),
            "errors": errors,
            "processing_timestamp": datetime.now().isoformat(),
            "source_directory": str(image_dir)
        }
        
        return merged_df, metadata
    
    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract feature vector from image.
        
        Returns: 1D array of shape (512,) for ResNet, or (96,) for histogram fallback
        """
        # Validate image dimensions
        if image.size[0] < self.MIN_IMAGE_SIZE[0] or image.size[1] < self.MIN_IMAGE_SIZE[1]:
            raise ValueError(f"Image too small: {image.size}. Min: {self.MIN_IMAGE_SIZE}")
        
        if image.size[0] > self.MAX_IMAGE_SIZE[0] or image.size[1] > self.MAX_IMAGE_SIZE[1]:
            logger.warning(f"Image too large: {image.size}. Max: {self.MAX_IMAGE_SIZE}")
            # Resize instead of rejecting
            image = image.resize(self.MAX_IMAGE_SIZE)
        
        # Convert to RGB if necessary (handle grayscale, RGBA, etc)
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Remove alpha channel
                image = image.convert('RGB')
            elif image.mode in ['L', 'P']:  # Grayscale or palette
                image = image.convert('RGB')
            else:
                image = image.convert('RGB')
        
        # Extract features using model or fallback
        if TORCH_AVAILABLE and self.model is not None:
            return self._extract_features_resnet(image)
        else:
            return self._extract_features_histogram(image)
    
    def _extract_features_resnet(self, image: Image.Image) -> np.ndarray:
        """Extract features using ResNet18."""
        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Flatten to 1D array
            feature_vector = features.squeeze().cpu().numpy()
            
            # Pad or truncate to FEATURE_DIM
            if len(feature_vector) < self.FEATURE_DIM:
                feature_vector = np.pad(feature_vector, (0, self.FEATURE_DIM - len(feature_vector)))
            else:
                feature_vector = feature_vector[:self.FEATURE_DIM]
            
            return feature_vector
        
        except Exception as e:
            logger.warning(f"ResNet extraction failed: {e}. Using histogram fallback.")
            return self._extract_features_histogram(image)
    
    def _extract_features_histogram(self, image: Image.Image) -> np.ndarray:
        """
        Fallback: Extract color histogram features.
        Returns: 96D vector (32 bins × 3 channels)
        """
        img_array = np.array(image.resize((128, 128)))  # Resize for faster histogram
        
        features = []
        
        # RGB histograms (or grayscale converted to L)
        if len(img_array.shape) == 3:
            for channel in range(min(3, img_array.shape[2])):
                hist, _ = np.histogram(img_array[:,:,channel], bins=self.HISTOGRAM_BINS, range=(0, 256))
                features.extend(hist)
        else:
            # Grayscale
            hist, _ = np.histogram(img_array, bins=self.HISTOGRAM_BINS, range=(0, 256))
            features.extend(hist)
        
        # Pad to consistent size if needed
        feature_vector = np.array(features[:96])
        if len(feature_vector) < 96:
            feature_vector = np.pad(feature_vector, (0, 96 - len(feature_vector)))
        
        return feature_vector
    
    def _extract_metadata(self, image_path: Path, image: Image.Image) -> Dict[str, Any]:
        """
        Extract metadata from image file.
        """
        metadata = {
            "filename": image_path.name,
            "file_size_kb": image_path.stat().st_size / 1024,
            "width_px": image.width,
            "height_px": image.height,
            "aspect_ratio": image.width / image.height if image.height > 0 else 0,
            "mode": image.mode,
            "format": image.format or "unknown"
        }
        
        # Try to extract EXIF data
        try:
            exif_data = image._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, f"tag_{tag_id}").lower()
                    # Only keep simple types
                    if isinstance(value, (int, float, str)):
                        metadata[f"exif_{tag_name}"] = value
        except:
            pass  # EXIF extraction optional
        
        return metadata


def process_images_to_csv(image_dir: str, output_csv: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    High-level API: Process images directory to CSV file.
    
    Args:
        image_dir: Path to directory with images
        output_csv: Output CSV path (auto-generated if None)
        
    Returns:
        (output_csv_path, metadata_dict)
    """
    processor = ImageProcessor()
    df_features, metadata = processor.process_images_directory(image_dir)
    
    # Generate output path if not provided
    if output_csv is None:
        output_csv = f"images_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Save to CSV
    df_features.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df_features)} image features to {output_csv}")
    
    # Save metadata
    metadata_file = output_csv.replace('.csv', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")
    
    return output_csv, metadata


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with data/images directory if it exists
    test_dir = Path("data/images")
    if test_dir.exists():
        csv_path, meta = process_images_to_csv(str(test_dir), "test_images_features.csv")
        print(f"\nProcessed to: {csv_path}")
        print(f"Coverage: {meta['coverage_percent']:.1f}%")
        print(f"Features per image: {meta['feature_dimension']}")
