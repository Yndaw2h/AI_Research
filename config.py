from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Dataset parameters
    data_path: str = "Path to your dataset"
    image_size: int = 640
    batch_size: int = 16
    num_workers: int = 4
    
    # Model parameters
    num_classes: int = 7
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    warmup_epochs: int = 3
    weight_decay: float = 0.0005
    
    # Validation parameters
    val_interval: int = 5
    save_interval: int = 10
    
    # Hardware
    device: str = 'cpu'
    
    # Experiment tracking
    exp_name: str = 'detection'
    save_dir: str = 'experiments/'

@dataclass
class DatasetInfo:
    """Dataset statistics and information"""
    total_images: int = 158
    train_images: int = 126
    val_images: int = 15
    test_images: int = 17
    class_distribution: dict = {0: 'paper', 1: 'mouse', 2: 'pen', 3: 'cup', 4: 'headphones', 5: 'remote', 6: 'background'}
    class_names: list = 'paper', 'mouse', 'pen', 'cup', 'headphones', 'remote', 'background'
