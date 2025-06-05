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
    class_distribution: dict = {0: 'label1', 1: 'label2', 2: 'label3', 3: 'label4', 4: 'label5', 5: 'label6', 6: 'label7'}
    class_names: list = 'label1', 'label2', 'label3', 'lanel4', 'label5', 'label6', 'label7'
