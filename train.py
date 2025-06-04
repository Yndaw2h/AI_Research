import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from math import cos, pi
from config import TrainingConfig

# Блок згортки для екстракції ознак
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# Клас датасету для задачі детекції
class DetectionDataset(Dataset):
    def __init__(self, image_paths, annotations):
        self.images = image_paths  # Шляхи до зображень
        self.annotations = annotations  # Анотації
        
    def __getitem__(self, idx):
        # Завантаження зображення
        image = self.load_image(self.images[idx])
        
        # Отримання межових рамок (boxes) та міток (labels)
        boxes = self.annotations[idx]['boxes']
        labels = self.annotations[idx]['labels']
        
        # Попередня обробка зображення та кодування міток
        image = self.normalize(image)
        targets = self.encode_targets(boxes, labels)
        return image, targets
    
    def __len__(self):
        return len(self.images)  # Кількість зображень у датасеті
    
    def encode_targets(self, boxes, labels):
        # Кодування міток для YOLO-формату
        grid_size = TrainingConfig.grid_size
        encoded = torch.zeros(grid_size, grid_size, 5 + TrainingConfig.num_classes)
        
        for box, label in zip(boxes, labels):
            # Обчислення координат клітинки сітки
            cell_x = int(box[0] * grid_size)
            cell_y = int(box[1] * grid_size)
            
            # Заповнення інформації про рамку, об'єктність та клас
            encoded[cell_y, cell_x, :4] = box
            encoded[cell_y, cell_x, 4] = 1
            encoded[cell_y, cell_x, 5 + label] = 1
        return encoded


# Визначення моделі детектора
class ObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Хребет (backbone) для екстракції ознак
        self.backbone = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        # Головка детектора (головний шар)
        self.detector = nn.Sequential(
            ConvBlock(256, 512),
            nn.Conv2d(512, num_classes + 4, kernel_size=1)
        )
    
    def forward(self, x):
        features = self.backbone(x)  # Екстракція ознак
        return self.detector(features)  # Прогнози


# Функція втрати для задачі детекції
class DetectionLoss:
    def __init__(self):
        self.mse = nn.MSELoss()  # Втрата для рамок
        self.bce = nn.BCEWithLogitsLoss()  # Втрата для об'єктності та класів
        
    def __call__(self, predictions, targets):
        # Розпакування прогнозів
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_cls = predictions[..., 5:]
        
        # Розпакування міток
        true_boxes = targets[..., :4]
        true_conf = targets[..., 4]
        true_cls = targets[..., 5:]
        
        # Обчислення втрат
        box_loss = self.mse(pred_boxes, true_boxes)
        conf_loss = self.bce(pred_conf, true_conf)
        cls_loss = self.bce(pred_cls, true_cls)
        
        return box_loss + conf_loss + cls_loss  # Загальна втрата


# Функція для розкладання швидкості навчання за косинусним законом
def cosine_annealing_lr(epoch, total_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))


# Основний цикл навчання
def train_model():
    config = TrainingConfig()  # Завантаження конфігурації
    dataset = DetectionDataset(config.image_paths, config.annotations)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = ObjectDetector(num_classes=config.num_classes)  # Ініціалізація моделі
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # Оптимізатор
    loss_fn = DetectionLoss()  # Функція втрат
    
    for epoch in range(config.epochs):
        model.train()  # Увімкнення режиму навчання
        total_loss = 0
        
        for batch_images, batch_targets in train_loader:
            predictions = model(batch_images)  # Передній прохід
            loss = loss_fn(predictions, batch_targets)  # Обчислення втрат
            
            optimizer.zero_grad()  # Очищення градієнтів
            loss.backward()  # Зворотний прохід
            optimizer.step()  # Оновлення параметрів
            total_loss += loss.item()  # Накопичення втрат
        
        print(f"Епоха {epoch+1}/{config.epochs}, Втрата: {total_loss/len(train_loader)}")


if __name__ == "__main__":
    train_model()
