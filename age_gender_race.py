import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


class AgeGenderRaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # List of (image_path, age_label, gender_label, race_label) tuples
        self._load_data()

    def _load_data(self):
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(self.root_dir, filename)
                age_label = int(filename.split('_')[0])
                gender_label = int(filename.split('_')[1])
                race_label = int(filename.split('_')[2])
                self.data.append((image_path, age_label, gender_label, race_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, age_label, gender_label, race_label = self.data[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, age_label, gender_label, race_label


class MultiTask(nn.Module):
    def __init__(self):
        super(MultiTask, self).__init__()
        # Load pre-trained ResNet50 model from torchvision
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Replace the final fully connected layer for multitask learning
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Output layers for age, gender, and race prediction
        self.age_fc = nn.Linear(512, 1)
        self.gender_fc = nn.Linear(512, 2)
        self.race_fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.model(x)

        age_pred = self.age_fc(x)
        gender_pred = self.gender_fc(x)
        race_pred = self.race_fc(x)

        return age_pred, gender_pred, race_pred


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def load_latest_checkpoint(checkpoint_dir, device):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Get list of checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch')]
    if not checkpoint_files:
        return MultiTask().to(device), 1

    # Find the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    # Load the latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MultiTask().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint['epoch'] + 1


def evaluate_model(model, val_loader, device):
    model.eval()
    age_preds, age_labels = [], []
    gender_preds, gender_labels = [], []
    race_preds, race_labels = [], []

    with torch.no_grad():
        for images, ages, genders, races in tqdm(val_loader, desc="Validation", ncols=70):
            images = images.to(device)
            age_pred, gender_pred, race_pred = model(images)

            # Append age predictions and labels
            age_preds.append(age_pred.cpu())
            age_labels.append(ages)

            # Append gender predictions and labels
            _, gender_pred_cls = torch.max(gender_pred, 1)
            gender_preds.append(gender_pred_cls.cpu())
            gender_labels.append(genders)

            # Append race predictions and labels
            _, race_pred_cls = torch.max(race_pred, 1)
            race_preds.append(race_pred_cls.cpu())
            race_labels.append(races)

    # Concatenate age predictions and labels
    age_preds = torch.cat(age_preds, dim=0)
    age_labels = torch.cat(age_labels, dim=0).view(-1, 1)

    # Calculate MAE for age prediction
    age_mae = torch.mean(torch.abs(age_labels - age_preds)).item()

    # Concatenate gender predictions and labels
    gender_preds = torch.cat(gender_preds, dim=0).view(-1, 1)
    gender_labels = torch.cat(gender_labels, dim=0).view(-1, 1)

    # Calculate gender prediction accuracy
    gender_accuracy = torch.mean((gender_preds == gender_labels).float()).item()

    # Concatenate race predictions and labels
    race_preds = torch.cat(race_preds, dim=0).view(-1, 1)
    race_labels = torch.cat(race_labels, dim=0).view(-1, 1)

    # Calculate race prediction accuracy
    race_accuracy = torch.mean((race_preds == race_labels).float()).item()

    return age_mae, gender_accuracy, race_accuracy


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    learning_rate = 0.00001
    num_epochs = 21
    checkpoint_dir = 'New folder/checkpoints'
    root_dir = './UTKFace'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = AgeGenderRaceDataset(root_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    model, start = load_latest_checkpoint(checkpoint_dir, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss functions
    age_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()
    race_criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(start, num_epochs + 1):
        model.train()
        running_loss = 0.0

        p_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100)

        for batch_idx, (images, ages, genders, races) in enumerate(p_bar):
            images, ages, genders, races = images.to(device), ages.float().to(device), genders.to(device), races.to(
                device)

            optimizer.zero_grad()
            age_pred, gender_pred, race_pred = model(images)

            age_loss = age_criterion(age_pred, ages.view(-1, 1))
            gender_loss = gender_criterion(gender_pred, genders)
            race_loss = race_criterion(race_pred, races)

            loss = age_loss + gender_loss + race_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            p_bar.set_postfix({'Average Loss': running_loss / (batch_idx + 1)})

        # scheduler.step()

        mae, gender_acc, race_acc = evaluate_model(model, val_loader, device)
        print(f"\tAge MAE = {mae:.4f}, Gender Accuracy = {gender_acc:.4f}, Race Accuracy = {race_acc:.4f}")

        save_checkpoint(model, optimizer, epoch, checkpoint_dir)

    torch.save(model.state_dict(), 'multi_task_regressor_final.pth')
