import os
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from age_gender_race import MultiTask

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

# Load the trained model
model = MultiTask()
model.load_state_dict(torch.load('multi_task_regressor_final.pth', map_location=torch.device('cpu')))
model.eval()

test_folder = 'test'
test_images = os.listdir(test_folder)

# Choose random images
random_images = random.sample(test_images, 5)

for image_filename in random_images:
    print(image_filename)
    age_str = image_filename.split('_')[0]
    name = image_filename.split('_')[1:-1]
    person_name = ' '.join(name)
    real_age = age_str

    # Load the image
    image_path = os.path.join(test_folder, image_filename)
    image = Image.open(image_path).convert('RGB')

    input_image = transform(image).unsqueeze(0)

    # Perform inference
    age_pred, gender_pred, race_pred = model(input_image)

    predicted_age = int(torch.round(age_pred).item())
    predicted_gender = 'Male' if torch.argmax(gender_pred) == 0 else 'Female'
    race_mapping = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
    predicted_race = race_mapping[int(torch.argmax(race_pred))]

    plt.imshow(image)
    plt.title(f"Predicted Age: {predicted_age}, Gender: {predicted_gender}, Race: {predicted_race}\n"
              f"{person_name}, Real Age: {real_age}")
    plt.axis('off')
    plt.show()


# End of script

