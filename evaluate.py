from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model
model = load_model('model_daun_pisang.h5')

# Label sesuai urutan class_indices saat training
label_mapping = ['penyakit_panama', 'penyakit_sigatoka', 'sehat']

# Generator untuk evaluasi
dataset_dir = 'dataset_daun_pisang'
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Prediksi
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Confusion Matrix & Classification Report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=label_mapping)

print("📊 Confusion Matrix:")
print(cm)
print("\n📋 Classification Report:")
print(report)

# Visualisasi Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping, yticklabels=label_mapping)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
