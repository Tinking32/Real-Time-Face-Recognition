import os
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import argparse
import matplotlib.pyplot as plt

def load_model(model_path):
    """
    Load the trained model with its metadata
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract model metadata
    class_to_idx = checkpoint.get('class_to_idx', {})
    classes = checkpoint.get('classes', list(class_to_idx.keys()))
    
    # Initialize model architecture (same as in training)
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    num_classes = len(classes)
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load state dictionary
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, classes, class_to_idx

def process_image(image_path):
    """
    Process an image to be suitable for model input
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the same image transformations used during validation
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = preprocess(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image

def predict(image_path, model, classes, device='cpu', top_k=3):
    """
    Predict the class of an image using a trained deep learning model
    """
    # Process the image
    image_tensor, original_image = process_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        
    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top k predictions and their indices
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Convert from indices to classes
    top_classes = [classes[idx] for idx in top_indices]
    
    return top_probs.tolist(), top_classes, original_image

def visualize_prediction(image, top_probs, top_classes):
    """
    Visualize the prediction results
    """
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)
    
    # Display the image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # Create horizontal bar plot of probabilities
    y_pos = range(len(top_classes))
    ax2.barh(y_pos, top_probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_classes)
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Predictions')
    
    plt.tight_layout()
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict face recognition with a trained model')
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--model_path', type=str, default='resnet50_face_recognition_finetuned.pth', 
                        help='Path to the trained model')
    parser.add_argument('--top_k', type=int, default=10, help='Return top K predictions')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model, classes, class_to_idx = load_model(args.model_path)
    model.to(device)
    
    # Predict on the image
    print(f"Performing inference on {args.image_path}...")
    top_probs, top_classes, original_image = predict(
        args.image_path, model, classes, device=device, top_k=args.top_k
    )
    
    # Print the results
    print("\nPrediction Results:")
    print("-" * 50)
    for i, (prob, class_name) in enumerate(zip(top_probs, top_classes)):
        print(f"{i+1}. {class_name}: {prob*100:.2f}%")
    
    # Visualize the results
    visualize_prediction(original_image, top_probs, top_classes)

if __name__ == '__main__':
    main()