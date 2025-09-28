import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_model.py <path_to_model.keras>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Extract model name from path
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join("model_visualizations", model_name)

    # Create folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # Print model summary in terminal
    print("\n========== MODEL SUMMARY ==========")
    model.summary()
    print("==================================\n")

    # Save high-resolution image of the model architecture
    output_file = os.path.join(output_dir, f"{model_name}_architecture.png")
    print(f"Saving model architecture to {output_file}...")

    plot_model(model,
               to_file=output_file,
               show_shapes=True,
               show_layer_names=True,
               expand_nested=True,
               dpi=150)  # High DPI for clear image

    print(f"Model architecture saved successfully in folder: {output_dir}")

if __name__ == "__main__":
    main()
