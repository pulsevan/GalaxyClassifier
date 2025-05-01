# GalaxyClassifier

This project utilizes supervised machine learning, specifically the ConvNeXt_tiny architecture via the fastai library (built on PyTorch), to classify images of galaxies. The primary goal is to identify specific types of galaxies, such as polar ring galaxies, using image data sourced from the Hyper-Suprime Cam Subaru Sky Survey (HSC-SSP).

![NGC 660 Polar Ring Galaxy](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/NGC_660_Polar_Galaxy_Gemini_Observatory.jpg/400px-NGC_660_Polar_Galaxy_Gemini_Observatory.jpg)
*Example of a Polar Ring Galaxy (NGC 660). Credit: Gemini Observatory/AURA*

## Project Overview

The core of this project is a Jupyter Notebook (`classifier5.ipynb` or similar) that performs the following steps:

1.  **Data Pre-processing:** Includes checks to identify and remove corrupted or truncated image files to ensure smooth model training and prediction.
2.  **Data Splitting:** Uses `scikit-learn` to randomly split the manually classified training images into training and validation sets.
3.  **Model Training:**
    * Employs the `fastai` library's `vision_learner` with a `ConvNeXt_tiny` pre-trained model.
    * Optionally uses `lr_find()` to determine an optimal learning rate.
    * Trains the model using `fine_tune` with callbacks like `EarlyStoppingCallback` and `SaveModelCallback` for efficiency and saving the best model state.
    * Leverages Nvidia CUDA for GPU acceleration if available.
4.  **Model Evaluation:** Includes code to generate a confusion matrix to visualize the model's performance on the validation set.
5.  **Model Export/Import:** Allows saving the trained model (`.pkl` file) for later use and loading pre-trained models.
6.  **Prediction:** Runs the trained or imported model on a larger dataset of galaxy images.
7.  **Probability Scaling:** Uses temperature scaling (`temperature_softmax`) to adjust prediction probabilities if needed.
8.  **Catalog Generation:** Creates a CSV file (`catalog_small.csv` or similar) listing each classified image, its predicted class, and the probability score for each possible class.

## Technology Stack

* **Primary Libraries:**
    * fastai
    * PyTorch
    * scikit-learn
    * pandas
    * Pillow (PIL Fork)
* **Hardware Acceleration:** Nvidia CUDA (optional, but recommended for performance)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:** Ensure you have Python installed. It's recommended to use a virtual environment. This may require using different versions of PyTorch (In the case of a 50 series GPU, fastai and PyTorch may or may not be updated correctly as of 4/30/25!).
    ```bash
    pip install pandas pathlib shutil fastai scikit-learn torch torchvision torchaudio Pillow
    ```
3.  **CUDA Setup (Optional):** If you have an Nvidia GPU, ensure you have the appropriate CUDA toolkit and cuDNN versions installed that are compatible with your PyTorch installation. Verify with:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    ```
4.  **Data:**
    * Place your manually classified training images into subfolders within a `training data/` directory (e.g., `training data/polar_ring/`, `training data/elliptical/`, etc.).
    * Place the larger set of images to be classified into an `ims/` directory (or adjust the path in the notebook, e.g., `small_test_ims`).

## Usage

1.  **Open the Jupyter Notebook:** Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
    Navigate to and open the `classifier5.ipynb` (or your notebook's name).

2.  **Run the Notebook:**
    * **Imports:** Run the first cell to import necessary libraries.
    * **(Optional) Pre-processing:** Run the cells in the "Quick pre-processing" section if you need to check for and remove corrupted images in the `ims` folder.
    * **Training (if needed):**
        * Run the "Create training sets" cell to split your `training data` into `train` and `valid` subdirectories.
        * Run the "Create learning model" cells to define the `DataLoaders` and the `vision_learner`. Adjust `bs` (batch size) based on your GPU VRAM.
        * Run the training cell (`learn.fine_tune(...)`).
        * (Optional) Run the confusion matrix cell to evaluate.
        * Run the `learn.export(...)` cell to save your trained model (e.g., as `full_model.pkl`).
    * **Prediction (Using a Pre-trained Model):**
        * If you have a saved model (`.pkl` file), skip the training steps.
        * Run the cells in the "Import Model and Full Dataset Test" section, making sure the `load_learner('your_model_name.pkl')` line points to your saved model file. Verify CUDA is active if applicable.
        * Adjust the `full_dataset` path (e.g., `Path('ims')`) to point to the directory containing the images you want to classify.
        * Run the subsequent cells to perform predictions and generate the `catalog.csv` file.

## Output

The primary output is a CSV file (e.g., `catalog_small.csv`) containing:

* `filename`: The name of the image file.
* `predicted_class`: The class the model predicted for the galaxy.
* One column for each possible galaxy class, showing the calculated probability for that class.



