Environment: 
	I am using Anaconda Python software and specifically Jupyter notebooks on Windows. I am not sure if this would be 
  different if you are running it on macbook so if yall want to try it and edit it in the future please feel free to. 

1. Overview
  RedTell is a machine learning pipeline for red blood cell image segmentation and classification. It uses a Mask R-CNN deep learning model to detect individual cells in microscopy images, extract 130+ morphological and intensity features per cell, and classify them using Decision Tree, Random Forest, and LightGBM classifiers.
  •	Pipeline Steps:
    1.	Segment — detect every cell in your images using Mask R-CNN
    2.	Extract Features — measure 130+ shape and texture features per cell
    3.	Annotate — manually label a sample of cells
    4.	Classify — train models to auto-label all remaining cells
2. Prerequisites
  •	Required Software:
    o	Anaconda (Python environment manager)
    o	Git (for cloning the repository)
    o	Microsoft C++ Build Tools (required for pycocotools)
    o	Jupyter Notebook (included with Anaconda)
  •	Install Microsoft C++ Build Tools:
  o	Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  o	During installation, check "Desktop development with C++" and click Install.
  o	Restart your computer after installation (~4GB download).
  •	Clone the Repository:
  o	git clone https://github.com/marrlab/RedTell C: (YOUROWNPATH)\Python\RedTell
3. Environment Setup
  Step 1 — Create Conda Environment (run one at a time in Anaconda Prompt):
    o	conda create -n redtell python=3.9 -y
    o	conda activate redtell
    o	python --version
      	You should see Python 3.9.x. The (redtell) prefix must appear at the start of your prompt before continuing.
  Step 2 — Install Dependencies:
    o	cd C: (YOUROWNPATH) \Python\RedTell
    o	python -m pip install --upgrade pip setuptools wheel
    o	conda install numpy=1.23 -y
    o	pip install -r requirements.txt --ignore-installed numpy
  Step 3 — Install PyTorch:
    o	pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cpu
  Step 4 — Add Kernel to Jupyter:
    o	pip install ipykernel
    o	python -m ipykernel install --user --name redtell --display-name "RedTell (Python 3.9)"
      	Open Jupyter and switch the kernel to "RedTell (Python 3.9)" via Kernel → Change Kernel.
4. Data Folder Structure
  •	RedTell/my_data/images/masks/segmentation_results/
  o	images ← put your .tif images here
  o	masks ← created automatically by segment
  o	segmentation_results← created automatically
5. Running the Pipeline
  •	Always run Cell 1 and Cell 2 at the start of every Jupyter session.
  •	Cell 1 — Navigate to RedTell folder:
    o	import os
    o	os.chdir(r'C: (YOUROWNPATH) \Python\RedTell')
    o	print(os.getcwd())
  •	Cell 2 — Set Python executable:
    o	import sys
    o	python_exe = sys.executable
      o	print(python_exe)
      •	Step 1 — Segmentation (Detects every individual cell using Mask R-CNN)
      o	!"{python_exe}" redtell.py --funct segment --data my_data
      o	To use Jesus’s Classifier model edit the name file 
        	!"{python_exe}" redtell.py --funct segment --data my_data --model my_custom_model
      o	Output: my_data/masks/ and my_data/segmentation_results/
    •	Step 2 — Feature Extraction
      o	Measures 130+ shape and texture features per cell:
      o	!"{python_exe}" redtell.py --funct extract_features --data my_data --channel mask bf
        	Output: my_data/features.csv (one row per cell, 76+ columns)
    •	Step 3 — Annotation (Randomly selects 200 cells for manual labeling)
      o	!"{python_exe}" redtell.py --funct annotate --data my_data --num_cells 200
        	Output: my_data/annotations/ (cell images) and my_data/annotations.csv
      o	After running, open annotations.csv in Excel and fill in the label column for each cell (e.g. Healthy, Crenated, Dead, Invalid). Save before proceeding.
      o	For my own record I also save it as an excel file, Open both the color labeled photos and the original photos
        	Every time I label a cell I take a screenshot of each cell the color labeled one and the original so that it can be referenced back by other people. 
    •	Step 4 — Merge Annotations with Features
      o	import pandas as pd
      o	features = pd.read_csv(r'C: (YOUROWNPATH)\Python\RedTell\my_data\features.csv')
      o	annotations = pd.read_csv(r'C: (YOUROWNPATH)\Python\RedTell\my_data\annotations.csv')
      o	merged = features.merge(annotations[['image', 'cell_id', 'label']], on=['image', 'cell_id'], how='left')
      o	merged.to_csv(r'C:(YOUROWNPATH)\Python\RedTell\my_data\features_labeled.csv', index=False)
      o	print("Labeled cells:", merged['label'].notna().sum())
    •	Step 5 — Classification
      o	Trains Decision Tree, Random Forest, and LightGBM models:
      o	python
      o	import os
      o	os.chdir(r'C: (YOUROWNPATH)\Python\RedTell\classification\src')
      o	!"{python_exe}" main.py -f "..\..\my_data\features_labeled.csv" --label label --cell cell_id -o "..\..\my_data\classification_results" -p random
      o	Output: classification_results/ with predictions, feature importance, and learning curves for all 3 models.
    •	Step 6 — View Results
      o	import os, pandas as pd
      o	os.chdir(r'C: (YOUROWNPATH)\Python\RedTell')
      o	results_base = r'C: (YOUROWNPATH)\Python\RedTell\my_data\classification_results'
      o	dt   = pd.read_csv(results_base + r'\features_labeled.csv\Decision Tree\inference_predictions.csv')
      o	rf   = pd.read_csv(results_base + r'\features_labeled.csv\Random Forest\inference_predictions.csv')
      o	lgbm = pd.read_csv(results_base + r'\features_labeled.csv\LightGBM\inference_predictions.csv')
      o	print("Decision Tree:", dt['y_pred'].value_counts().to_dict())
      o	print("Random Forest:", rf['y_pred'].value_counts().to_dict())
      o	print("LightGBM:",      lgbm['y_pred'].value_counts().to_dict())
6. Training a Custom Segmentation Model
  •	Required folder structure:
    o	my_data/Images/Ground-Truth/        
      	Images← raw microscopy images (.tif)
      	Ground_Truth← hand-labeled segmentation masks (.tif)
  •	Each mask must have the same filename as its image. Pixel values represent cell IDs (0 = background, 1 = first cell, etc.). Minimum 25 annotated images required.
  •	Run training:
    o	!"{python_exe}" redtell.py --funct train_segmentation --data my_data --model my_custom_model
  •	Training takes several hours on CPU (~4 min per epoch, 15 epochs).
  •	Move the model file after training:
    o	import shutil
    o	shutil.move(
    o	r'C: (YOUROWNPATH)\Python\RedTell\my_custom_model.model',
    o	r'C: (YOUROWNPATH)\Python\RedTell\segmentation\models\my_custom_model.model'
    o	)
7. Issues I ran into
  o	Issue 1 — Python Version Incompatibility
    	Error: AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
    	Cause: Python 3.12+ removed ImpImporter. RedTell requires Python 3.9.
    	Fix:
      	conda create -n redtell python=3.9 -y
      	conda activate redtell
  o	Issue 2 — pycocotools Build Failure
    	Error: error: Microsoft Visual C++ 14.0 or greater is required
    	Cause: pycocotools must be compiled from source and requires C++ Build Tools.
    	Fix: 
      	Install Microsoft C++ Build Tools, check "Desktop development with C++", restart computer.
  o	Issue 3 — pip Cannot Upgrade Inside Jupyter
    	Error: ERROR: To modify pip, please run: python.exe -m pip install --upgrade pip
    	Cause: Jupyter intercepts pip upgrade commands for its own environment.
    	Fix: 
      	Run the upgrade from Anaconda Prompt instead:
      	python -m pip install --upgrade pip setuptools wheel
  o	Issue 4 — torch Not Found When Running redtell.py
    	Error: ModuleNotFoundError: No module named 'torch'
    	Cause: The ! command uses a different Python than the Jupyter kernel.
     Fix:
      	import sys
      	python_exe = sys.executable
    	!"{python_exe}" redtell.py --funct segment --data my_data
  o	Issue 5 — Fake torch Module Shadowing Real PyTorch
    	Error: AttributeError: module 'torch' has no attribute '__version__'
    	Cause: A namespace package named torch existed in the environment root, found before the real PyTorch.
     Fix:
      	pip uninstall torch torchvision -y
      	pip cache purge
      	pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cpu
  o	Issue 6 — Windows Path Bug (split("/") fails)
    	Error: FileNotFoundError: 'my_data\\masks\\my_data\\images\\Control1_1.tif'
    	Cause: RedTell was written for Linux/Mac. split("/")[-1] fails on Windows backslash paths.
    	Fix: 
      	Replace all split("/")[-1] with os.path.basename() in predict.py, extract_features.py, generate_annotations.py, and datasets.py.
  o	Issue 7 — matplotlib Backend Incompatibility
    	Error: AttributeError: 'RcParams' object has no attribute '_get'
    	Cause: matplotlib_inline backend has a compatibility bug with this version of matplotlib.
    	Fix: 
      	Add to the top of segmentation_utils.py and main.py:
      	import matplotlib
      	matplotlib.use('Agg')
  o	Issue 8 — Mask Size Mismatch (572x572 vs 562x562)
    	Error: RuntimeError: Input "labelImage" has size [572, 572] which does not match [562, 562]
    	Cause: The model resizes images to 572x572 but original images may differ in size.
    	Fix: 
      	Add resize step in extract_features.py:
        	if masks.shape != img.shape:
        	masks = np.array(Image.fromarray(masks).resize(
        	(img.shape[1], img.shape[0]), Image.NEAREST))
  o	Issue 9 — "No such function" for feature_extraction
    	Error: No such function.
    	Cause: The correct function name is extract_features, not feature_extraction.
    	Fix:
      	!"{python_exe}" redtell.py --funct extract_features --data my_data --channel mask bf
  o	Issue 10 — Classification KeyError: None (group_column)
    	Error: KeyError: 'None' not found in columns
    	Cause: When no --group argument is passed, group_column_name is None and is used as a dictionary key.
    	Fix: 
      	Add None checks in data_ingest.py and data_sets.py:
      	if data_set_meta_data.group_column_name is not None:
      	types[data_set_meta_data.group_column_name] = ...
  o	Issue 11 — image Column Treated as Numeric Feature
    	Error: ValueError: could not convert string to float: 'my_data\\images\\Control1_1.tif'
    	Cause: The image column contains file paths but the classifier tries to convert it to float32.
    	Fix: 
      	Add 'image' to non_predictor_column_names in data_ingest.py:
      	return {self.label_column_name, self.group_column_name, self.cell_id_column_name, 'image'}
  o	Issue 12 — Invalid Bounding Boxes During Training
    	Error: AssertionError: All bounding boxes should have positive height and width
    	Cause: Some ground truth masks contain degenerate single-pixel cells with zero-size bounding boxes.
    	Fix: 
      	Add validation in datasets.py to skip invalid cells:
      	if cell_bbox[2] > cell_bbox[0] and cell_bbox[3] > cell_bbox[1] and cell_area > 0:
      	masks.append(cell_mask)
      	bounding_boxes.append(cell_bbox)
      	areas.append(cell_area)
  o	Issue 13 — Windows Multiprocessing Error During Training
    	Error: RuntimeError: Couldn't open shared event
    	Cause: PyTorch DataLoader spawns worker processes that conflict on Windows shared memory.
    	Fix: 
      	Set num_workers=0 in train.py:
      	# Replace all num_workers=2 with:
      	num_workers=0
  o	Issue 14 — Model File Saved to Wrong Location
    	Error: FileNotFoundError: 'segmentation/models\\my_custom_model.model'
    	Cause: After training, RedTell saves the model to the root folder instead of segmentation/models/.
    	Fix:
      	import shutil
      	shutil.move(
      	r'C:\ (YOUROWNPATH)\Python\RedTell\my_custom_model.model',
      	r'C:\ (YOUROWNPATH)\Python\RedTell\segmentation\models\my_custom_model.model'
      	)
  o	Issue 15 — Training Images Not Found
    o	Error: Training starts but processes 0 images
    o	Cause: train.py looks in images_dir/images/*.tif but training images are in my_data/Images/.
    o	Fix: 
      	Update the glob pattern in train.py:
      	# Replace:
      	images_paths = glob.glob(os.path.join(images_dir, "images", "*.tif"))
      	# With:
      	images_paths = glob.glob(os.path.join(images_dir, "Images", "*.tif"))
