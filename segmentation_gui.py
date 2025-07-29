import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from config import Config
from model import create_model
from utils import get_colored_mask

class SegmentationTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Land Cover Segmentation Test")
        self.root.geometry("1400x900")
        
        # Variables
        self.model = None
        self.device = None
        self.current_image_path = None
        self.current_mask_path = None
        
        # Setup UI
        self.setup_ui()
        
        # Load model on startup
        self.load_model()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model status
        self.model_status = ttk.Label(control_frame, text="Model: Not loaded", foreground="red")
        self.model_status.pack(side=tk.LEFT)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Select Model", command=self.select_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="List Models", command=self.list_available_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cleanup Models", command=self.cleanup_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Select Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Select Mask (Optional)", command=self.select_mask).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Prediction", command=self.run_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def load_model(self):
        """Load the best available model"""
        try:
            self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
            
            # Find the best model checkpoint
            best_model_path = self.find_best_model()
            
            if not best_model_path or not os.path.exists(best_model_path):
                self.model_status.config(text="Model: No checkpoint found", foreground="red")
                messagebox.showwarning("Warning", "No model checkpoint found. Please train a model first or select one manually.")
                return
            
            self.status_bar.config(text="Loading model...")
            self.progress.start()
            
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model = create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.progress.stop()
            
            # Get model info
            epoch = checkpoint.get('epoch', 'Unknown')
            accuracy = checkpoint.get('metrics', {}).get('accuracy', 0)
            model_name = checkpoint.get('config', {}).get('model_name', Config.MODEL_NAME)
            
            status_text = f"Model: {model_name} (Epoch {epoch}, Acc: {accuracy:.2f})"
            self.model_status.config(text=status_text, foreground="green")
            self.status_bar.config(text=f"Best model loaded successfully on {self.device}")
            
        except Exception as e:
            self.progress.stop()
            self.model_status.config(text="Model: Error loading", foreground="red")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def find_best_model(self):
        """Find the best model checkpoint based on accuracy and IoU"""
        checkpoints_dir = os.path.dirname(Config.MODEL_SAVE_PATH)
        
        if not os.path.exists(checkpoints_dir):
            return None
        
        best_path = None
        best_score = -1
        best_info = {}
        
        # Check for specific best model file first
        if os.path.exists(Config.MODEL_SAVE_PATH):
            try:
                checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location='cpu', weights_only=False)
                accuracy = checkpoint.get('metrics', {}).get('accuracy', 0)
                iou = checkpoint.get('metrics', {}).get('mean_iou', 0)
                epoch = checkpoint.get('epoch', 0)
                
                # Combined score: give equal weight to accuracy and IoU
                score = (accuracy + iou) / 2
                
                print(f"Found best_model.pth - Epoch: {epoch}, Accuracy: {accuracy:.4f}, IoU: {iou:.4f}, Score: {score:.4f}")
                
                # Check if this is actually a good model (not just epoch 0)
                if epoch > 0 and score > 0.1:  # Basic sanity check
                    best_path = Config.MODEL_SAVE_PATH
                    best_score = score
                    best_info = {'accuracy': accuracy, 'iou': iou, 'epoch': epoch, 'score': score}
            except Exception as e:
                print(f"Error loading best_model.pth: {e}")
        
        # Search for other checkpoint files to find potentially better models
        checkpoint_patterns = ['*.pth', '*.pt']
        
        for pattern in checkpoint_patterns:
            files = glob.glob(os.path.join(checkpoints_dir, pattern))
            
            for file_path in files:
                if file_path == Config.MODEL_SAVE_PATH:  # Skip the one we already checked
                    continue
                    
                try:
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    accuracy = checkpoint.get('metrics', {}).get('accuracy', 0)
                    iou = checkpoint.get('metrics', {}).get('mean_iou', 0)
                    epoch = checkpoint.get('epoch', 0)
                    
                    # Combined score: give equal weight to accuracy and IoU
                    score = (accuracy + iou) / 2
                    
                    print(f"Checking {os.path.basename(file_path)} - Epoch: {epoch}, Accuracy: {accuracy:.4f}, IoU: {iou:.4f}, Score: {score:.4f}")
                    
                    # Select best based on combined score
                    if score > best_score:
                        best_score = score
                        best_path = file_path
                        best_info = {'accuracy': accuracy, 'iou': iou, 'epoch': epoch, 'score': score}
                        
                except Exception as e:
                    print(f"Could not load {file_path}: {e}")
                    continue
        
        if best_path:
            print(f"Best model selected: {os.path.basename(best_path)}")
            print(f"  - Epoch: {best_info.get('epoch', 'Unknown')}")
            print(f"  - Accuracy: {best_info.get('accuracy', 0):.4f}")
            print(f"  - IoU: {best_info.get('iou', 0):.4f}")
            print(f"  - Combined Score: {best_info.get('score', 0):.4f}")
        else:
            print("No valid model found!")
        
        return best_path
    
    def select_model(self):
        """Manually select a model checkpoint"""
        filetypes = [
            ("PyTorch checkpoints", "*.pth *.pt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=filetypes,
            initialdir=os.path.dirname(Config.MODEL_SAVE_PATH)
        )
        
        if filename:
            try:
                self.status_bar.config(text="Loading selected model...")
                self.progress.start()
                
                checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
                self.model = create_model()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.progress.stop()
                
                # Get model info
                epoch = checkpoint.get('epoch', 'Unknown')
                accuracy = checkpoint.get('metrics', {}).get('accuracy', 0)
                model_name = checkpoint.get('config', {}).get('model_name', 'Unknown')
                
                status_text = f"Model: {model_name} (Epoch {epoch}, Acc: {accuracy:.2f})"
                self.model_status.config(text=status_text, foreground="blue")
                self.status_bar.config(text=f"Custom model loaded: {os.path.basename(filename)}")
                
            except Exception as e:
                self.progress.stop()
                self.model_status.config(text="Model: Error loading custom model", foreground="red")
                messagebox.showerror("Error", f"Failed to load selected model: {str(e)}")
                # Try to reload the default best model
                self.load_model()
    
    def list_available_models(self):
        """Show available model checkpoints with their performance"""
        checkpoints_dir = os.path.dirname(Config.MODEL_SAVE_PATH)
        
        if not os.path.exists(checkpoints_dir):
            messagebox.showinfo("No Models", "No checkpoints directory found.")
            return
        
        import glob
        checkpoint_files = []
        patterns = ['*.pth', '*.pt']
        
        for pattern in patterns:
            checkpoint_files.extend(glob.glob(os.path.join(checkpoints_dir, pattern)))
        
        if not checkpoint_files:
            messagebox.showinfo("No Models", "No model checkpoints found.")
            return
        
        # Create a window to show model information
        model_window = tk.Toplevel(self.root)
        model_window.title("Available Models")
        model_window.geometry("800x600")
        
        # Create treeview to show model details
        columns = ('File', 'Epoch', 'Accuracy', 'IoU', 'F1', 'Timestamp')
        tree = ttk.Treeview(model_window, columns=columns, show='headings', height=15)
        
        # Define headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Add data
        for file_path in sorted(checkpoint_files):
            try:
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                filename = os.path.basename(file_path)
                epoch = checkpoint.get('epoch', 'N/A')
                accuracy = checkpoint.get('metrics', {}).get('accuracy', 0)
                iou = checkpoint.get('metrics', {}).get('mean_iou', 0)
                f1 = checkpoint.get('metrics', {}).get('mean_f1', 0)
                timestamp = checkpoint.get('timestamp', 'Unknown')
                
                tree.insert('', 'end', values=(
                    filename,
                    epoch,
                    f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else "N/A",
                    f"{iou:.4f}" if isinstance(iou, (int, float)) else "N/A",
                    f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A",
                    timestamp
                ))
            except Exception as e:
                tree.insert('', 'end', values=(
                    os.path.basename(file_path),
                    'Error',
                    'Could not load',
                    str(e)[:20],
                    '',
                    ''
                ))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add load button
        def load_selected():
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                filename = item['values'][0]
                full_path = os.path.join(checkpoints_dir, filename)
                
                try:
                    self.status_bar.config(text="Loading selected model...")
                    self.progress.start()
                    
                    checkpoint = torch.load(full_path, map_location=self.device, weights_only=False)
                    self.model = create_model()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    
                    self.progress.stop()
                    
                    # Get model info
                    epoch = checkpoint.get('epoch', 'Unknown')
                    accuracy = checkpoint.get('metrics', {}).get('accuracy', 0)
                    model_name = checkpoint.get('config', {}).get('model_name', 'Unknown')
                    
                    status_text = f"Model: {model_name} (Epoch {epoch}, Acc: {accuracy:.2f})"
                    self.model_status.config(text=status_text, foreground="purple")
                    self.status_bar.config(text=f"Model loaded: {filename}")
                    
                    model_window.destroy()
                    
                except Exception as e:
                    self.progress.stop()
                    messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            else:
                messagebox.showwarning("Warning", "Please select a model first.")
        
        button_frame = ttk.Frame(model_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Load Selected Model", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=model_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def cleanup_models(self):
        """Simple cleanup of old milestone files"""
        try:
            checkpoints_dir = os.path.dirname(Config.MODEL_SAVE_PATH)
            
            if not os.path.exists(checkpoints_dir):
                messagebox.showinfo("No Models", "No checkpoints directory found.")
                return
            
            # Find milestone files
            milestone_files = glob.glob(os.path.join(checkpoints_dir, '*_milestone_*.pth'))
            
            if len(milestone_files) <= 2:
                messagebox.showinfo("Cleanup", "No old milestone files to clean up.")
                return
            
            # Calculate total size of old milestones
            total_size = sum(os.path.getsize(f) for f in milestone_files[:-2])
            
            response = messagebox.askyesno(
                "Cleanup Old Milestones",
                f"Found {len(milestone_files)-2} old milestone files to clean.\n\n"
                f"Space to reclaim: {total_size / (1024*1024):.1f} MB\n\n"
                "Keep only the 2 most recent milestones?"
            )
            
            if response:
                # Sort by creation time and delete old ones
                milestone_files.sort(key=os.path.getctime)
                files_to_delete = milestone_files[:-2]
                
                deleted_count = 0
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except:
                        pass
                
                messagebox.showinfo(
                    "Cleanup Complete",
                    f"Successfully deleted {deleted_count} old milestone files.\n"
                    f"Kept the 2 most recent milestones."
                )
                self.status_bar.config(text=f"Cleanup complete: deleted {deleted_count} files")
            else:
                self.status_bar.config(text="Cleanup cancelled")
                
        except Exception as e:
            messagebox.showerror("Error", f"Cleanup failed: {str(e)}")
            self.status_bar.config(text="Cleanup failed")
    
    def select_image(self):
        """Select input image"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=filetypes
        )
        
        if filename:
            self.current_image_path = filename
            
            # Try to automatically find corresponding mask
            self.auto_find_mask(filename)
            
            self.status_bar.config(text=f"Image selected: {os.path.basename(filename)}")
    
    def auto_find_mask(self, image_path):
        """Automatically try to find corresponding mask file"""
        # Get image directory and filename without extension
        image_dir = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"Searching for mask for image: {image_name}")
        print(f"Image directory: {image_dir}")
        
        # Common mask patterns to search for
        mask_patterns = [
            # Dataset structure patterns (MOST SPECIFIC FIRST)
            os.path.join(image_dir.replace("images_png", "masks_png"), f"{image_name}.png"),
            
            # Check if we're in a dataset structure
            *self._check_dataset_structure(image_path, image_name),
            
            # Parent/sibling directory patterns
            os.path.join(os.path.dirname(image_dir), "masks_png", f"{image_name}.png"),
            os.path.join(os.path.dirname(image_dir), "masks", f"{image_name}.png"),
            
            # Same directory patterns (LEAST SPECIFIC LAST)
            os.path.join(image_dir, f"{image_name}_mask.png"),
        ]
        
        # Debug: print all patterns being checked
        print("Checking mask patterns:")
        for i, pattern in enumerate(mask_patterns):
            exists = os.path.exists(pattern)
            print(f"  {i+1}. {pattern} - {'EXISTS' if exists else 'NOT FOUND'}")
        
        # Try to find mask file
        for mask_path in mask_patterns:
            if os.path.exists(mask_path):
                # Test if mask can be loaded and has valid data
                test_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if test_mask is not None:
                    unique_vals = np.unique(test_mask)
                    print(f"Testing mask: {mask_path}")
                    print(f"Mask unique values: {unique_vals}")
                    
                    # Check if this looks like a valid segmentation mask
                    # Valid masks should have values 0-7 or be easily mappable
                    if self._is_valid_mask(unique_vals, mask_path):
                        self.current_mask_path = mask_path
                        self.status_bar.config(text=f"Auto-found mask: {os.path.basename(mask_path)} (values: {len(unique_vals)} classes)")
                        return
                    else:
                        print(f"Mask rejected - invalid values or format")
                else:
                    print(f"Mask file exists but cannot be loaded: {mask_path}")
        
        # If no mask found, clear current mask
        print("No valid mask found")
        self.current_mask_path = None
        self.status_bar.config(text="No mask found automatically")
    
    def _is_valid_mask(self, unique_vals, mask_path):
        """Check if mask has valid segmentation values"""
        # Skip if this is in images_png directory (likely another image)
        if "images_png" in mask_path:
            return False
        
        # Check if values are in expected range (0-7) or can be mapped
        min_val, max_val = unique_vals.min(), unique_vals.max()
        
        # Perfect case: values are 0-7
        if min_val >= 0 and max_val <= 7:
            return True
        
        # Acceptable case: values can be mapped (not too many unique values)
        if len(unique_vals) <= 20 and max_val < 300:
            return True
            
        # Reject if too many unique values (likely an image, not a mask)
        if len(unique_vals) > 50:
            return False
            
        return False
    
    def _check_dataset_structure(self, image_path, image_name):
        """Check for dataset structure patterns"""
        potential_masks = []
        
        # Check if we're in a dataset structure like: dataset/Train|Val|Test/Urban|Rural/images_png/
        path_parts = image_path.replace('\\', '/').split('/')
        
        for i, part in enumerate(path_parts):
            if part in ['Train', 'Val', 'Test']:
                # Found dataset split folder
                base_path = '/'.join(path_parts[:i+1])
                
                # Check for region folders
                for j in range(i+1, len(path_parts)):
                    if path_parts[j] in ['Urban', 'Rural']:
                        region_path = '/'.join(path_parts[:j+1])
                        mask_path = os.path.join(region_path, 'masks_png', f"{image_name}.png")
                        potential_masks.append(mask_path.replace('/', os.path.sep))
                        break
                break
        
        return potential_masks
    
    def select_mask(self):
        """Select ground truth mask (optional)"""
        filetypes = [
            ("PNG files", "*.png"),
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Ground Truth Mask (Optional)",
            filetypes=filetypes
        )
        
        if filename:
            self.current_mask_path = filename
            self.status_bar.config(text=f"Mask selected: {os.path.basename(filename)}")
    
    def run_prediction(self):
        """Run prediction on selected image"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        if self.current_image_path is None:
            messagebox.showerror("Error", "Please select an image first!")
            return
        
        try:
            self.status_bar.config(text="Running prediction...")
            self.progress.start()
            
            # Load and preprocess image
            image = cv2.imread(self.current_image_path)
            if image is None:
                raise ValueError(f"Could not load image: {self.current_image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # Resize to model input size
            image_resized = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
            
            # Normalize
            image_norm = image_resized.astype(np.float32) / 255.0
            image_norm = (image_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to tensor and predict
            image_tensor = torch.FloatTensor(image_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Resize prediction back to original size
            prediction_resized = cv2.resize(
                prediction.astype(np.uint8), 
                (original_size[1], original_size[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Load ground truth mask if available
            gt_mask = None
            if self.current_mask_path and os.path.exists(self.current_mask_path):
                print(f"Loading mask from: {self.current_mask_path}")
                gt_mask = cv2.imread(self.current_mask_path, cv2.IMREAD_GRAYSCALE)
                
                if gt_mask is not None:
                    print(f"Mask loaded successfully. Shape: {gt_mask.shape}")
                    original_unique = np.unique(gt_mask)
                    print(f"Original mask unique values: {original_unique}")
                    print(f"Original mask min: {gt_mask.min()}, max: {gt_mask.max()}")
                    
                    # Map mask values to 0-7 range if needed
                    gt_mask = self._normalize_mask_values(gt_mask, original_unique)
                    
                    if gt_mask.shape != original_size:
                        gt_mask = cv2.resize(gt_mask, (original_size[1], original_size[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                        print(f"Mask resized to: {gt_mask.shape}")
                    
                    final_unique = np.unique(gt_mask)
                    print(f"Final mask unique values: {final_unique}")
                else:
                    print("ERROR: Could not load mask file!")
                    self.status_bar.config(text="Warning: Could not load mask file")
                    self.current_mask_path = None
            
            # Visualize results
            self.visualize_results(image, prediction_resized, gt_mask)
            
            self.progress.stop()
            self.status_bar.config(text="Prediction completed successfully!")
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_bar.config(text="Prediction failed!")
    
    def _normalize_mask_values(self, mask, unique_values):
        """Normalize mask values to 0-7 range"""
        # If already in 0-7 range, return as is
        if unique_values.min() >= 0 and unique_values.max() <= 7:
            print("Mask already in correct range (0-7)")
            return mask
        
        # Create a mapping from original values to 0-7
        normalized_mask = np.zeros_like(mask)
        
        # Sort unique values and map them to 0-7
        sorted_unique = np.sort(unique_values)
        
        print(f"Mapping mask values:")
        for i, original_val in enumerate(sorted_unique):
            if i < 8:  # Only map first 8 unique values to classes 0-7
                normalized_mask[mask == original_val] = i
                print(f"  {original_val} -> {i}")
            else:
                # Map excess values to background (class 1)
                normalized_mask[mask == original_val] = 1
                print(f"  {original_val} -> 1 (background)")
        
        return normalized_mask
    
    def visualize_results(self, image, prediction, gt_mask=None):
        """Visualize prediction results"""
        self.fig.clear()
        
        # Determine subplot layout
        if gt_mask is not None:
            # 2x3 layout if ground truth is available
            rows, cols = 2, 3
        else:
            # 2x2 layout if no ground truth
            rows, cols = 2, 2
        
        class_names = ["No data", "Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
        
        # 1. Original image
        ax1 = self.fig.add_subplot(rows, cols, 1)
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. Prediction
        ax2 = self.fig.add_subplot(rows, cols, 2)
        colored_pred = get_colored_mask(prediction)
        ax2.imshow(colored_pred)
        ax2.set_title('Predicted Segmentation', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        if gt_mask is not None:
            # Debug: Check ground truth mask
            print(f"Ground truth mask info:")
            print(f"  Shape: {gt_mask.shape}")
            print(f"  Unique values: {np.unique(gt_mask)}")
            print(f"  Min: {gt_mask.min()}, Max: {gt_mask.max()}")
            
            # 3. Ground truth
            ax3 = self.fig.add_subplot(rows, cols, 3)
            colored_gt = get_colored_mask(gt_mask)
            ax3.imshow(colored_gt)
            ax3.set_title('Ground Truth', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # 4. Overlay with prediction
            ax4 = self.fig.add_subplot(rows, cols, 4)
            overlay_pred = image.astype(np.float32) * 0.6 + colored_pred.astype(np.float32) * 0.4
            overlay_pred = np.clip(overlay_pred, 0, 255).astype(np.uint8)
            ax4.imshow(overlay_pred)
            ax4.set_title('Prediction Overlay', fontsize=12, fontweight='bold')
            ax4.axis('off')
            
            # 5. Overlay with ground truth
            ax5 = self.fig.add_subplot(rows, cols, 5)
            overlay_gt = image.astype(np.float32) * 0.6 + colored_gt.astype(np.float32) * 0.4
            overlay_gt = np.clip(overlay_gt, 0, 255).astype(np.uint8)
            ax5.imshow(overlay_gt)
            ax5.set_title('Ground Truth Overlay', fontsize=12, fontweight='bold')
            ax5.axis('off')
            
            # 6. Difference map
            ax6 = self.fig.add_subplot(rows, cols, 6)
            diff = (prediction != gt_mask).astype(np.uint8) * 255
            ax6.imshow(diff, cmap='Reds')
            ax6.set_title('Prediction Errors (Red)', fontsize=12, fontweight='bold')
            ax6.axis('off')
            
            # Calculate accuracy
            accuracy = (prediction == gt_mask).mean() * 100
            self.fig.suptitle(f'Segmentation Results (Accuracy: {accuracy:.2f}%)', 
                            fontsize=14, fontweight='bold')
        else:
            # 3. Overlay (without ground truth)
            ax3 = self.fig.add_subplot(rows, cols, 3)
            overlay = image.astype(np.float32) * 0.6 + colored_pred.astype(np.float32) * 0.4
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            ax3.imshow(overlay)
            ax3.set_title('Prediction Overlay', fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # 4. Class distribution
            ax4 = self.fig.add_subplot(rows, cols, 4)
            unique_classes, counts = np.unique(prediction, return_counts=True)
            total_pixels = prediction.size
            percentages = (counts / total_pixels) * 100
            
            # Use consistent colors matching the training colors
            plot_colors = ['black', '#A9A9A9', 'red', 'yellow', 'blue', '#8B4513', '#228B22', 'cyan']
            selected_colors = [plot_colors[i] for i in unique_classes]
            selected_names = [class_names[i] for i in unique_classes]
            
            ax4.pie(percentages, labels=selected_names, colors=selected_colors, autopct='%1.1f%%')
            ax4.set_title('Class Distribution', fontsize=12, fontweight='bold')
            
            self.fig.suptitle('Segmentation Results', fontsize=14, fontweight='bold')
        
        # Add legend
        colors = {
            0: [0, 0, 0],           # No data - Black
            1: [169, 169, 169],     # Background - Gray
            2: [255, 0, 0],         # Building - Red
            3: [255, 255, 0],       # Road - Yellow
            4: [0, 0, 255],         # Water - Blue
            5: [139, 69, 19],       # Barren - Brown
            6: [34, 139, 34],       # Forest - Green
            7: [0, 255, 255],       # Agriculture - Cyan
        }
        
        legend_elements = []
        unique_classes = np.unique(prediction)
        for class_id in unique_classes:
            color = np.array(colors[class_id]) / 255.0
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=class_names[class_id]))
        
        self.fig.legend(handles=legend_elements, loc='lower center', 
                       bbox_to_anchor=(0.5, 0.02), ncol=len(unique_classes))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        self.canvas.draw()
    
    def save_results(self):
        """Save the current results"""
        if self.fig.get_axes():
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                title="Save Results"
            )
            
            if filename:
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"Results saved to: {filename}")
                self.status_bar.config(text=f"Results saved: {os.path.basename(filename)}")
        else:
            messagebox.showwarning("Warning", "No results to save!")

def main():
    root = tk.Tk()
    app = SegmentationTestApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
