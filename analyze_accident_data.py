import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm

# Path configurations
OPERATION_DATA_DIR = "Operation_csv_data"
DOSE_DATA_DIR = "Dose_csv_data"
TRANSIENT_DATA_DIR = "NPPAD"
OUTPUT_DIR = "analysis_output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_operation_data(accident_type, file_id):
    """Load operation data for a given accident type and file ID."""
    file_path = os.path.join(OPERATION_DATA_DIR, accident_type, f"{file_id}.csv")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_dose_data(accident_type, file_id):
    """Load dose data for a given accident type and file ID."""
    file_path = os.path.join(DOSE_DATA_DIR, accident_type, f"{file_id}dose.csv")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_transient_report(accident_type, file_id):
    """Load transient report for a given accident type and file ID."""
    file_path = os.path.join(TRANSIENT_DATA_DIR, accident_type, f"{file_id}Transient Report.txt")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def visualize_time_series(accident_type, file_id, parameters=None):
    """Visualize time series data for specific parameters."""
    df = load_operation_data(accident_type, file_id)
    
    if df is None:
        return
    
    # Use all parameters except TIME
    if parameters is None:
        parameters = [col for col in df.columns if col != 'TIME']
    
    # Create plots
    fig, axes = plt.subplots(len(parameters), 1, figsize=(12, 4*len(parameters)))
    
    if len(parameters) == 1:
        axes = [axes]
    
    for i, param in enumerate(parameters):
        if param in df.columns:
            axes[i].plot(df['TIME'], df[param])
            axes[i].set_title(f"{param} vs Time - {accident_type} (File {file_id})")
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel(param)
            axes[i].grid(True)
        else:
            print(f"Parameter {param} not found in the data.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{accident_type}_{file_id}_time_series.png"))
    plt.close()

def compare_accident_types(parameters=None):
    """Compare the same parameter across different accident types."""
    
    # Get all accident types
    accident_types = [d for d in os.listdir(OPERATION_DATA_DIR) 
                     if os.path.isdir(os.path.join(OPERATION_DATA_DIR, d)) and d != 'Normal']
    
    # Get a sample file to determine available parameters
    sample_file = None
    for accident_type in accident_types:
        file_pattern = os.path.join(OPERATION_DATA_DIR, accident_type, "*.csv")
        files = glob.glob(file_pattern)
        if files:
            sample_file = load_operation_data(accident_type, os.path.basename(files[0]).split('.')[0])
            if sample_file is not None:
                break
    
    # If no parameters specified and sample file is available, get all parameters
    if parameters is None and sample_file is not None:
        parameters = [col for col in sample_file.columns if col != 'TIME']
    elif parameters is None:
        # Default parameters if no sample file is available
        parameters = ['P', 'TAVG', 'VOID', 'PRB', 'QMWT', 'WHPI', 'WBK']
    
    # For each parameter, plot its behavior across different accident types
    for param_idx, param in enumerate(parameters):
        print(f"Comparing parameter {param_idx+1}/{len(parameters)}: {param}")
        
        plt.figure(figsize=(14, 8))
        
        for accident_type in tqdm(accident_types, desc=f"Processing accident types for {param}"):
            # Get all files for this accident type
            file_pattern = os.path.join(OPERATION_DATA_DIR, accident_type, "*.csv")
            files = glob.glob(file_pattern)
            
            # Limit to 3 files per accident type to avoid overcrowding the plot
            max_files = min(3, len(files))
            for file_path in files[:max_files]:
                file_id = os.path.basename(file_path).split('.')[0]
                df = load_operation_data(accident_type, file_id)
                
                if df is not None and param in df.columns:
                    plt.plot(df['TIME'], df[param], alpha=0.7, label=f"{accident_type}_{file_id}")
        
        plt.title(f"Comparison of {param} across different accident types")
        plt.xlabel("Time (s)")
        plt.ylabel(param)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_{param}.png"))
        plt.close()

def visualize_data_distribution():
    """Visualize the distribution of data across accident types."""
    
    # Get all accident types
    accident_types = [d for d in os.listdir(OPERATION_DATA_DIR) 
                     if os.path.isdir(os.path.join(OPERATION_DATA_DIR, d)) and d != 'Normal']
    
    # Count files per accident type
    counts = []
    for accident_type in accident_types:
        file_pattern = os.path.join(OPERATION_DATA_DIR, accident_type, "*.csv")
        files = glob.glob(file_pattern)
        counts.append(len(files))
    
    # Plot distribution
    plt.figure(figsize=(14, 8))
    bars = plt.bar(accident_types, counts)
    plt.title("Number of Sample Files per Accident Type")
    plt.xlabel("Accident Type")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "data_distribution.png"))
    plt.close()

def extract_features_for_visualization(accident_types=None):
    """Extract features for visualization using dimensionality reduction."""
    
    if accident_types is None:
        # Get all accident types
        accident_types = [d for d in os.listdir(OPERATION_DATA_DIR) 
                         if os.path.isdir(os.path.join(OPERATION_DATA_DIR, d)) and d != 'Normal']
    
    # Get a sample file to determine available parameters
    sample_file = None
    for accident_type in accident_types:
        file_pattern = os.path.join(OPERATION_DATA_DIR, accident_type, "*.csv")
        files = glob.glob(file_pattern)
        if files:
            sample_file = load_operation_data(accident_type, os.path.basename(files[0]).split('.')[0])
            if sample_file is not None:
                break
    
    # Determine parameters to use for feature extraction
    if sample_file is not None:
        key_parameters = [col for col in sample_file.columns if col != 'TIME']
    else:
        # Default parameters if no sample file is available
        key_parameters = ['P', 'TAVG', 'VOID', 'PRB', 'QMWT', 'WHPI', 'WBK']
    
    features = []
    labels = []
    file_ids = []
    
    for accident_type in tqdm(accident_types, desc="Extracting features"):
        # Get all files for this accident type
        file_pattern = os.path.join(OPERATION_DATA_DIR, accident_type, "*.csv")
        files = glob.glob(file_pattern)
        
        # Process all files for each accident type
        for file_path in files:
            file_id = os.path.basename(file_path).split('.')[0]
            df = load_operation_data(accident_type, file_id)
            
            if df is None:
                continue
            
            # Extract simple features (mean values of key parameters)
            file_features = []
            for param in key_parameters:
                if param in df.columns:
                    file_features.append(df[param].mean())
                else:
                    file_features.append(0)
            
            features.append(file_features)
            labels.append(accident_type)
            file_ids.append(f"{accident_type}_{file_id}")
    
    # Convert to numpy arrays
    features_array = np.array(features)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)
    
    # Apply t-SNE for dimensionality reduction
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_2d = tsne.fit_transform(features_scaled)
    
    # Create DataFrame for plotting
    viz_df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'accident_type': labels,
        'file_id': file_ids
    })
    
    # Plot
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=viz_df, x='x', y='y', hue='accident_type', style='accident_type', s=100)
    plt.title("t-SNE Visualization of Accident Types Based on All Parameters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_visualization.png"))
    
    # Save interactive HTML version using plotly if available
    try:
        import plotly.express as px
        import plotly.io as pio
        
        fig = px.scatter(viz_df, x='x', y='y', color='accident_type', hover_data=['file_id'])
        fig.update_layout(
            title="Interactive t-SNE Visualization of Accident Types",
            width=1000,
            height=800
        )
        pio.write_html(fig, os.path.join(OUTPUT_DIR, "tsne_visualization_interactive.html"))
        print("Interactive t-SNE visualization saved.")
    except ImportError:
        print("Plotly not available. Skipping interactive visualization.")
    
    plt.close()
    
    return viz_df

def analyze_transient_reports(accident_types=None):
    """Analyze transient reports to understand event sequences."""
    
    if accident_types is None:
        # Get all accident types
        accident_types = [d for d in os.listdir(TRANSIENT_DATA_DIR) 
                         if os.path.isdir(os.path.join(TRANSIENT_DATA_DIR, d))]
    
    report_summary = {}
    
    for accident_type in accident_types:
        report_summary[accident_type] = []
        
        # Get all files for this accident type
        file_pattern = os.path.join(OPERATION_DATA_DIR, accident_type, "*.csv")
        files = glob.glob(file_pattern)
        
        for file_path in tqdm(files, desc=f"Analyzing reports for {accident_type}"):
            file_id = os.path.basename(file_path).split('.')[0]
            report = load_transient_report(accident_type, file_id)
            
            if report:
                # Add to summary
                report_summary[accident_type].append({
                    'file_id': file_id,
                    'report': report
                })
    
    # Save summary to file
    with open(os.path.join(OUTPUT_DIR, "transient_report_summary.txt"), 'w') as f:
        for accident_type, reports in report_summary.items():
            f.write(f"=== {accident_type} ===\n\n")
            f.write(f"Number of reports: {len(reports)}\n\n")
            
            for report_data in reports[:5]:  # Show details for first 5 reports only to keep file manageable
                f.write(f"File ID: {report_data['file_id']}\n")
                f.write(f"Report:\n{report_data['report']}\n\n")
                f.write("-" * 50 + "\n\n")
            
            if len(reports) > 5:
                f.write(f"... {len(reports) - 5} more reports ...\n\n")
    
    print(f"Saved transient report summary for {sum(len(reports) for reports in report_summary.values())} files.")

def main():
    """Main function to run data analysis."""
    print("Starting comprehensive data analysis...")
    
    print("Creating output directory:", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Visualizing data distribution...")
    visualize_data_distribution()
    
    print("Visualizing time series for sample files...")
    # Visualize time series for a few accident types - we'll still limit these
    # to avoid generating hundreds of large plots
    accident_samples = [
        ('LOCA', '1'),
        ('LOCA', '50'),
        ('SGBTR', '1'),
        ('SLBIC', '1'),
        ('ATWS', '1'),  # Added since it's a rare class
        ('SP', '1')     # Added since it's a rare class
    ]
    
    for accident_type, file_id in accident_samples:
        print(f"Visualizing {accident_type} - File {file_id}")
        visualize_time_series(accident_type, file_id)
    
    print("Comparing parameters across accident types...")
    # We'll use automatic parameter detection rather than specifying them
    compare_accident_types()
    
    print("Generating t-SNE visualization using all files and parameters...")
    viz_df = extract_features_for_visualization()
    
    print("Analyzing all available transient reports...")
    analyze_transient_reports()
    
    print("Analysis complete. Results saved to:", OUTPUT_DIR)
    print("Note: This comprehensive analysis processed all available data and parameters.")

if __name__ == "__main__":
    main() 