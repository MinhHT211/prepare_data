import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
DATA_DIR = "data"        # Input folder
RESULTS_DIR = "results"  # Output folder
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    # 1. LOAD DATA
    print("--- 1. Loading data from 'data/' folder ---")
    try:
        X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
        y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        print(f"   Original Dataset Shape: {X.shape}") 
    except FileNotFoundError:
        print("❌ ERROR: .npy files not found. Please check 'data/' folder!")
        return

    # 2. STANDARDIZATION (StandardScaler)
    # Bước này đưa dữ liệu về Mean=0, Std=1
    print("--- 2. Standardizing data (StandardScaler) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. RUN PCA (Analysis)

    print("--- 3. Running PCA Analysis ---")
    pca = PCA()
    pca.fit(X_scaled)
    
    # Tính tổng phương sai tích lũy
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    # 4. PLOT EXPLAINED VARIANCE
    # Vẽ biểu đồ cho báo cáo
    print("--- 4. Plotting Explained Variance Chart ---")
    plt.figure(figsize=(10, 6))
    plt.plot(cumsum, linewidth=2, color='#007acc')
    plt.xlabel('Number of Components (Dimensions)', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title('PCA - Explained Variance Ratio', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Vẽ đường mốc 95%
    d_95 = np.argmax(cumsum >= 0.95) + 1
    plt.axhline(y=0.95, color='r', linestyle='--', label=f'95% Information ({d_95} dims)')
    plt.axvline(x=d_95, color='r', linestyle='--')
    plt.legend(loc='best')

    # Lưu biểu đồ
    chart_path = os.path.join(RESULTS_DIR, 'pca_variance_plot.png')
    plt.savefig(chart_path, dpi=300) # dpi=300 để ảnh sắc nét
    print(f"   ✅ Chart saved to: {chart_path}")
    # plt.show() 

    # 5. EXPORT REDUCED DATA
    print("--- 5. Exporting dimensionality-reduced datasets ---")
    target_dims = [64, 128, 256] 
    
    for dim in target_dims:
        print(f"   -> Reducing to {dim} dimensions...")
        pca_k = PCA(n_components=dim)
        X_pca = pca_k.fit_transform(X_scaled)
        
        # Lưu file
        save_path = os.path.join(RESULTS_DIR, f'X_pca_{dim}.npy')
        np.save(save_path, X_pca)
        print(f"      Saved: {save_path}")

    # Copy nhãn y sang folder results
    np.save(os.path.join(RESULTS_DIR, 'y_labels.npy'), y)
    print("\n🎉 PCA TASK COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()