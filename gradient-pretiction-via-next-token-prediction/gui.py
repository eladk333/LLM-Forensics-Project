import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons, Button
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

# --- הגדרות נתיבים ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(CURRENT_DIR, "forensic_merged_features.csv")

# --- רשימת הפיצ'רים המקצועית ---
FEATURE_CONFIG = {
    'Embedding_Norm': 'Embedding Norm (L2)',
    'Logit_Norm': 'Logit Norm',
    'Weight_Variance': 'Weight Variance',
    'L1_Norm': 'L1 Norm',
    'Next_Token_Prob': 'Next-Token Prob (Dynamic)'
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Dataset not found at {INPUT_CSV}")
        return
    
    df_all = pd.read_csv(INPUT_CSV)
    
    # הגדרת החלון והגרף
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.3, bottom=0.2) # משאירים מקום לכפתורים בצד

    # משתנים גלובליים למצב ה-GUI
    current_state = {
        'size': '124M',
        'model_type': 'Linear',
        'active_features': [FEATURES_LIST[0]] # מתחילים עם הנורמה
    }

    def update_plot(event=None):
        ax.clear()
        size = current_state['size']
        m_type = current_state['model_type']
        features = current_state['active_features']
        
        if not features:
            ax.set_title("Please select at least one feature")
            plt.draw()
            return

        subset = df_all[df_all['Model_Size'] == size].sort_values('Step')
        X = subset[features]
        y = subset['Step']

        # אימון המודל הנבחר
        if m_type == 'Linear':
            model = make_pipeline(StandardScaler(), LinearRegression())
        else:
            model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50,50), max_iter=2000, random_state=42))
        
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # ציור
        ax.scatter(y, y_pred, color='blue', alpha=0.6, label='Predicted')
        ax.plot(y, y, 'r--', label='Ground Truth (Ideal)')
        
        ax.set_title(f"Forensic Prediction: {size} Model\nFeatures: {len(features)} | Model: {m_type} | R² = {r2:.4f}", fontsize=14)
        ax.set_xlabel("Actual Gradient Updates")
        ax.set_ylabel("Predicted Updates")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.draw()

    # --- יצירת ה-GUI (כמו שהחברים עשו) ---

    # 1. בחירת גודל מודל
    ax_size = plt.axes([0.05, 0.7, 0.15, 0.15], facecolor='#f0f0f0')
    radio_size = RadioButtons(ax_size, ('7M', '30M', '124M'), active=2)
    def change_size(label):
        current_state['size'] = label
        update_plot()
    radio_size.on_clicked(change_size)

    # 2. בחירת סוג רגרסיה
    ax_model = plt.axes([0.05, 0.5, 0.15, 0.15], facecolor='#f0f0f0')
    radio_model = RadioButtons(ax_model, ('Linear', 'MLP'))
    def change_model(label):
        current_state['model_type'] = label
        update_plot()
    radio_model.on_clicked(change_model)

    # 3. בחירת פיצ'רים (Checkboxes)
    ax_check = plt.axes([0.05, 0.1, 0.2, 0.3], facecolor='#f0f0f0')
    labels = [FEATURE_CONFIG[f] for f in FEATURES_LIST]
    check = CheckButtons(ax_check, labels, [True] + [False]*(len(FEATURES_LIST)-1))
    
    def change_features(label):
        # הפיכת הלייבל חזרה לשם העמודה
        reverse_map = {v: k for k, v in FEATURE_CONFIG.items()}
        feat_name = reverse_map[label]
        if feat_name in current_state['active_features']:
            current_state['active_features'].remove(feat_name)
        else:
            current_state['active_features'].append(feat_name)
        update_plot()
    check.on_clicked(change_features)

    update_plot()
    print("🚀 GUI Started. Switch between features to see the prediction in real-time.")
    plt.show()

if __name__ == "__main__":
    main()