from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def impute_missing_data(players_df):
    
    print("Missing Data Imputation - Players\n")
    print("Using ML-based imputation for height and weight\n")

    # Initial statistics
    print("Initial Missing Data Summary:")
    print(f"  Zero heights: {(players_df['height'] == 0).sum()}")
    print(f"  Zero weights: {(players_df['weight'] == 0).sum()}\n")

    print("\nHeight statistics before imputation:")
    print(players_df['height'].describe())

    print("\nWeight statistics before imputation:")
    print(players_df['weight'].describe())

    # ==========================================
    # STEP 1: Height Imputation
    # ==========================================
    print("=" * 60)
    print("STEP 1: Height Imputation (Random Forest Regression)")
    print("=" * 60)

    # Prepare data: players with valid height to train on
    players_with_height = players_df[(players_df['height'] > 0) & (players_df['pos'].notna())].copy()
    players_missing_height = players_df[(players_df['height'] == 0) & (players_df['pos'].notna())].copy()

    if len(players_missing_height) > 0:
        print(f"\nPlayers with missing height: {len(players_missing_height)}")
        
        # One-hot encode position
        X_height = pd.get_dummies(players_with_height[['pos', 'weight']], columns=['pos'])
        y_height = players_with_height['height']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_height, y_height, test_size=0.2, random_state=42
        )
        
        # Train Random Forest Regressor
        rf_regressor_height = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        rf_regressor_height.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred_test = rf_regressor_height.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print("\nModel Performance on Test Set:")
        print(f"  Mean Absolute Error: {mae:.2f} inches")
        print(f"  R² Score: {r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_regressor_height, X_height, y_height, 
                                    cv=5, scoring='neg_mean_absolute_error')
        print(f"  5-Fold CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Retrain on full data
        rf_regressor_height.fit(X_height, y_height)
        
        # Predict missing heights
        X_missing_height = pd.get_dummies(
            players_missing_height[['pos', 'weight']], 
            columns=['pos']
        )
        # Align columns with training data
        X_missing_height = X_missing_height.reindex(columns=X_height.columns, fill_value=0)
        
        predicted_heights = rf_regressor_height.predict(X_missing_height)
        players_df.loc[players_missing_height.index, 'height'] = predicted_heights.round(1)
        
        print(f"\nPredicted heights for {len(players_missing_height)} players")
        print(f"  Predicted height range: {predicted_heights.min():.1f} - {predicted_heights.max():.1f} inches")
    else:
        print("\nNo missing heights to impute")

    # ==========================================
    # STEP 2: Weight Imputation
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 2: Weight Imputation (Random Forest Regression)")
    print("=" * 60)

    # Prepare data: players with valid weight to train on
    players_with_weight = players_df[(players_df['weight'] > 0) & (players_df['pos'].notna())].copy()
    players_missing_weight = players_df[(players_df['weight'] == 0) & (players_df['pos'].notna())].copy()

    if len(players_missing_weight) > 0:
        print(f"\nPlayers with missing weight: {len(players_missing_weight)}")
        
        # One-hot encode position
        X_weight = pd.get_dummies(players_with_weight[['pos', 'height']], columns=['pos'])
        y_weight = players_with_weight['weight']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_weight, y_weight, test_size=0.2, random_state=42
        )
        
        # Train Random Forest Regressor
        rf_regressor_weight = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        rf_regressor_weight.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred_test = rf_regressor_weight.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print("\nModel Performance on Test Set:")
        print(f"  Mean Absolute Error: {mae:.2f} lbs")
        print(f"  R² Score: {r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_regressor_weight, X_weight, y_weight, 
                                    cv=5, scoring='neg_mean_absolute_error')
        print(f"  5-Fold CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Retrain on full data
        rf_regressor_weight.fit(X_weight, y_weight)
        
        # Predict missing weights
        X_missing_weight = pd.get_dummies(
            players_missing_weight[['pos', 'height']], 
            columns=['pos']
        )
        # Align columns with training data
        X_missing_weight = X_missing_weight.reindex(columns=X_weight.columns, fill_value=0)
        
        predicted_weights = rf_regressor_weight.predict(X_missing_weight)
        players_df.loc[players_missing_weight.index, 'weight'] = predicted_weights.round(1)
        
        print(f"\nPredicted weights for {len(players_missing_weight)} players")
        print(f"  Predicted weight range: {predicted_weights.min():.1f} - {predicted_weights.max():.1f} lbs")
    else:
        print("\nNo missing weights to impute")

    # ==========================================
    # FINAL VERIFICATION
    # ==========================================
    print("\n" + "=" * 60)
    print("Final Verification")
    print("=" * 60)

    print("\nAfter Imputation:")
    print(f"  Zero heights: {(players_df['height'] == 0).sum()}")
    print(f"  Zero weights: {(players_df['weight'] == 0).sum()}")

    print("\nHeight statistics after imputation:")
    print(players_df[players_df['height'] > 0]['height'].describe())

    print("\nWeight statistics after imputation:")
    print(players_df[players_df['weight'] > 0]['weight'].describe())
    
    return players_df
