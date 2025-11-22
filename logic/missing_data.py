from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def impute_missing_data(players_df):
    
    print("=== ADVANCED MISSING DATA IMPUTATION ===\n")
    print("Using ML-based imputation with proper experimental setup to prevent data leakage\n")

    # Initial statistics
    print("Initial Missing Data Summary:")
    print(f"  Null positions: {players_df['pos'].isna().sum()}")
    print(f"  Zero heights: {(players_df['height'] == 0).sum()}")
    print(f"  Zero weights: {(players_df['weight'] == 0).sum()}")
    print(f"  Invalid birthdates: {(players_df['birthDate'] == '0000-00-00').sum()}\n")

    print("\nPosition distribution before imputation:")
    print(players_df['pos'].value_counts())

    print("\nHeight statistics before imputation:")
    print(players_df['height'].describe())

    print("\nWeight statistics before imputation:")
    print(players_df['weight'].describe())

    # ==========================================
    # STEP 1: Initial Height/Weight Imputation
    # ==========================================
    print("=" * 60)
    print("STEP 1: Initial Height/Weight Imputation (for known positions)")
    print("=" * 60)

    # For players with known positions but missing height/weight, use position-based means
    for pos in players_df[players_df['pos'].notna()]['pos'].unique():
        pos_data = players_df[(players_df['pos'] == pos) & (players_df['height'] > 0) & (players_df['weight'] > 0)]
        
        if len(pos_data) > 0:
            mean_height = pos_data['height'].mean()
            mean_weight = pos_data['weight'].mean()
            
            # Fill missing heights for this position
            height_mask = (players_df['pos'] == pos) & (players_df['height'] == 0)
            if height_mask.any():
                players_df.loc[height_mask, 'height'] = mean_height
                
            # Fill missing weights for this position
            weight_mask = (players_df['pos'] == pos) & (players_df['weight'] == 0)
            if weight_mask.any():
                players_df.loc[weight_mask, 'weight'] = mean_weight

    print(f"✓ Initial imputation complete for players with known positions")
    #print(f"  Remaining zero heights: {(players_df['height'] == 0).sum()}")
    #print(f"  Remaining zero weights: {(players_df['weight'] == 0).sum()}\n")

    # ==========================================
    # STEP 2: Position Imputation
    # ==========================================
    print("=" * 60)
    print("STEP 2: Position Imputation (Random Forest Classification)")
    print("=" * 60)

    # Now all players with positions should have height/weight
    players_with_pos = players_df[(players_df['pos'].notna()) & 
                                   (players_df['height'] > 0) & 
                                   (players_df['weight'] > 0)].copy()

    players_missing_pos = players_df[players_df['pos'].isna()].copy()

    if len(players_missing_pos) > 0:
        print(f"\nPlayers with missing positions: {len(players_missing_pos)}")
        
        # Features and target
        X_pos = players_with_pos[['height', 'weight']]
        y_pos = players_with_pos['pos']
        
        # Train-test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_pos, y_pos, test_size=0.2, random_state=42, stratify=y_pos
        )
        
        # Train Random Forest Classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=150, 
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced' 
        )
        rf_classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred_test = rf_classifier.predict(X_test)
        test_accuracy = (y_pred_test == y_test).mean()
        
        print(f"\nModel Performance on Test Set:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        #print("\nDetailed Classification Report:")
        #print(classification_report(y_test, y_pred_test, zero_division=0))
        
        # Cross-validation score
        cv_scores = cross_val_score(rf_classifier, X_pos, y_pos, cv=5)
        print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': ['height', 'weight'],
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nFeature Importance:")
        #display(feature_importance)
        
        # Retrain on full data and predict missing positions
        rf_classifier.fit(X_pos, y_pos)
        X_missing = players_missing_pos[['height', 'weight']]
        predicted_positions = rf_classifier.predict(X_missing)
        
        # Get prediction probabilities for confidence assessment
        pred_proba = rf_classifier.predict_proba(X_missing)
        max_proba = pred_proba.max(axis=1)
        
        print(f"\nPrediction Confidence Statistics:")
        print(f"  Mean confidence: {max_proba.mean():.4f}")
        print(f"  Min confidence: {max_proba.min():.4f}")
        print(f"  Max confidence: {max_proba.max():.4f}")
        
        # Apply predictions
        players_df.loc[players_missing_pos.index, 'pos'] = predicted_positions
        
        print(f"\n✓ Predicted positions for {len(players_missing_pos)} players using Random Forest")
    else:
        print("\nNo missing positions to impute")

    # ==========================================
    # STEP 3: Refined Height Imputation
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 3: Refined Height Imputation (Random Forest Regression)")
    print("=" * 60)

    # Prepare data: players with valid height to train on
    players_with_height = players_df[players_df['height'] > 0].copy()
    players_missing_height = players_df[players_df['height'] == 0].copy()

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
        
        print(f"\n✓ Predicted heights for {len(players_missing_height)} players")
        print(f"  Predicted height range: {predicted_heights.min():.1f} - {predicted_heights.max():.1f} inches")
    else:
        print("\nNo missing heights to impute")

    # ==========================================
    # STEP 4: Refined Weight Imputation
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 4: Refined Weight Imputation (Random Forest Regression)")
    print("=" * 60)

    # Prepare data: players with valid weight to train on
    players_with_weight = players_df[players_df['weight'] > 0].copy()
    players_missing_weight = players_df[players_df['weight'] == 0].copy()

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
        
        print(f"\n✓ Predicted weights for {len(players_missing_weight)} players")
        print(f"  Predicted weight range: {predicted_weights.min():.1f} - {predicted_weights.max():.1f} lbs")
    else:
        print("\nNo missing weights to impute")

    # ==========================================
    # STEP 5: Birthdate Handling
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 5: Birthdate Handling")
    print("=" * 60)

    # Convert invalid birthdates to NaT
    invalid_birthdates = (players_df['birthDate'] == '0000-00-00').sum()
    players_df['birthDate'] = players_df['birthDate'].replace('0000-00-00', pd.NaT)
    print(f"\n✓ Converted {invalid_birthdates} invalid birthdates to NaT")
    print("  Note: Birthdates left as NaT since they cannot be reliably predicted")
    #print("        (age information may be derived from career start year if needed)")

    # ==========================================
    # FINAL VERIFICATION
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    print("\nAfter Advanced Imputation:")
    print(f"  Missing positions: {players_df['pos'].isna().sum()}")
    print(f"  Zero heights: {(players_df['height'] == 0).sum()}")
    print(f"  Zero weights: {(players_df['weight'] == 0).sum()}")
    print(f"  Invalid birthdates: {players_df['birthDate'].isna().sum()}")

    print("\nPosition distribution after imputation:")
    print(players_df['pos'].value_counts())

    print("\nHeight statistics after imputation:")
    print(players_df['height'].describe())

    print("\nWeight statistics after imputation:")
    print(players_df['weight'].describe())

    # Sanity check: verify imputed values are reasonable
    print("\nSanity Checks:")
    print(f"  Min height: {players_df['height'].min():.1f} inches (expected: ~60-70)")
    print(f"  Max height: {players_df['height'].max():.1f} inches (expected: ~75-80)")
    print(f"  Min weight: {players_df['weight'].min():.1f} lbs (expected: ~115-140)")
    print(f"  Max weight: {players_df['weight'].max():.1f} lbs (expected: ~200-260)")

    print("\n" + "=" * 60)
    print("✓ ADVANCED IMPUTATION COMPLETE")
    print("=" * 60)
    
    return players_df
