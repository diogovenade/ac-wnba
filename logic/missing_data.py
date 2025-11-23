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


def impute_missing_data_teams(teams_df):
    
    print("=== ADVANCED MISSING DATA IMPUTATION - TEAMS ===\n")
    print("Using statistical and ML-based imputation with proper experimental setup\n")

    # Initial statistics
    print("Initial Missing Data Summary:")
    print(f"  Missing attend values: {teams_df['attend'].isna().sum()}")
    print(f"  Zero attend values: {(teams_df['attend'] == 0).sum()}")
    
    # Check if divID and confID exist (may have been dropped earlier)
    if 'divID' in teams_df.columns:
        print(f"  Empty divID: {(teams_df['divID'] == '').sum()}")
    if 'confID' in teams_df.columns:
        print(f"  Empty confID: {(teams_df['confID'] == '').sum()}")
    print()

    # ==========================================
    # STEP 1: Attendance Imputation
    # ==========================================
    print("=" * 60)
    print("STEP 1: Attendance Imputation (Random Forest Regression)")
    print("=" * 60)
    
    # Create features for attendance prediction
    # Use team performance stats, year, playoff status, etc.
    
    teams_with_attend = teams_df[(teams_df['attend'].notna()) & (teams_df['attend'] > 0)].copy()
    teams_missing_attend = teams_df[(teams_df['attend'].isna()) | (teams_df['attend'] == 0)].copy()
    
    if len(teams_missing_attend) > 0:
        print(f"\nTeams with missing/zero attendance: {len(teams_missing_attend)}")
        
        # Select relevant features for prediction
        feature_cols = ['year', 'won', 'lost', 'rank', 'o_pts', 'd_pts', 'homeW', 'awayW']
        
        # Prepare features - encode playoff status
        X_attend = teams_with_attend[feature_cols].copy()
        X_attend['made_playoffs'] = (teams_with_attend['playoff'] == 'Y').astype(int)
        
        y_attend = teams_with_attend['attend']
        
        # Train-test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_attend, y_attend, test_size=0.2, random_state=42
        )
        
        # Train Random Forest Regressor
        rf_regressor_attend = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            random_state=42
        )
        rf_regressor_attend.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred_test = rf_regressor_attend.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print("\nModel Performance on Test Set:")
        print(f"  Mean Absolute Error: {mae:.0f} attendees")
        print(f"  R² Score: {r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_regressor_attend, X_attend, y_attend, 
                                    cv=5, scoring='neg_mean_absolute_error')
        print(f"  5-Fold CV MAE: {-cv_scores.mean():.0f} (+/- {cv_scores.std() * 2:.0f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_attend.columns,
            'importance': rf_regressor_attend.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop Feature Importances:")
        print(feature_importance.head(5).to_string(index=False))
        
        # Retrain on full data
        rf_regressor_attend.fit(X_attend, y_attend)
        
        # Predict missing attendance
        X_missing_attend = teams_missing_attend[feature_cols].copy()
        X_missing_attend['made_playoffs'] = (teams_missing_attend['playoff'] == 'Y').astype(int)
        
        predicted_attend = rf_regressor_attend.predict(X_missing_attend)
        teams_df.loc[teams_missing_attend.index, 'attend'] = predicted_attend.round(0).astype(int)
        
        print(f"\n✓ Predicted attendance for {len(teams_missing_attend)} teams")
        print(f"  Predicted attendance range: {predicted_attend.min():.0f} - {predicted_attend.max():.0f}")
    else:
        print("\nNo missing attendance to impute")

    # ==========================================
    # STEP 2: Conference/Division Handling
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 2: Conference and Division ID Handling")
    print("=" * 60)
    
    # Check if these columns exist (they may have been dropped earlier in the notebook)
    has_divid = 'divID' in teams_df.columns
    has_confid = 'confID' in teams_df.columns
    
    if not has_divid and not has_confid:
        print("\n✓ divID and confID columns not present (may have been dropped)")
        print("  Skipping conference/division handling")
    else:
        # For confID and divID, use team and year context
        # Fill empty divID with 'Unknown' or use team/year patterns
        
        if has_divid:
            empty_divid = (teams_df['divID'] == '').sum()
            
            if empty_divid > 0:
                # Fill empty divID based on team/year patterns
                for tmID in teams_df['tmID'].unique():
                    team_data = teams_df[teams_df['tmID'] == tmID]
                    known_divs = team_data[team_data['divID'] != '']['divID'].unique()
                    
                    if len(known_divs) > 0:
                        # Use the most common division for this team
                        most_common_div = team_data[team_data['divID'] != '']['divID'].mode()
                        if len(most_common_div) > 0:
                            teams_df.loc[(teams_df['tmID'] == tmID) & (teams_df['divID'] == ''), 'divID'] = most_common_div[0]
                
                # Any remaining empty divIDs are filled with 'NA'
                teams_df['divID'] = teams_df['divID'].replace('', 'NA')
                print(f"\n✓ Filled {empty_divid} empty divID values (team-based patterns + NA fallback)")
            else:
                print("\n✓ No empty divID values to fill")
        
        if has_confid:
            empty_confid = (teams_df['confID'] == '').sum()
            
            if empty_confid > 0:
                # confID is more stable, use team-based imputation
                for tmID in teams_df['tmID'].unique():
                    team_data = teams_df[teams_df['tmID'] == tmID]
                    known_confs = team_data[team_data['confID'] != '']['confID'].unique()
                    
                    if len(known_confs) > 0:
                        most_common_conf = team_data[team_data['confID'] != '']['confID'].mode()
                        if len(most_common_conf) > 0:
                            teams_df.loc[(teams_df['tmID'] == tmID) & (teams_df['confID'] == ''), 'confID'] = most_common_conf[0]
                
                teams_df['confID'] = teams_df['confID'].replace('', 'NA')
                print(f"✓ Filled {empty_confid} empty confID values (team-based patterns + NA fallback)")
            else:
                print("✓ No empty confID values to fill")

    # ==========================================
    # STEP 3: Playoff Round Data Handling
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 3: Playoff Round Data Handling")
    print("=" * 60)
    
    # Fill empty playoff round strings with "NA" for teams that didn't make playoffs
    # or didn't advance to those rounds
    playoff_cols = ['firstRound', 'semis', 'finals']
    existing_playoff_cols = [col for col in playoff_cols if col in teams_df.columns]
    
    if not existing_playoff_cols:
        print("\n✓ Playoff round columns not present (may have been dropped)")
        print("  Skipping playoff round handling")
    else:
        for col in existing_playoff_cols:
            empty_count = (teams_df[col] == '').sum()
            na_count = teams_df[col].isna().sum()
            
            if empty_count > 0 or na_count > 0:
                teams_df[col] = teams_df[col].replace('', 'NA').fillna('NA')
                print(f"\n✓ Filled {col}: {empty_count} empty strings + {na_count} NaN values → 'NA'")
        
        print("\nThese categorical playoff outcomes cannot be reliably predicted,")
        print("so 'NA' represents teams that didn't participate or didn't advance.")

    # ==========================================
    # FINAL VERIFICATION
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    print("\nAfter Advanced Imputation:")
    print(f"  Missing attend values: {teams_df['attend'].isna().sum()}")
    print(f"  Zero attend values: {(teams_df['attend'] == 0).sum()}")
    
    if 'divID' in teams_df.columns:
        print(f"  Empty divID: {(teams_df['divID'] == '').sum()}")
    if 'confID' in teams_df.columns:
        print(f"  Empty confID: {(teams_df['confID'] == '').sum()}")

    print("\nAttendance statistics after imputation:")
    print(teams_df['attend'].describe())

    print("\nSanity Checks:")
    print(f"  Min attendance: {teams_df['attend'].min():.0f} (expected: ~50,000-100,000)")
    print(f"  Max attendance: {teams_df['attend'].max():.0f} (expected: ~150,000-200,000)")
    print(f"  Mean attendance: {teams_df['attend'].mean():.0f}")

    print("\n" + "=" * 60)
    print("✓ ADVANCED IMPUTATION COMPLETE - TEAMS")
    print("=" * 60)
    
    return teams_df
