"""
Cross-Platform Data Loader for Educational Data Mining
Handles both NeurIPS 2020 and ASSISTments datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class CrossPlatformDataLoader:
    """Enhanced data loader for cross-platform educational datasets"""

    def __init__(self, data_path='data/'):
        """
        Initialize the data loader

        Args:
            data_path: Path to the directory containing data files
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_mappings = {}

    def load_neurips_dataset(self, task='both'):
        """
        Load NeurIPS 2020 Education Challenge dataset

        Args:
            task: 'correctness', 'option', or 'both'

        Returns:
            X: Feature matrix
            y_correct: Correctness labels (if task includes correctness)
            y_option: Option labels (if task includes option)
            features: Feature names
        """
        print("Loading NeurIPS dataset files...")

        # Load all required CSV files
        try:
            train_data = pd.read_csv(os.path.join(self.data_path, 'train_task_1_2_mini.csv'))
            answer_metadata = pd.read_csv(os.path.join(self.data_path, 'answer_metadata_task_1_2_10k.csv'))
            question_metadata = pd.read_csv(os.path.join(self.data_path, 'question_metadata_after.csv'))
            student_metadata = pd.read_csv(os.path.join(self.data_path, 'student_metadata_task_1_2.csv'))
        except FileNotFoundError as e:
            print(f"Error: Required data file not found: {e}")
            raise

        print(f"Loaded train data: {train_data.shape}")
        print(f"Loaded answer metadata: {answer_metadata.shape}")
        print(f"Loaded question metadata: {question_metadata.shape}")
        print(f"Loaded student metadata: {student_metadata.shape}")

        # Data merging
        print("Merging datasets...")
        train_data = pd.merge(train_data, answer_metadata, on='AnswerId', how='left')
        train_data = pd.merge(train_data, question_metadata, on='QuestionId', how='left')

        # Process student metadata
        student_metadata = self._process_student_metadata(student_metadata)
        train_data = pd.merge(train_data, student_metadata, on='UserId', how='left')

        # Feature engineering
        print("Engineering features...")
        train_data = self._engineer_neurips_features(train_data)

        # Feature selection based on task
        if task == 'correctness' or task == 'both':
            features_correct = self._get_correctness_features()
        if task == 'option' or task == 'both':
            features_option = self._get_option_features()

        # Combine features for both tasks
        if task == 'both':
            feature_cols = list(set(features_correct + features_option))
        elif task == 'correctness':
            feature_cols = features_correct
        else:
            feature_cols = features_option

        # Ensure all features exist in the dataframe
        available_features = [f for f in feature_cols if f in train_data.columns]
        missing_features = set(feature_cols) - set(available_features)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")

        # Extract features and labels
        X = train_data[available_features].fillna(train_data[available_features].mean())

        # Convert to numpy array
        X = X.values

        # Prepare labels based on task
        if task == 'correctness':
            y_correct = train_data['IsCorrect'].astype(float).values
            return X, y_correct, None, available_features
        elif task == 'option':
            # Encode answer options
            le = LabelEncoder()
            y_option = le.fit_transform(train_data['AnswerValue'])
            self.label_encoders['option'] = le
            return X, None, y_option, available_features
        else:  # both
            y_correct = train_data['IsCorrect'].astype(float).values
            le = LabelEncoder()
            y_option = le.fit_transform(train_data['AnswerValue'])
            self.label_encoders['option'] = le
            return X, y_correct, y_option, available_features

    def _process_student_metadata(self, student_metadata):
        """Process student metadata with birth year extraction"""

        def extract_year(date_str):
            try:
                return int(str(date_str)[:4]) if pd.notna(date_str) else np.nan
            except:
                return np.nan

        student_metadata['BirthYear'] = student_metadata['DateOfBirth'].apply(extract_year)
        # Keep only students with valid birth year
        student_metadata = student_metadata[student_metadata['BirthYear'].notna()]

        # Process gender if available
        if 'Gender' in student_metadata.columns:
            student_metadata['Gender'] = student_metadata['Gender'].fillna('Unknown')

        # Process premium status if available
        if 'PremiumPupil' in student_metadata.columns:
            student_metadata['PremiumPupil'] = student_metadata['PremiumPupil'].fillna(0).astype(int)

        return student_metadata[['UserId', 'BirthYear'] +
                                [col for col in ['Gender', 'PremiumPupil'] if col in student_metadata.columns]]

    def _engineer_neurips_features(self, df):
        """Engineer features for NeurIPS dataset"""

        # Extract year from dates
        def safe_extract_year(date_str):
            try:
                return int(str(date_str)[:4]) if pd.notna(date_str) else np.nan
            except:
                return np.nan

        # Basic temporal features
        df['AnswerYear'] = df['DateAnswered'].apply(safe_extract_year)
        df['StudentAge'] = df['AnswerYear'] - df['BirthYear']

        # Historical performance features (as per paper's methodology)
        print("Computing historical features...")
        # Sort by user and date for proper historical calculation
        df = df.sort_values(['UserId', 'DateAnswered'])

        df['HistoricalAccuracy'] = df.groupby('UserId')['IsCorrect'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.5)  # Initialize with 0.5 for first attempt

        # Streak features
        df['StreakCorrect'] = self._compute_streak(df, 'IsCorrect', 1)
        df['StreakIncorrect'] = self._compute_streak(df, 'IsCorrect', 0)

        # Question difficulty (empirical)
        df['QuestionDifficulty'] = df.groupby('QuestionId')['IsCorrect'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.5)

        # Student ability estimation
        df['StudentAbility'] = df.groupby('UserId')['IsCorrect'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.5)

        # Interaction features
        df['AbilityDifficultyGap'] = df['StudentAbility'] - df['QuestionDifficulty']

        # Option selection patterns for Task 2
        df['PreviousOptionPattern'] = df.groupby('UserId')['AnswerValue'].transform(
            lambda x: x.shift(1)
        )
        # Encode previous option pattern
        # if 'PreviousOptionPattern' in df.columns:
        #     le = LabelEncoder()
        #     df['PreviousOptionPattern'] = df.groupby('UserId')['PreviousOptionPattern'].transform(
        #         lambda x: le.fit_transform(x.fillna('None'))
        #     )
        if 'PreviousOptionPattern' in df.columns:
            df['PreviousOptionPattern'] = df.groupby('UserId')['PreviousOptionPattern'].transform(
                lambda x: pd.factorize(x.fillna('None'))[0]  # 返回编码后的数字
            )
        df['QuestionOptionCount'] = df.groupby('QuestionId')['AnswerValue'].transform('nunique')

        # User option preference (most frequently selected option)
        df['UserOptionPreference'] = df.groupby('UserId')['AnswerValue'].transform(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
        )
        # Encode user option preference
        if 'UserOptionPreference' in df.columns:
            le = LabelEncoder()
            df['UserOptionPreference'] = le.fit_transform(df['UserOptionPreference'].fillna('None'))

        # Confidence-based features
        if 'Confidence' in df.columns:
            df['ConfidenceCorrectInteraction'] = df['Confidence'] * df['IsCorrect']
            df['LowConfidenceFlag'] = (df['Confidence'] <= 1).astype(int)

        # Time-based features
        df['DateAnswered_dt'] = pd.to_datetime(df['DateAnswered'])
        df['DayOfWeek'] = df['DateAnswered_dt'].dt.dayofweek
        df['Hour'] = df['DateAnswered_dt'].dt.hour

        # Attempt count per question
        df['AttemptNumber'] = df.groupby(['UserId', 'QuestionId']).cumcount() + 1

        # Subject encoding if needed
        if 'SubjectId' in df.columns:
            # Keep SubjectId as numeric
            df['SubjectId'] = df['SubjectId'].fillna(0).astype(int)

        # Correct answer encoding if needed
        if 'CorrectAnswer' in df.columns:
            # Encode correct answer
            le = LabelEncoder()
            df['CorrectAnswer'] = le.fit_transform(df['CorrectAnswer'].fillna('Unknown'))

        return df

    def _compute_streak(self, df, column, value):
        """Compute streak of consecutive values"""

        def streak_calc(series, target_value):
            streak = []
            count = 0
            for val in series:
                streak.append(count)  # ← 先记录，再更新 逻辑bug
                if val == target_value:
                    count += 1
                else:
                    count = 0
                streak.append(count)
            return streak

        streaks = []


        # List of possible column names for user ID (ordered by priority)
        possible_user_id_columns = ['user_id', 'UserId', 'userId', 'UserId']

        # Find the actual column name in the DataFrame
        user_id_column = None
        for col in possible_user_id_columns:
            if col in df.columns:
                user_id_column = col
                break

        # Fallback: case-insensitive match if exact names not found
        if user_id_column is None:
            for col in df.columns:
                if col.lower() in ['user_id', 'userid', 'usreid']:
                    user_id_column = col
                    break

        # Raise error if no valid column found
        if not user_id_column:
            raise KeyError(f"User ID column not found. Tried: {possible_user_id_columns}")

        # Process each user's streak data
        streaks = []
        for user_id in df[user_id_column].unique():
            # Get and sort user's data by date
            #bug -->user_data = df[df[user_id_column] == user_id].sort_values('DateAnswered')
            sort_col = 'DateAnswered' if 'DateAnswered' in df.columns else \
                'order_id' if 'order_id' in df.columns else \
                    df.index.name or 'index'
            user_data = df[df[user_id_column] == user_id]
            if sort_col in df.columns:
                user_data = user_data.sort_values(sort_col)

            # Calculate streak for the specified column and value
            user_streak = streak_calc(user_data[column].values, value)

            # Store results with original index
            for idx, streak_val in zip(user_data.index, user_streak):
                streaks.append((idx, streak_val))

        # for user_id in df['UserId'].unique():
        #     user_data = df[df['UserId'] == user_id].sort_values('DateAnswered')
        #     user_streak = streak_calc(user_data[column].values, value)
        #     for idx, streak_val in zip(user_data.index, user_streak):
        #         streaks.append((idx, streak_val))

        streak_df = pd.DataFrame(streaks, columns=['index', 'streak'])
        streak_df = streak_df.set_index('index')

        return streak_df.reindex(df.index, fill_value=0)['streak']

    def _get_correctness_features(self):
        """Get feature list for correctness prediction"""
        return [
            'CorrectAnswer',
            'HistoricalAccuracy',
            'Confidence',
            'StudentAge',
            'SubjectId',
            'QuestionDifficulty',
            'StudentAbility',
            'AbilityDifficultyGap',
            'StreakCorrect',
            'StreakIncorrect',
            'ConfidenceCorrectInteraction',
            'LowConfidenceFlag',
            'DayOfWeek',
            'Hour',
            'AttemptNumber'
        ]

    def _get_option_features(self):
        """Get feature list for option selection prediction"""
        return [
            'CorrectAnswer',
            'HistoricalAccuracy',
            'Confidence',
            'StudentAge',
            'SubjectId',
            'QuestionOptionCount',
            'PreviousOptionPattern',
            'UserOptionPreference',
            'QuestionDifficulty',
            'StudentAbility',
            'DayOfWeek',
            'Hour'
        ]

    def load_assistments_dataset(self, task='correctness'):
        """
        Load ASSISTments 2009-2010 dataset

        Args:
            task: 'correctness' only (ASSISTments doesn't have option labels)

        Returns:
            X: Feature matrix
            y_correct: Correctness labels
            y_option: None (not available for ASSISTments)
            features: Feature names
        """
        print("Loading ASSISTments dataset...")

        # Check if we have a preprocessed file or need to process raw data
        processed_path = os.path.join(self.data_path, 'assistments_processed.csv')
        raw_path = os.path.join(self.data_path, 'assistments_2009_2010.csv')

        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
        elif os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
        else:
            # Try alternative naming conventions
            alt_paths = [
                'skill_builder_data_corrected.csv',
                'skill_builder_data.csv',
                'assistments_data.csv'
            ]
            df = None
            for alt_path in alt_paths:
                full_path = os.path.join(self.data_path, alt_path)
                if os.path.exists(full_path):
                    df = pd.read_csv(full_path)
                    break

            if df is None:
                raise FileNotFoundError(f"ASSISTments data not found in {self.data_path}")

        print(f"Loaded ASSISTments data: {df.shape}")

        # Map ASSISTments columns to standard names if needed
        column_mapping = {
            'user_id': 'user_id',
            'order_id': 'order_id',
            'problem_id': 'problem_id',
            'correct': 'correct',
            'skill_id': 'skill_id',
            'skill_name': 'skill_name',
            'attempt_count': 'attempt_count',
            'ms_first_response': 'response_time',
            'hint_count': 'hint_count'
        }

        # Rename columns if they exist
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

        # Engineer features similar to NeurIPS
        df = self._engineer_assistments_features(df)

        # Feature columns
        feature_cols = [
            'user_id_encoded', 'problem_id_encoded', 'skill_id_encoded',
            'historical_accuracy', 'response_time_norm', 'confidence_inferred',
            'problem_difficulty', 'user_ability', 'attempt_count',
            'hint_count', 'time_of_day', 'day_of_week',
            'streak_correct', 'streak_incorrect', 'time_since_last'
        ]

        # Filter to available features
        available_features = [f for f in feature_cols if f in df.columns]

        # Extract features and labels
        X = df[available_features].fillna(0).values
        y_correct = df['correct'].values

        # For ASSISTments, create dummy option labels if needed
        y_option = None

        # Standardize features
        X = self.scaler.fit_transform(X)

        if task == 'correctness':
            return X, y_correct, None, available_features
        elif task == 'option':
            return X, None, y_option, available_features
        else:
            return X, y_correct, y_option, available_features

    def _engineer_assistments_features(self, df):
        """Engineer features for ASSISTments dataset"""

        # Encode categorical variables
        for col in ['user_id', 'problem_id', 'skill_id']:
            if col in df.columns:
                le = LabelEncoder()
                # Handle missing values
                df[col] = df[col].fillna('unknown')
                df[f'{col}_encoded'] = le.fit_transform(df[col])

        # Sort by user and order for temporal features
        if 'order_id' in df.columns:
            df = df.sort_values(['user_id', 'order_id'])
        else:
            df = df.sort_values(['user_id'])

        # Historical accuracy
        df['historical_accuracy'] = df.groupby('user_id')['correct'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.5)

        # Response time normalization
        if 'response_time' in df.columns:
            df['response_time_norm'] = np.log1p(df['response_time'].fillna(df['response_time'].median()))
            # Inferred confidence from response time (faster = more confident)
            df['confidence_inferred'] = 1 / (1 + np.exp(
                (df['response_time_norm'] - df['response_time_norm'].mean()) / df['response_time_norm'].std()
            ))
        else:
            df['response_time_norm'] = 0
            df['confidence_inferred'] = 0.5

        # Problem difficulty
        df['problem_difficulty'] = df.groupby('problem_id')['correct'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.5)

        # User ability
        df['user_ability'] = df.groupby('user_id')['correct'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.5)

        # Attempt count (if not already present)
        if 'attempt_count' not in df.columns:
            df['attempt_count'] = df.groupby(['user_id', 'problem_id']).cumcount() + 1

        # Hint count (if not present, set to 0)
        if 'hint_count' not in df.columns:
            df['hint_count'] = 0

        # Time features (if timestamp available)
        if 'start_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['start_time'])
            df['time_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        else:
            # Use order_id as proxy for time if available
            df['time_of_day'] = 12  # Default to noon
            df['day_of_week'] = 3  # Default to Wednesday

        # Streaks
        df['streak_correct'] = self._compute_streak(df, 'correct', 1)
        df['streak_incorrect'] = self._compute_streak(df, 'correct', 0)

        # Time since last attempt
        if 'timestamp' in df.columns:
            df['time_since_last'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        else:
            df['time_since_last'] = 0

        return df

    def create_cross_platform_features(self, neurips_data, assistments_data):
        """
        Create aligned features for cross-platform learning

        Args:
            neurips_data: Tuple of (X, y) for NeurIPS
            assistments_data: Tuple of (X, y) for ASSISTments

        Returns:
            unified_features: List of unified feature names
            feature_mapping: Dictionary mapping between platforms
        """

        # Feature mapping between platforms
        feature_mapping = {
            'neurips': {
                'student_ability': 'StudentAbility',
                'question_difficulty': 'QuestionDifficulty',
                'historical_accuracy': 'HistoricalAccuracy',
                'confidence': 'Confidence',
                'age': 'StudentAge',
                'streak_correct': 'StreakCorrect',
                'streak_incorrect': 'StreakIncorrect',
                'time_of_day': 'Hour',
                'day_of_week': 'DayOfWeek'
            },
            'assistments': {
                'student_ability': 'user_ability',
                'question_difficulty': 'problem_difficulty',
                'historical_accuracy': 'historical_accuracy',
                'confidence': 'confidence_inferred',
                'age': None,  # Not available in ASSISTments
                'streak_correct': 'streak_correct',
                'streak_incorrect': 'streak_incorrect',
                'time_of_day': 'time_of_day',
                'day_of_week': 'day_of_week'
            }
        }

        # Create unified feature set
        unified_features = []

        for universal_name in feature_mapping['neurips'].keys():
            neurips_col = feature_mapping['neurips'][universal_name]
            assistments_col = feature_mapping['assistments'][universal_name]

            if neurips_col and assistments_col:
                # Both platforms have this feature
                unified_features.append(universal_name)

        self.feature_mappings = feature_mapping
        return unified_features, feature_mapping


# Utility functions for data preprocessing
def create_temporal_features(df, date_column='DateAnswered'):
    """Create temporal features from timestamps"""
    df['datetime'] = pd.to_datetime(df[date_column])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Semester indicator
    df['semester'] = df['month'].apply(lambda x: 1 if x <= 6 else 2)

    return df


def handle_missing_data(df, strategy='mean', columns=None):
    """Handle missing data with various strategies"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    if strategy == 'mean':
        df[columns] = df[columns].fillna(df[columns].mean())
    elif strategy == 'median':
        df[columns] = df[columns].fillna(df[columns].median())
    elif strategy == 'forward_fill':
        df[columns] = df[columns].fillna(method='ffill')
    elif strategy == 'interpolate':
        df[columns] = df[columns].interpolate()

    return df


def create_interaction_features(df):
    """Create interaction features between key variables"""
    interactions = []

    if 'StudentAbility' in df.columns and 'QuestionDifficulty' in df.columns:
        df['ability_difficulty_gap'] = df['StudentAbility'] - df['QuestionDifficulty']
        df['ability_difficulty_product'] = df['StudentAbility'] * df['QuestionDifficulty']
        interactions.extend(['ability_difficulty_gap', 'ability_difficulty_product'])

    if 'Confidence' in df.columns and 'HistoricalAccuracy' in df.columns:
        df['confidence_accuracy_product'] = df['Confidence'] * df['HistoricalAccuracy']
        interactions.append('confidence_accuracy_product')

    return df, interactions

def extract_aligned_features(df, source='neurips'):
    """
    提取并对齐 NeurIPS / ASSISTments 的公共字段，确保特征结构一致。
    """
    rename_dict = {}

    if source == 'neurips':
        rename_dict = {
            'question_difficulty': 'problem_difficulty',
            'confidence': 'confidence_inferred',
            'group_performance': 'class_performance'
        }
    elif source == 'assistments':
        rename_dict = {
            'problem_difficulty': 'problem_difficulty',
            'confidence_inferred': 'confidence_inferred',
            'class_performance': 'class_performance'
        }

    df_renamed = df.rename(columns=rename_dict)

    aligned_cols = [
        'historical_accuracy', 'streak_correct', 'streak_incorrect',
        'student_ability', 'problem_difficulty', 'ability_difficulty_gap',
        'confidence_inferred', 'class_performance'
    ]

    df_aligned = df_renamed[[col for col in aligned_cols if col in df_renamed.columns]].copy()

    return df_aligned

# Main function for testing
if __name__ == "__main__":
    # Test data loading
    loader = CrossPlatformDataLoader(data_path='./data/')

    # Load NeurIPS dataset for both tasks
    print("Loading NeurIPS dataset...")
    try:
        X_neurips, y_correct, y_option, features = loader.load_neurips_dataset(task='both')

        print(f"\nNeurIPS Dataset Summary:")
        print(f"Features shape: {X_neurips.shape}")
        print(f"Correctness labels shape: {y_correct.shape}")
        print(f"Option labels shape: {y_option.shape}")
        print(f"Number of features: {len(features)}")
        print(f"Feature names: {features}")

        # Basic statistics
        print(f"\nClass distribution (correctness): {np.bincount(y_correct.astype(int))}")
        print(f"Number of unique options: {len(np.unique(y_option))}")
    except Exception as e:
        print(f"Error loading NeurIPS data: {e}")

    # Load ASSISTments dataset
    print("\n\nLoading ASSISTments dataset...")
    try:
        X_assist, y_assist, _, features_assist = loader.load_assistments_dataset(task='correctness')

        print(f"\nASSISTments Dataset Summary:")
        print(f"Features shape: {X_assist.shape}")
        print(f"Labels shape: {y_assist.shape}")
        print(f"Number of features: {len(features_assist)}")
        print(f"Feature names: {features_assist}")

        # Basic statistics
        print(f"\nClass distribution: {np.bincount(y_assist.astype(int))}")
    except Exception as e:
        print(f"Error loading ASSISTments data: {e}")