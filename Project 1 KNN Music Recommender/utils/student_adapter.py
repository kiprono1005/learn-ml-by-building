"""
Student Implementation Adapter for Music Recommender Web API
=============================================================
This file bridges your notebook implementation with the Flask web application.

INSTRUCTIONS:
1. Copy your COMPLETE, TESTED implementations below
2. Do NOT modify the get_recommendations_for_api function
3. Test using: python -m utils.test_student_adapter
"""

import numpy as np
import pandas as pd
import os

# ============================================================================
# STUDENT IMPLEMENTATION SECTION
# Copy your complete, final implementations from the notebook below
# ============================================================================

class FeatureScaler:
    """
    TODO: Copy your complete FeatureScaler implementation here.
    Must include: __init__, fit, transform, fit_transform methods
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """
        Learn mean and std for each feature in X.
        
        Args:
            X (np.ndarray): Shape (n_samples, n_features)
            
        Sets:
            self.mean: Mean per feature
            self.std: Std per feature (replace 0 with 1)
        """
        # --- SOLUTION ---
        # TODO: Implement
        # NOTE: Make sure you set the values of self.mean and self.std
        # No need to return any values
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
    
    def transform(self, X):
        """
        Apply scaling: (X - mean) / std
        
        Args:
            X (np.ndarray): Shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Scaled data, same shape as X
            
        Raises:
            RuntimeError: If not fitted yet
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        # --- SOLUTION ---
        # TODO: Implement
        return (X - self.mean) / self.std
        
            
    def fit_transform(self, X):
        """A convenience method to fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class KNNRecommender:
    """
    TODO: Copy your complete KNNRecommender implementation here.
    Must include all methods and static distance functions.
    """
    
    def __init__(self, k=10):
        self.k = k
        self.item_profile = None
        self.features_matrix = None
        self.feature_columns = None
        self.track_id_to_index = {}
        
    @staticmethod
    def euclidean_distance(a, b):
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            a (np.ndarray): Vector of shape (n,)
            b (np.ndarray): Vector of shape (n,)
            
        Returns:
            float: Euclidean distance
            
        Example:
            >>> euclidean_distance(np.array([1,2,3]), np.array([4,5,6]))
            5.196152422706632
        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        # --- SOLUTION ---
        result = np.subtract(a, b) # Computes the difference between the corresponding elements in each array
        result = np.power(result, 2) # Squares every element in the combined array
        result_sum = np.sum(result) # Sums all the squared elements together

        return np.sqrt(result_sum) # Returns the square root of the summed elements
        

    @staticmethod
    def cosine_distance(a, b):
        """
        Calculates the Cosine distance between two numerical vectors a and b.
        Formula: 1 - (a·b) / (||a|| * ||b||)
                
        Args:
            a (np.ndarray): Vector of shape (n,)
            b (np.ndarray): Vector of shape (n,)
            
        Returns:
            float: Cosine distance (between 0 and 1)
            
        Example:
            >>> cosine_distance(np.array([1,2,3]), np.array([4,5,6]))
            0.025368153802923787
            
        Note: Return 1.0 if either vector has zero norm.
        """
        # --- YOUR IMPLEMENTATION GOES HERE ---
        # --- SOLUTION ---
        dot_product = np.dot(a, b)
        a_mag = np.linalg.norm(a)
        b_mag = np.linalg.norm(b)
        
        if a_mag == 0 or b_mag == 0:
            return 1.0
        return 1 - (dot_product / (a_mag * b_mag))
        
        
    def fit(self, item_profile_df, feature_columns):
        """Prepares the recommender by loading and processing the track data."""
        self.item_profile = item_profile_df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.features_matrix = self.item_profile[self.feature_columns].values
        self.track_id_to_index = {track_id: i for i, track_id in enumerate(self.item_profile['id'])}
        print(f"Fit complete. Loaded {len(self.item_profile)} tracks.")

    def find_neighbors(self, track_id, n_neighbors=None, distance_metric='euclidean'):
        """
        Find k nearest neighbors for a track.
        
        Args:
            track_id (str): Query track ID
            n_neighbors (int): Number of neighbors (default: self.k)
            distance_metric (str): 'euclidean' or 'cosine'
            
        Returns:
            list: [(distance, track_id), ...] sorted by distance
            
        Example:
            >>> neighbors = recommender.find_neighbors('track123', n_neighbors=5)
            [(0.23, 'track456'), (0.31, 'track789'), ...]
        """
        if n_neighbors is None: n_neighbors = self.k
        distance_functions = {'euclidean': self.euclidean_distance, 'cosine': self.cosine_distance}
        if distance_metric not in distance_functions: raise ValueError(f"Unknown metric: {distance_metric}")
        if track_id not in self.track_id_to_index: raise ValueError(f"Track ID {track_id} not found.")

        # --- YOUR IMPLEMENTATION GOES HERE ---
        # --- SOLUTION ---
        # TODO: Implement k-NN search
        # Don't include the query track itself in results
        distance_func = distance_functions[distance_metric]
        target_index = self.track_id_to_index[track_id]
        target_features = self.features_matrix[target_index]

        distances = []
        for other_track_id, other_index in self.track_id_to_index.items():
            if other_track_id != track_id:
                other_features = self.features_matrix[other_index]
                distance = distance_func(target_features, other_features)
                distances.append((distance, other_track_id))

        distances.sort()

        return distances[:n_neighbors]
        

    def recommend(self, track_id, n_recommendations=None, distance_metric='euclidean'):
        if self.item_profile is None: raise RuntimeError("Recommender has not been fitted.")
        neighbors = self.find_neighbors(track_id, n_recommendations, distance_metric)
        neighbor_ids = [tid for distance, tid in neighbors]
        results_df = self.item_profile[self.item_profile['id'].isin(neighbor_ids)].copy()
        distances_map = {tid: dist for dist, tid in neighbors}
        results_df['distance'] = results_df['id'].map(distances_map)
        return results_df.sort_values('distance')


# Optional: If you implemented HybridKNNRecommender, add it here
class HybridKNNRecommender(KNNRecommender):
    """
    OPTIONAL: If you completed the hybrid distance implementation, copy it here.
    """
    def custom_hybrid_distance(self, track_a_data, track_b_data, audio_features_a, audio_features_b, 
                              w_artist, w_date, w_audio, context):
        """
        Design your hybrid distance function here.
        
        Args:
            track_a_data (pd.Series): Full data row for track A
            track_b_data (pd.Series): Full data row for track B
            audio_features_a (np.ndarray): Audio feature vector for track A
            audio_features_b (np.ndarray): Audio feature vector for track B
            w_artist (float): Weight for metadata component (0-1)
            
        Returns:
            float: Combined distance value
            
        Ideas to consider:
            - Audio similarity (using cosine or euclidean distance)
            - Artist similarity (same artist = lower distance)
            - You could also consider: genre, year, popularity, etc.
        """
        # TODO: Implement your hybrid distance
        if track_a_data['artist'] == track_b_data['artist']:
            artist_distance = 0.0
        else:
            artist_distance = 1.0 
       
        date_distance = 0.5
        try:
            if 'date' in track_a_data and 'date' in track_b_data:
                year_a = int(track_a_data['date'][:4])
                year_b = int(track_b_data['date'][:4])
                year_diff = abs(year_a - year_b)
                
                if year_diff == 0:
                    date_distance = 0.0
                elif year_diff <= 2:
                    date_distance = 0.1
                elif year_diff <= 5:
                    date_distance = 0.2
                elif year_diff <= 10:
                    date_distance = 0.4
                else:
                    date_distance = 0.7
                    
        except (ValueError, TypeError, KeyError):
            date_distance = 0.5
        
        # Simple audio distance using existing cosine_distance function
        if context == 'general':
            # Use simple euclidean distance for general context (maintains backward compatibility)
            audio_distance = self.euclidean_distance(audio_features_a, audio_features_b)
        else:
            # Use context-aware audio distance for specific contexts
            audio_distance = self.calculate_context_audio_distance(
                audio_features_a, audio_features_b, track_a_data, track_b_data, context)
        
        # Normalize weights
        total_weight = w_artist + w_date + w_audio
        if total_weight > 0:
            w_artist_norm = w_artist / total_weight
            w_date_norm = w_date / total_weight
            w_audio_norm = w_audio / total_weight
        else:
            w_artist_norm = w_date_norm = w_audio_norm = 1/3
        
        # Combined distance
        hybrid_distance = (w_artist_norm * artist_distance + 
                          w_date_norm * date_distance + 
                          w_audio_norm * audio_distance)
        
        return hybrid_distance
        
    def calculate_context_audio_distance(self, audio_features_a, audio_features_b, 
                                         track_a_data, track_b_data, context):
        context_weights = {
            'workout': {
                'energy': 0.35,
                'valence': 0.1,
                'danceability': 0.25,
                'acousticness': 0.05,
                'tempo': 0.2,
                'loudness': 0.05,
                'speechiness': 0.0,
                'instrumentalness': 0.0,
                'liveness': 0.0
            },
            'chill': {
                'energy': 0.25,
                'valence': 0.2,
                'danceability': 0.08,
                'acousticness': 0.18,
                'tempo': 0.12,
                'loudness': 0.04,
                'speechiness': 0.02,
                'instrumentalness': 0.1,
                'liveness': 0.01
            },
            'study': {
                'energy': 0.1,
                'valence': 0.05,
                'danceability': 0.0,
                'acousticness': 0.15,
                'tempo': 0.1,
                'loudness': 0.05,
                'speechiness': 0.05,
                'instrumentalness': 0.5,
                'liveness': 0.0
            },
            'party': {
                'energy': 0.3,
                'valence': 0.25,
                'danceability': 0.35,
                'acousticness': 0.0,
                'tempo': 0.1,
                'loudness': 0.0,
                'speechiness': 0.0,
                'instrumentalness': 0.0,
                'liveness': 0.0
            },
            'general': {
                'energy': 0.112,
                'valence': 0.111,
                'danceability': 0.111,
                'acousticness': 0.111,
                'tempo': 0.111,
                'loudness': 0.111,
                'speechiness': 0.111,
                'instrumentalness': 0.111,
                'liveness': 0.111
            }
        }
        
        # Get weights for the context
        weights = context_weights.get(context, context_weights['general'])
        
        feature_names = ['energy', 'danceability', 'acousticness', 'valence', 'tempo', 
                        'instrumentalness', 'loudness', 'liveness', 'speechiness']
        
        # Create weight vector based on context (features are already scaled)
        weight_vector = np.array([weights.get(name, 0.0) for name in feature_names])
        
        # Apply weights to the scaled features using square root for euclidean distance
        weighted_features_a = audio_features_a * np.sqrt(weight_vector)
        weighted_features_b = audio_features_b * np.sqrt(weight_vector)
        
        # Use the existing euclidean_distance function for consistency
        return self.euclidean_distance(weighted_features_a, weighted_features_b)


    def find_neighbors(self, track_id, n_neighbors=None, distance_metric='hybrid', w_artist=0.5, 
                       w_date=0.2, w_audio=0.4, context='general'):
        """
        Find neighbors using hybrid distance that combines audio features and metadata.
        
        This method extends the base KNNRecommender to use the custom_hybrid_distance
        function when distance_metric='hybrid'.
        """
        if distance_metric != 'hybrid':
            return super().find_neighbors(track_id, n_neighbors, distance_metric)
        
        if n_neighbors is None: 
            n_neighbors = self.k
            
        if track_id not in self.track_id_to_index: 
            raise ValueError(f"Track ID {track_id} not found.")
        
        # --- YOUR IMPLEMENTATION GOES HERE ---
        # TODO: Implement hybrid k-NN search
        # 1. Get query track's features and metadata
        # 2. For each other track:
        #    - Calculate hybrid distance using custom_hybrid_distance
        #    - Store (distance, track_id)
        # 3. Sort and return top n_neighbors
        query_index = self.track_id_to_index[track_id]
        query_track_data = self.item_profile.iloc[query_index]
        query_audio_features = self.features_matrix[query_index]
        
        distances = []
        
        for other_track_id, other_index in self.track_id_to_index.items():
            if other_track_id == track_id:
                continue
                
            other_track_data = self.item_profile.iloc[other_index]
            other_audio_features = self.features_matrix[other_index]
            
            # Calculate hybrid distance
            hybrid_dist = self.custom_hybrid_distance(
                track_a_data=query_track_data,
                track_b_data=other_track_data,
                audio_features_a=query_audio_features,
                audio_features_b=other_audio_features,
                w_artist=w_artist,
                w_date=w_date,
                w_audio=w_audio,
                context=context
            )
            
            distances.append((hybrid_dist, other_track_id))
        
        # Sort by distance
        distances.sort()
        
        return distances[:n_neighbors]



# ============================================================================
# API ADAPTER SECTION - DO NOT MODIFY ANYTHING BELOW THIS LINE
# ============================================================================

# Cache for the recommender instance
_recommender_cache = None
_audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                   'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']


def get_recommendations_for_api(track_id, k=10, metric='cosine', use_hybrid=False):
    """
    Bridge function between student implementation and web API.
    DO NOT MODIFY THIS FUNCTION.
    """
    global _recommender_cache
    
    try:
        # Load data and initialize recommender if needed
        if _recommender_cache is None:
            # Find the data file
            possible_paths = [
                'data/mergedFile.csv',
                '../data/mergedFile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
                'data/item_profile.csv',
                '../data/item_profile.csv',
                os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
            ]
            
            item_profile = None
            for path in possible_paths:
                if os.path.exists(path):
                    item_profile = pd.read_csv(path, dtype={'id': str})
                    break
            
            if item_profile is None:
                raise FileNotFoundError("Could not find mergedFile.csv or item_profile.csv")
            
            # Initialize the appropriate recommender
            if use_hybrid and 'HybridKNNRecommender' in globals():
                _recommender_cache = HybridKNNRecommender(k=k)
            else:
                _recommender_cache = KNNRecommender(k=k)
            
            # Use only features that exist in the loaded file
            available_feats = [f for f in _audio_features if f in item_profile.columns]
            if not available_feats:
                raise ValueError("No required audio feature columns found in data file")

            _recommender_cache.fit(item_profile, available_feats)
            print(f"Initialized {type(_recommender_cache).__name__} with {len(item_profile)} tracks")
        
        # Get recommendations
        recommendations = _recommender_cache.recommend(
            track_id, 
            n_recommendations=k, 
            distance_metric=metric
        )
        
        # Convert to API format
        result = {}
        for _, row in recommendations.iterrows():
            result[row['id']] = {
                'distance': float(row['distance']),
                'song': row.get('song', 'Unknown'),
                'artist': row.get('artist', 'Unknown'),
                'features': {feat: float(row.get(feat, 0)) for feat in _audio_features if feat in row}
            }
        
        return result
        
    except Exception as e:
        print(f"Error in student implementation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def test_implementation():
    """
    Test function to verify your implementation works.
    Run this after copying your code above.
    """
    try:
        from utils.test_student_adapter import run_comprehensive_tests
        return run_comprehensive_tests()
    except ImportError:
        # Fallback if test file is not in expected location
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-m', 'utils.test_student_adapter'], 
                              capture_output=False)
        return result.returncode == 0


if __name__ == "__main__":
    print("To test your implementation, run:")
    print("  python -m utils.test_student_adapter")             

# ============================================================================
# AUTO-INITIALIZATION FOR API INTEGRATION
# ============================================================================

def initialize_for_api():
    """
    Initialize the recommender for API usage.
    This is called when api_helpers imports this module.
    """
    import os
    import pandas as pd
    
    # Try to find and load the data (prefer mergedFile.csv which has full feature set)
    possible_paths = [
        'data/mergedFile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'mergedFile.csv'),
        'data/item_profile.csv',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'item_profile.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, dtype={'id': str})
                audio_features = ['energy', 'danceability', 'acousticness', 'valence', 
                                 'tempo', 'instrumentalness', 'loudness', 'liveness', 'speechiness']
                
                # Create and fit the recommender
                recommender = KNNRecommender(k=10)
                recommender.fit(df, audio_features)
                
                print(f"✅ Student recommender initialized with {len(df)} tracks")
                return recommender
                
            except Exception as e:
                print(f"Error loading data from {path}: {e}")
                continue
    
    print("⚠️ Could not initialize student recommender - data files not found")
    return None

# Export the initialized recommender for api_helpers to use (disabled to avoid duplicate init; api_helpers handles it)
# student_recommender_instance = initialize_for_api()