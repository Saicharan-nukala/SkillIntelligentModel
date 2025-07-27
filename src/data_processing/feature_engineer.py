# src/data_processing/feature_engineer.py

import pandas as pd
import numpy as np
import json
import logging
import os
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize with a cleaned DataFrame"""
        self.df = df.copy()
        self._validate_input()

    def _validate_input(self):
        """Validate input data structure"""
        required_cols = {
            'difficulty_level': 'numeric',
            'learning_time_days': 'numeric',
            'popularity_score': 'numeric',
            'job_demand_score': 'numeric',
            'salary_impact_percent': 'numeric',
            'future_relevance_score': 'numeric',
            'learning_resources_quality': 'numeric',
            'prerequisites': 'list',
            'complementary_skills': 'list',
            'industry_usage': 'list'
        }
        
        for col, col_type in required_cols.items():
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
            
            if col_type == 'numeric' and not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f"Column {col} must be numeric")
            
            if col_type == 'list' and not all(isinstance(x, list) for x in self.df[col]):
                self._convert_string_lists(col)

    def _convert_string_lists(self, col: str):
        """Convert stringified lists to actual lists"""
        try:
            self.df[col] = self.df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            logging.info(f"Converted stringified list column '{col}' to actual lists")
        except Exception as e:
            raise ValueError(f"Failed to parse {col} as JSON: {str(e)}")

    def calculate_skill_complexity_score(self) -> pd.Series:
        """Enhanced complexity score with prerequisites"""
        self.df['prereq_count'] = self.df['prerequisites'].apply(len)
        self.df['skill_complexity_score'] = (
            self.df['difficulty_level'] * 
            self.df['learning_time_days'] * 
            (1 + self.df['prereq_count'] / 10)
        )
        return self.df['skill_complexity_score']

    def calculate_learning_metrics(self) -> Dict[str, pd.Series]:
        """Calculate multiple learning-related metrics"""
        # Avoid division by zero
        learning_time_adj = self.df['learning_time_days'] + 1e-6
        
        self.df['learning_roi'] = (
            self.df['salary_impact_percent'] / learning_time_adj
        )
        self.df['advanced_learning_roi'] = (
            self.df['salary_impact_percent'] * 
            self.df['job_demand_score'] / 
            learning_time_adj
        )
        self.df['learning_accessibility_score'] = (
            self.df['learning_resources_quality'] * 
            (1 / (self.df['difficulty_level'] + 1e-6))
        )
        
        return {
            'learning_roi': self.df['learning_roi'],
            'advanced_learning_roi': self.df['advanced_learning_roi'],
            'learning_accessibility_score': self.df['learning_accessibility_score']
        }

    def calculate_market_metrics(self) -> Dict[str, pd.Series]:
        """Calculate market-related metrics"""
        trend_mapping = {
            'increasing': 1.5,
            'stable': 1.0,
            'decreasing': 0.5
        }
        
        self.df['market_trend_multiplier'] = (
            self.df['market_trend'].map(trend_mapping).fillna(1.0)
        )
        
        self.df['market_momentum_score'] = (
            self.df['popularity_score'] * 
            self.df['market_trend_multiplier']
        )
        
        self.df['skill_momentum_index'] = (
            self.df['popularity_score'] * 
            self.df['job_demand_score'] * 
            self.df['market_trend_multiplier']
        )
        
        return {
            'market_momentum_score': self.df['market_momentum_score'],
            'skill_momentum_index': self.df['skill_momentum_index']
        }

    def calculate_risk_metrics(self) -> Dict[str, pd.Series]:
        """Calculate risk-related metrics"""
        if 'market_trend_multiplier' not in self.df.columns:
            self.calculate_market_metrics()
            
        self.df['risk_of_obsolescence'] = (
            (1 - self.df['future_relevance_score']) * 
            (1 / (self.df['market_trend_multiplier'] + 0.5))
        )
        
        # Normalize to 0-1 range and convert to binary
        self.df['risk_of_obsolescence'] = (
            self.df['risk_of_obsolescence'] / 
            self.df['risk_of_obsolescence'].max()
        )
        self.df['risk_of_obsolescence_binary'] = (
            self.df['risk_of_obsolescence'] > 0.7
        ).astype(int)
        
        return {
            'risk_of_obsolescence': self.df['risk_of_obsolescence'],
            'risk_of_obsolescence_binary': self.df['risk_of_obsolescence_binary']
        }

    def calculate_ecosystem_metrics(self) -> Dict[str, pd.Series]:
        """Calculate ecosystem-related metrics"""
        self.df['ecosystem_richness'] = (
            self.df['complementary_skills'].apply(len) + 
            self.df['prerequisites'].apply(len)
        )
        
        self.df['industry_diversity_metric'] = (
            self.df['industry_usage'].apply(len)
        )
        
        self.df['resource_availability_index'] = (
            self.df['learning_resources_quality'] * 
            self.df['popularity_score']
        )
        
        return {
            'ecosystem_richness': self.df['ecosystem_richness'],
            'industry_diversity_metric': self.df['industry_diversity_metric'],
            'resource_availability_index': self.df['resource_availability_index']
        }

    def engineer_all_features(self) -> pd.DataFrame:
        """Execute all feature engineering steps"""
        logging.info("Starting comprehensive feature engineering...")
        
        # Calculate all metrics
        self.calculate_skill_complexity_score()
        self.calculate_learning_metrics()
        market_metrics = self.calculate_market_metrics()
        risk_metrics = self.calculate_risk_metrics()
        ecosystem_metrics = self.calculate_ecosystem_metrics()
        
        # Add interaction features
        self.df['demand_popularity_interaction'] = (
            self.df['job_demand_score'] * 
            self.df['popularity_score']
        )
        
        self.df['difficulty_squared'] = (
            self.df['difficulty_level'] ** 2
        )
        
        logging.info("Feature engineering complete. Added features:\n" +
                    f"- Market metrics: {list(market_metrics.keys())}\n" +
                    f"- Risk metrics: {list(risk_metrics.keys())}\n" +
                    f"- Ecosystem metrics: {list(ecosystem_metrics.keys())}")
        
        return self.df

if __name__ == "__main__":
    # Example usage
    input_path = 'data/processed/cleaned_skills_data.jsonl'
    output_path = 'data/processed/skills_engineered_features.jsonl'
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_json(input_path, lines=True)
    engineer = FeatureEngineer(df)
    engineered_df = engineer.engineer_all_features()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    engineered_df.to_json(output_path, orient='records', lines=True)
    logging.info(f"Saved engineered features to {output_path}")