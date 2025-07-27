# Data validation functions
"""
Data Validator Module for Skill Intelligence Model
Implements Step 1: Data Quality Assessment from the plan
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any] # This stats is for specific record validation, not the aggregated one

class SkillDataValidator:
    """
    Validates skill dataset according to the defined schema and business rules
    """

    def __init__(self):
        self.required_fields = [
            'skill_name', 'category', 'skill_type', 'difficulty_level',
            'learning_time_days', 'popularity_score', 'job_demand_score',
            'salary_impact_percent', 'prerequisites', 'complementary_skills',
            'market_trend', 'industry_usage', 'certification_available',
            'future_relevance_score', 'learning_resources_quality'
        ]

        self.field_types = {
            'skill_name': str,
            'category': str,
            'skill_type': str,
            'difficulty_level': int,
            'learning_time_days': int,
            'popularity_score': float,
            'job_demand_score': float,
            'salary_impact_percent': int,
            'prerequisites': list,
            'complementary_skills': list,
            'market_trend': str,
            'industry_usage': list,
            'certification_available': bool,
            'future_relevance_score': float,
            'learning_resources_quality': float
        }

        self.numerical_ranges = {
            'difficulty_level': (1, 5),
            'learning_time_days': (0, 365),
            'popularity_score': (0, 10),
            'job_demand_score': (0, 10),
            'salary_impact_percent': (0, 100),
            'future_relevance_score': (0, 10),
            'learning_resources_quality': (0, 10)
        }

        # Base trend categories with their variations
        self.market_trend_categories = {
            'increasing': [
                'increasing', 'rapidly increasing', 'exponentially increasing',
                'slowly increasing', 'increasing (due to industry challenges)',
                'increasing (niche)', 'increasing (research)', 'increasing (with ARM adoption)',
                'increasing rapidly'
            ],
            'stable': [
                'stable', 'niche, stable', 'stable (SAP ecosystem)', 'stable (academic)',
                'stable (academic/niche)', 'stable (academic/research)', 'stable (enterprise)',
                'stable (evolving with tech)', 'stable (for indie/retro)', 'stable (for large enterprises)',
                'stable (highly niche)', 'stable (industrial)', 'stable (legacy system support)',
                'stable (niche)', 'stable (still widely used)', 'stable (very niche)',
                'stable (within AWS ecosystem)', 'stable (within domain)'
            ],
            'declining': [
                'declining', 'decreasing', 'rapidly declining', 'rapidly declining (legacy maintenance)',
                'rapidly declining (legacy)', 'declining (due to newer automation tools)',
                'declining (replaced by online tools)', 'declining (replaced by other DBs)',
                'declining (specialized)', 'declining (traditional fuels)', 'declining (traditional)',
                'decreasing (for direct game logic)', 'decreasing (towards containers/cloud)'
            ],
            'emerging': [
                'emerging', 'transforming'
            ],
            'niche': [
                'niche/historical'
            ]
        }

        # Skill type categories with their variations
        self.skill_type_categories = {
            'Technical': [
                'Technical', 'Programming', 'Programming Language', 'Software Framework',
                'Software Tool', 'Technical Art', 'Technical Artistic', 'Technical/Analytical',
                'Technology', 'Technology/Operational', 'Technology/Practical', 'Hardware',
                'Programming/Tool', 'Software/Analysis', 'Software/System'
            ],
            'Business': [
                'Business', 'Business/Finance', 'Business/Methodology', 'Business/Technical',
                'Financial', 'Financial/Strategic', 'Marketing', 'Marketing/Strategy',
                'Sales', 'Sales/Marketing', 'Sales/Strategic'
            ],
            'Management': [
                'Management', 'Project Management', 'Management/Legal', 'Management/Process',
                'HR', 'HR/Analytical', 'HR/Management', 'HR/Strategic', 'Change Management',
                'Risk Management', 'Practical/Management'
            ],
            'Soft Skill': [
                'Soft', 'Soft Skill', 'Personal Effectiveness', 'Soft Skill/Change Management',
                'Soft Skill/Ethics', 'Soft Skill/Methodology', 'Soft/Technical', 'HR/Soft Skill',
                'Customer Focused'
            ],
            'Design': [
                'Design', 'Artistic', 'Creative/Methodology', 'Design Philosophy',
                'Design Principle', 'Design/Applied Science', 'Design/Planning', 'Design/Process'
            ],
            'Process/Methodology': [
                'Methodology', 'Process', 'Methodology/Tool', 'Tool/Process', 'Process/Legal',
                'Practical/Process', 'Paradigm'
            ],
            'Legal/Compliance': [
                'Legal', 'Legal/Business', 'Legal/Compliance', 'Legal/Ethics',
                'Legal/Risk', 'Legal/Strategic', 'Operational/Legal'
            ],
            'Strategic': [
                'Strategic', 'Strategic Tool', 'Strategic/Ethics', 'Strategic/Operational',
                'Operational/Strategic'
            ],
            'Analytical': [
                'Analysis', 'Analytical', 'Data Management'
            ],
            'Tool': [
                'Tool', 'Platform', 'Framework', 'Operating System', 'Role/Tool'
            ],
            'Domain Specific': [
                'Domain', 'Domain Specific', 'Specialized', 'Specialized Domain',
                'Material Science', 'Interdisciplinary'
            ]
        }

        # Create reverse mapping for skill types
        self.skill_type_to_category = {}
        for category, types in self.skill_type_categories.items():
            for skill_type in types:
                self.skill_type_to_category[skill_type.lower()] = category

        # Category groupings (from your analysis)
        self.category_groups = {
            'Engineering': [
                'CSE', 'Civil', 'ECE', 'EEE', 'Mechanical'
            ],
            'Technology': [
                'AI / ML', 'Cloud', 'Cybersecurity', 'Data Science & Analytics',
                'DevOps', 'Mobile Development', 'Programming & Technical Skills',
                'Programming Languages', 'Software Development (SDE)', 'Web Development'
            ],
            'Business': [
                'Accounting', 'Business & Management', 'Finance', 'Production & Business'
            ],
            'Creative': [
                'Design', 'Game Art & Animation', 'Game Audio', 'Game Design & Theory',
                'UI/UX', 'Writing'
            ],
            'Skills': [
                'Soft', 'Soft Skills', 'Soft Skills & General Professionalism',
                'Language', 'SEO'
            ],
            'Tools': [
                'Office Tools', 'Tool', 'Technical'
            ],
            'Methodology': [
                'Methodology'
            ]
        }

        # Create reverse mapping for quick lookup
        self.trend_to_category = {}
        for category, trends in self.market_trend_categories.items():
            for trend in trends:
                self.trend_to_category[trend.lower()] = category

        # Create reverse mapping for categories
        self.category_to_group = {}
        for group, categories in self.category_groups.items():
            for category in categories:
                self.category_to_group[category.lower()] = group

    def get_normalized_field_values(self, field_name: str, value: str) -> Tuple[str, str]:
        """
        Get normalized field values and their categories

        Args:
            field_name: Name of the field
            value: Original field value

        Returns:
            Tuple of (normalized_value, category)
        """
        normalized_value = value.lower().strip()

        if field_name == 'market_trend':
            category = self.trend_to_category.get(normalized_value, 'unknown')
            return normalized_value, category
        elif field_name == 'skill_type':
            category = self.skill_type_to_category.get(normalized_value, 'unknown')
            return normalized_value, category
        elif field_name == 'category':
            group = self.category_to_group.get(normalized_value, 'unknown')
            return normalized_value, group
        else:
            return normalized_value, 'N/A'

    def validate_dataset(self, data: List[Dict]) -> ValidationResult:
        """
        Comprehensive validation of the entire dataset

        Args:
            data: List of skill records

        Returns:
            ValidationResult object with validation details
        """
        logger.info(f"Starting validation of {len(data)} records")

        errors = []
        warnings = []
        stats = {
            'total_records': len(data),
            'valid_records': 0,
            'duplicate_skills': 0,
            'missing_fields': defaultdict(int),
            'invalid_ranges': defaultdict(int),
            'circular_dependencies': [],
            'trend_categories': defaultdict(int),      # Initialize as defaultdict(int)
            'skill_type_categories': defaultdict(int), # Initialize as defaultdict(int)
            'category_groups': defaultdict(int)        # Initialize as defaultdict(int)
        }

        # Check for empty dataset
        if not data:
            errors.append("Dataset is empty")
            return ValidationResult(False, errors, warnings, stats)

        # Validate each record
        valid_count = 0
        skill_names = []

        for idx, record in enumerate(data):
            record_validation = self._validate_record(record, idx, stats) # Pass the main stats dictionary

            if record_validation.is_valid:
                valid_count += 1
                skill_names.append(record.get('skill_name', ''))
            else:
                errors.extend(record_validation.errors)
                warnings.extend(record_validation.warnings)

            # Update statistics for missing fields (already handled within _validate_record for other stats)
            for field in self.required_fields:
                if field not in record or record[field] is None:
                    stats['missing_fields'][field] += 1


        stats['valid_records'] = valid_count

        # Check for duplicates
        duplicate_skills = self._find_duplicates(skill_names)
        stats['duplicate_skills'] = len(duplicate_skills)
        if duplicate_skills:
            warnings.append(f"Found {len(duplicate_skills)} duplicate skill names: {duplicate_skills[:5]}...")

        # Check for circular dependencies
        circular_deps = self._check_circular_dependencies(data)
        stats['circular_dependencies'] = circular_deps
        if circular_deps:
            errors.append(f"Found {len(circular_deps)} circular dependencies in prerequisites")

        # Overall validation result
        is_valid = len(errors) == 0 and stats['valid_records'] > 0

        logger.info(f"Validation complete. Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}")

        return ValidationResult(is_valid, errors, warnings, stats)

    def _validate_record(self, record: Dict, index: int, overall_stats: Dict) -> ValidationResult:
        """Validate a single skill record"""
        errors = []
        warnings = []
        # No local 'stats' for aggregation here, directly update overall_stats

        # Check required fields
        for field in self.required_fields:
            if field not in record:
                errors.append(f"Record {index}: Missing required field '{field}'")
                continue

            value = record[field]

            # Check for None values
            if value is None:
                errors.append(f"Record {index}: Field '{field}' is None")
                continue

            # Type validation
            expected_type = self.field_types[field]
            if not isinstance(value, expected_type):
                errors.append(f"Record {index}: Field '{field}' has invalid type. Expected {expected_type.__name__}, got {type(value).__name__}")
                continue

            # Range validation for numerical fields
            if field in self.numerical_ranges:
                min_val, max_val = self.numerical_ranges[field]
                if not (min_val <= value <= max_val):
                    errors.append(f"Record {index}: Field '{field}' value {value} out of range [{min_val}, {max_val}]")

            # Specific field validations
            if field == 'skill_name' and not value.strip():
                errors.append(f"Record {index}: skill_name cannot be empty")

            elif field == 'market_trend':
                # Normalize and categorize market trend
                normalized_trend = value.lower().strip()
                if normalized_trend not in self.trend_to_category:
                    warnings.append(f"Record {index}: Unknown market_trend '{value}' - please verify")
                else:
                    # Add normalized category to stats for analysis
                    category = self.trend_to_category[normalized_trend]
                    overall_stats['trend_categories'][category] += 1 # Update overall_stats

            elif field == 'skill_type':
                # Normalize and categorize skill type
                normalized_type = value.lower().strip()
                if normalized_type not in self.skill_type_to_category:
                    warnings.append(f"Record {index}: Unknown skill_type '{value}' - please verify")
                else:
                    category = self.skill_type_to_category[normalized_type]
                    overall_stats['skill_type_categories'][category] += 1 # Update overall_stats

            elif field == 'category':
                # Normalize and group category
                normalized_cat = value.lower().strip()
                if normalized_cat in self.category_to_group:
                    group = self.category_to_group[normalized_cat]
                    overall_stats['category_groups'][group] += 1 # Update overall_stats

            elif field in ['prerequisites', 'complementary_skills', 'industry_usage']:
                if not isinstance(value, list):
                    errors.append(f"Record {index}: Field '{field}' must be a list")
                elif len(value) > 10:  # Reasonable limit
                    warnings.append(f"Record {index}: Field '{field}' has many entries ({len(value)}). Consider reviewing for relevance.")

                # Check for non-string elements in lists
                if not all(isinstance(item, str) for item in value):
                    errors.append(f"Record {index}: Field '{field}' must contain only string elements.")

        return ValidationResult(len(errors) == 0, errors, warnings, {}) # stats field here is not used for these aggregate counts

    def _find_duplicates(self, skill_names: List[str]) -> List[str]:
        """Find duplicate skill names"""
        skill_counts = defaultdict(int)
        for skill in skill_names:
            skill_counts[skill] += 1
        return [skill for skill, count in skill_counts.items() if count > 1]

    def _check_circular_dependencies(self, data: List[Dict]) -> List[List[str]]:
        """
        Check for circular dependencies in skill prerequisites

        Args:
            data: List of skill records

        Returns:
            List of detected circular dependency cycles
        """
        graph = defaultdict(list)
        all_skills = set() # To keep track of all unique skills/nodes

        for record in data:
            skill_name = record.get('skill_name')
            prerequisites = record.get('prerequisites', [])

            if skill_name:
                all_skills.add(skill_name) # Add skill_name to all_skills
                for prereq in prerequisites:
                    graph[skill_name].append(prereq)
                    all_skills.add(prereq) # Add prerequisite to all_skills too

        # Ensure all skills found are keys in the graph, even if they have no outgoing edges
        # This prevents RuntimeError if a prerequisite itself is not a primary skill
        for skill in all_skills:
            if skill not in graph:
                graph[skill] = [] # Initialize with an empty list if not already a key

        # Using DFS to detect cycles
        visited = set()
        recursion_stack = set()
        cycles = []

        def dfs(node, path):
            visited.add(node)
            recursion_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor])
                elif neighbor in recursion_stack:
                    cycle_start_index = path.index(neighbor)
                    cycles.append(path[cycle_start_index:] + [neighbor])
            recursion_stack.remove(node)

        # Iterate over a static list of all skills to avoid dictionary size change during iteration
        for skill in list(all_skills): # Iterate over a copy or the set directly
            if skill not in visited: # Only start DFS if not already visited
                dfs(skill, [skill])

        return cycles

    def generate_validation_report(self, result: ValidationResult) -> str:
        """
        Generate a human-readable validation report

        Args:
            result: ValidationResult object

        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("Skill Data Validation Report")
        report.append("=" * 60)
        report.append(f"Overall Status: {'VALID' if result.is_valid else 'INVALID'}")
        report.append(f"Total Records Processed: {result.stats['total_records']}")
        report.append(f"Valid Records: {result.stats['valid_records']}")
        report.append("\nSummary of Issues:")
        report.append(f"  Errors Found: {len(result.errors)}")
        report.append(f"  Warnings Found: {len(result.warnings)}")
        report.append(f"  Duplicate Skills Found: {result.stats['duplicate_skills']}")
        report.append(f"  Circular Dependencies: {len(result.stats['circular_dependencies'])}")

        if result.errors:
            report.append("\nErrors Details:")
            for error in result.errors:
                report.append(f"  - {error}")

        if result.warnings:
            report.append("\nWarnings Details:")
            for warning in result.warnings:
                report.append(f"  - {warning}")

        if result.stats['missing_fields']:
            report.append("\nMissing Fields Breakdown:")
            for field, count in result.stats['missing_fields'].items():
                report.append(f"  - '{field}': {count} records missing")

        if result.stats['invalid_ranges']:
            report.append("\nInvalid Numerical Ranges Breakdown:")
            for field, count in result.stats['invalid_ranges'].items():
                report.append(f"  - '{field}': {count} records out of range")

        if result.stats['circular_dependencies']:
            report.append("\nCircular Dependencies:")
            for cycle in result.stats['circular_dependencies']:
                report.append(f"   • {' -> '.join(cycle)}")

        if result.stats['trend_categories']:
            report.append("\nMarket Trend Categories Breakdown:")
            for category, count in result.stats['trend_categories'].items():
                report.append(f"  - '{category}': {count} records")

        if result.stats['skill_type_categories']:
            report.append("\nSkill Type Categories Breakdown:")
            for category, count in result.stats['skill_type_categories'].items():
                report.append(f"  - '{category}': {count} records")

        if result.stats['category_groups']:
            report.append("\nSkill Category Groups Breakdown:")
            for group, count in result.stats['category_groups'].items():
                report.append(f"  - '{group}': {count} records")


        report.append("\n" + "=" * 60)

        return "\n".join(report)


def main():
    """Example usage of the validator"""
    data_file_path = Path("data/raw/all_skills.jsonl") # Adjust this path as needed

    if not data_file_path.exists():
        print(f"Error: Data file not found at {data_file_path}")
        print("Please place your JSONL data file in 'data/raw/'.")
        return

    sample_data = []
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
                continue

    validator = SkillDataValidator()
    result = validator.validate_dataset(sample_data)

    print(validator.generate_validation_report(result))

    # Test field normalization (these tests can remain as they don't depend on the dataset validation flow)
    trend_norm, trend_cat = validator.get_normalized_field_values('market_trend', 'Rapidly Increasing')
    print(f"\nNormalized Trend: {trend_norm}, Category: {trend_cat}")

    skill_type_norm, skill_type_cat = validator.get_normalized_field_values('skill_type', 'Programming Language')
    print(f"Normalized Skill Type: {skill_type_norm}, Category: {skill_type_cat}")

    category_norm, category_group = validator.get_normalized_field_values('category', 'Programming Languages')
    print(f"Normalized Category: {category_norm}, Group: {category_group}")

if __name__ == '__main__':
    main()