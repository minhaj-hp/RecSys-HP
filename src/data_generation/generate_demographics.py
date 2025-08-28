import pandas as pd
import numpy as np
from typing import Dict, List
import os

class DemographicDataGenerator:
    """Generate realistic categorical demographic data correlating with existing age/income."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Define categorical mappings
        self.profession_categories = [
            "Technology", "Healthcare", "Education", "Finance", 
            "Retail", "Manufacturing", "Services", "Other"
        ]
        
        self.location_categories = ["Urban", "Suburban", "Rural"]
        
        self.education_categories = [
            "High School", "Some College", "Bachelor's", "Master's", "PhD+"
        ]
        
        self.marital_categories = ["Single", "Married", "Divorced", "Widowed"]
    
    def generate_profession(self, age: int, income: float, gender: str) -> str:
        """Generate profession based on age, income, and gender correlations."""
        
        # Age-based profession probabilities
        if age < 25:
            # Young adults - more likely in retail, services, some tech
            probs = [0.15, 0.10, 0.08, 0.05, 0.25, 0.10, 0.20, 0.07]
        elif age < 35:
            # Early career - tech, healthcare, finance growth
            probs = [0.25, 0.15, 0.10, 0.15, 0.12, 0.08, 0.10, 0.05]
        elif age < 50:
            # Mid career - established in all fields
            probs = [0.20, 0.18, 0.15, 0.18, 0.08, 0.12, 0.07, 0.02]
        else:
            # Senior career - more in education, healthcare, services
            probs = [0.15, 0.20, 0.20, 0.15, 0.05, 0.15, 0.08, 0.02]
        
        # Income adjustments
        if income > 90000:  # High income
            # Boost tech, finance, healthcare
            probs[0] *= 1.5  # Technology
            probs[3] *= 1.5  # Finance
            probs[1] *= 1.3  # Healthcare
            probs[4] *= 0.5  # Retail
            probs[6] *= 0.7  # Services
        elif income < 40000:  # Lower income
            # Boost retail, services, manufacturing
            probs[4] *= 2.0  # Retail
            probs[6] *= 1.8  # Services
            probs[5] *= 1.5  # Manufacturing
            probs[0] *= 0.3  # Technology
            probs[3] *= 0.3  # Finance
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        return np.random.choice(self.profession_categories, p=probs)
    
    def generate_location(self, income: float, profession: str) -> str:
        """Generate location based on income and profession."""
        
        # Base probabilities (roughly US distribution)
        probs = [0.62, 0.27, 0.11]  # Urban, Suburban, Rural
        
        # Income adjustments
        if income > 80000:
            # Higher income -> more suburban
            probs = [0.45, 0.45, 0.10]
        elif income < 35000:
            # Lower income -> more urban/rural
            probs = [0.70, 0.15, 0.15]
        
        # Profession adjustments
        if profession in ["Technology", "Finance"]:
            # Tech/Finance -> more urban
            probs[0] *= 1.4
            probs[2] *= 0.5
        elif profession in ["Manufacturing", "Other"]:
            # Manufacturing -> more rural/suburban
            probs[1] *= 1.3
            probs[2] *= 1.5
            probs[0] *= 0.7
        
        # Normalize
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        return np.random.choice(self.location_categories, p=probs)
    
    def generate_education_level(self, age: int, income: float, profession: str) -> str:
        """Generate education level based on age, income, and profession."""
        
        # Base probabilities (roughly US distribution)
        probs = [0.27, 0.20, 0.33, 0.13, 0.07]  # HS, Some College, Bachelor's, Master's, PhD+
        
        # Age adjustments (older generations had less college access)
        if age > 55:
            probs = [0.40, 0.25, 0.25, 0.08, 0.02]
        elif age > 40:
            probs = [0.32, 0.23, 0.30, 0.12, 0.03]
        elif age < 30:
            # Younger generation has more education
            probs = [0.20, 0.15, 0.40, 0.18, 0.07]
        
        # Income adjustments
        if income > 100000:
            # High income -> more advanced degrees
            probs = [0.10, 0.10, 0.35, 0.30, 0.15]
        elif income > 70000:
            # Good income -> more bachelor's/master's
            probs = [0.15, 0.15, 0.45, 0.20, 0.05]
        elif income < 40000:
            # Lower income -> less higher education
            probs = [0.45, 0.30, 0.20, 0.04, 0.01]
        
        # Profession adjustments
        if profession in ["Technology", "Healthcare", "Finance"]:
            # Professional fields -> more degrees
            probs = [0.05, 0.10, 0.40, 0.30, 0.15]
        elif profession == "Education":
            # Education -> even more advanced degrees
            probs = [0.02, 0.05, 0.25, 0.45, 0.23]
        elif profession in ["Retail", "Services", "Manufacturing"]:
            # Service industries -> less higher education
            probs = [0.40, 0.25, 0.25, 0.08, 0.02]
        
        # Normalize
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        return np.random.choice(self.education_categories, p=probs)
    
    def generate_marital_status(self, age: int, gender: str) -> str:
        """Generate marital status based on age and gender."""
        
        # Age-based probabilities
        if age < 25:
            probs = [0.85, 0.13, 0.02, 0.00]  # Single, Married, Divorced, Widowed
        elif age < 35:
            probs = [0.45, 0.50, 0.05, 0.00]
        elif age < 50:
            probs = [0.15, 0.70, 0.14, 0.01]
        elif age < 65:
            probs = [0.10, 0.65, 0.20, 0.05]
        else:
            probs = [0.08, 0.55, 0.15, 0.22]
        
        # Gender adjustments (women tend to be widowed more often in older ages)
        if age > 65 and gender == 'female':
            probs[3] *= 2.0  # More widowed women
            probs[1] *= 0.8  # Fewer married
        
        # Normalize
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
        return np.random.choice(self.marital_categories, p=probs)
    
    def generate_user_demographics(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Generate all demographic features for all users."""
        
        print(f"Generating demographic data for {len(users_df)} users...")
        
        # Create a copy to avoid modifying original
        enhanced_users = users_df.copy()
        
        # Generate each demographic feature
        professions = []
        locations = []
        education_levels = []
        marital_statuses = []
        
        for idx, row in users_df.iterrows():
            age = row['age']
            income = row['income']
            gender = row['gender']
            
            # Generate profession first as it influences other features
            profession = self.generate_profession(age, income, gender)
            professions.append(profession)
            
            # Generate location based on income and profession
            location = self.generate_location(income, profession)
            locations.append(location)
            
            # Generate education based on age, income, and profession
            education = self.generate_education_level(age, income, profession)
            education_levels.append(education)
            
            # Generate marital status based on age and gender
            marital_status = self.generate_marital_status(age, gender)
            marital_statuses.append(marital_status)
        
        # Add new columns
        enhanced_users['profession'] = professions
        enhanced_users['location'] = locations
        enhanced_users['education_level'] = education_levels
        enhanced_users['marital_status'] = marital_statuses
        
        return enhanced_users
    
    def print_demographic_statistics(self, users_df: pd.DataFrame):
        """Print statistics about the generated demographics."""
        
        print("\n=== Demographic Statistics ===")
        
        # Profession distribution
        print(f"\nProfession Distribution:")
        prof_counts = users_df['profession'].value_counts()
        for prof, count in prof_counts.items():
            pct = (count / len(users_df)) * 100
            print(f"  {prof}: {count:,} ({pct:.1f}%)")
        
        # Location distribution
        print(f"\nLocation Distribution:")
        loc_counts = users_df['location'].value_counts()
        for loc, count in loc_counts.items():
            pct = (count / len(users_df)) * 100
            print(f"  {loc}: {count:,} ({pct:.1f}%)")
        
        # Education distribution
        print(f"\nEducation Level Distribution:")
        edu_counts = users_df['education_level'].value_counts()
        for edu, count in edu_counts.items():
            pct = (count / len(users_df)) * 100
            print(f"  {edu}: {count:,} ({pct:.1f}%)")
        
        # Marital status distribution
        print(f"\nMarital Status Distribution:")
        marital_counts = users_df['marital_status'].value_counts()
        for status, count in marital_counts.items():
            pct = (count / len(users_df)) * 100
            print(f"  {status}: {count:,} ({pct:.1f}%)")
        
        print(f"\nTotal users: {len(users_df):,}")
        
        # Cross-tabulations to show correlations
        print(f"\n=== Key Correlations ===")
        
        # High income professions
        high_income = users_df[users_df['income'] > 80000]
        print(f"\nTop professions for high income (>${80000:,}+):")
        high_income_prof = high_income['profession'].value_counts(normalize=True) * 100
        for prof, pct in high_income_prof.head().items():
            print(f"  {prof}: {pct:.1f}%")
        
        # Education by profession
        print(f"\nEducation levels in Technology:")
        tech_edu = users_df[users_df['profession'] == 'Technology']['education_level'].value_counts(normalize=True) * 100
        for edu, pct in tech_edu.items():
            print(f"  {edu}: {pct:.1f}%")


def main():
    """Main function to generate and save enhanced demographic data."""
    
    # Load existing users data
    users_path = "datasets/users.csv"
    if not os.path.exists(users_path):
        print(f"Error: {users_path} not found!")
        return
    
    print(f"Loading users data from {users_path}")
    users_df = pd.read_csv(users_path)
    
    print(f"Original data shape: {users_df.shape}")
    print(f"Original columns: {list(users_df.columns)}")
    
    # Generate demographic data
    generator = DemographicDataGenerator(seed=42)
    enhanced_users = generator.generate_user_demographics(users_df)
    
    # Print statistics
    generator.print_demographic_statistics(enhanced_users)
    
    # Save enhanced data
    output_path = "datasets/users_enhanced.csv"
    enhanced_users.to_csv(output_path, index=False)
    print(f"\nEnhanced users data saved to {output_path}")
    
    print(f"Enhanced data shape: {enhanced_users.shape}")
    print(f"New columns: {list(enhanced_users.columns)}")


if __name__ == "__main__":
    main()