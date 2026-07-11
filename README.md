# Tech Stack Recommender

## Description
Tech Stack Recommender is an AI-based recommendation system that suggests the most suitable career paths based on a user's skills. It uses TF-IDF Vectorization and Cosine Similarity to match user-input skills against a dataset of job roles, returning the Top-3 best-matching career paths with their similarity scores.

The system takes 3 skills as input from the user, converts both the user profile and job role descriptions into numerical vectors, calculates how closely they align, and displays the highest-scoring matches.

## How to Run

1. Install the required libraries:
pip install pandas scikit-learn

2. Make sure raw_skills.csv is present in the same folder as recommend.py.

3. Run the script:
python recommend.py

4. Enter 3 skills when prompted:
Enter skill 1: python
Enter skill 2: machine learning
Enter skill 3: deep learning

5. View your Top-3 recommended career paths with match scores.
