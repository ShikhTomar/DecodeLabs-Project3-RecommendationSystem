import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Welcome to AI Recommendation System (Tech Stack Recommender)")

# ---- STEP 1: INGESTION (Input) ----
# Requirement: minimum 3 user inputs
skill1 = input("Enter skill 1: ").lower().strip()
skill2 = input("Enter skill 2: ").lower().strip()
skill3 = input("Enter skill 3: ").lower().strip()

user_profile = f"{skill1} {skill2} {skill3}"

# ---- STEP 2: LOAD DATASET ----
data = pd.read_csv("raw_skills.csv")

# ---- STEP 3: VECTOR MAPPING (TF-IDF) ----
# We combine user profile + all job role skills into one corpus
# so they share the same vocabulary space
corpus = list(data["skills"]) + [user_profile]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Last row of tfidf_matrix = user profile vector
user_vector = tfidf_matrix[-1]
job_vectors = tfidf_matrix[:-1]

# ---- STEP 4: SCORING (Cosine Similarity) ----
scores = cosine_similarity(user_vector, job_vectors).flatten()

data["match_score"] = scores

# ---- STEP 5: SORTING + FILTERING (Top-N) ----
top_matches = data.sort_values(by="match_score", ascending=False).head(3)

print("\n🎯 Top 3 Recommended Career Paths:\n")
for i, row in top_matches.iterrows():
    match_percent = row['match_score'] * 100
    print(f"👉 {row['role']}  —  Match: {match_percent:.1f}%")