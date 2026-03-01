from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import OrdinalEncoder
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import numpy as np
import joblib
import math
import re
import torch
import seaborn as sns
import spacy
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from sklearn.metrics.pairwise import cosine_similarity

##--Loading---##
nlp = spacy.load("en_core_web_lg")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

filename = 'models/resume_rf_title_classifier.joblib'
rf_title_classifier = joblib.load(filename)

filename = 'models/resume_label_encoder.pkl'
resume_label_encoder = joblib.load(filename)

filename = 'models/ordinal_encoder.pkl'
encoder = joblib.load(filename)

filename = 'models/job_listing_vectorizer.joblib'
vectorizer = joblib.load(filename)

filename = 'models/job_listing_rf_classifier.joblib'
job_listing_tf_classifier = joblib.load(filename)

filename = 'models/job_listing_label_encoder.joblib'
job_listing_label_encoder = joblib.load(filename)

##---Resume Processing---##

##---Get Vector For Word---##
def word_to_vector(word: str, nlp: spacy.lang.en.English) -> np.ndarray:
    token = nlp(word)
    if token.has_vector:
        return token.vector
    else:
        # fallback for unknown words
        return np.zeros(300)

##---Return Textual Information And Vectorize Words For Title Classification---##
def vectorize_resume(resume_dir: str, encoder: OrdinalEncoder=encoder, nlp: spacy.lang.en.English=nlp) -> (pd.DataFrame, list):
    resume_data = []
    with pdfplumber.open(f'{resume_dir}') as pdf:
        # Loop through each page
        for page in pdf.pages:
            # Store words in page
            words = page.extract_words(extra_attrs=["fontname", "size"])

            for w in words:
                # Store specific attributes from every word and char in the resume page
                resume_data.append({"word": w["text"],
                            "width": w["width"],
                            "height": w["height"],
                            "left_distance": w["x0"], 
                            "bottom_distance": w["bottom"], 
                            "right_distance": w["x1"], 
                            "top_distance": w["top"],
                            "fontname": w["fontname"],
                            "fontsize": w["size"]})
        # Store pages
        pages = pdf.pages

    # Transform dict to pandas dataframe
    resume_data = pd.DataFrame(resume_data)
    # Rename col
    resume_data['original_word'] = resume_data['word']
    # Add vector to single col for each word
    resume_data['word'] = resume_data['word'].apply(lambda x: word_to_vector(x, nlp))
    # Create matrix
    vector_matrix = np.vstack(resume_data['word'].values)
    # Turn matrix into a df
    vector_df = pd.DataFrame(vector_matrix, index=resume_data.index)
    # Rename col names
    vector_df.columns = [f"vec_{i}" for i in range(vector_df.shape[1])]
    # Remove word col
    meta_df = resume_data.drop(columns=['word'])
    # Concatenate data frames
    resume_data_vec = pd.concat([vector_df, meta_df], axis=1)

    # Encode fontnames into their respective numbers for classification
    resume_data_vec['fontname'] = encoder.transform(resume_data_vec['fontname'])
    return resume_data_vec, pages

def get_cluster_centers(df: pd.DataFrame, predicted: np.ndarray, original_words: pd.Series, show_cluster_centers: bool) -> (pd.DataFrame, dict):
    cluster_centers = {"EDUCATION": [], "EXPERIENCE": [], "SKILLS":[], "PERSONAL":[]}

    rows_to_drop = []
    for i, p in enumerate(predicted):
        if pd.notna(p):
            if show_cluster_centers == True:
                print(f'Cluster: {original_words[i]}')
            cluster_centers[p].append(((df.iloc[i]['left_distance'] + df.iloc[i]['right_distance']) / 2, 
                                       (df.iloc[i]['bottom_distance'] + df.iloc[i]['top_distance']) / 2))
            rows_to_drop.append(i)
            
    df.drop(index=rows_to_drop, inplace=True)
    
    return df, cluster_centers

def calculate_word_x_and_y_position(df: pd.DataFrame, left: pd.Series, right:pd.Series, up: pd.Series, down: pd.Series) -> pd.DataFrame:
    calculate_center_position = lambda a, b: (a + b) / 2
    
    df["X"] = calculate_center_position(left, right) 
    df["Y"] =  calculate_center_position(down, up) 
    
    return df

##---Combine close clusters to eachother of the same category---##
def combine_close_cluster_titles(clusters: dict, threshold: int=250) -> dict:
    new_cluster_centers = {"EDUCATION": [], "EXPERIENCE": [], "SKILLS":[], "PERSONAL":[], "UNKOWN":[]}
    
    clusters_to_skip = []
    for index, cluster_arr in clusters.items():
        # Check if more titles exist for this category
        if len(cluster_arr) > 1:
            for i in cluster_arr:
                closest_index_to_i_val = float('inf')
                closest_index_to_i_pos = None
                # Skip if i has already combined previously
                if i in clusters_to_skip:
                    continue
                else:
                    for j in cluster_arr:
                        # Prevent combining same cluster
                        if i == j:
                            continue
                        else:
                            # Calculate distance from title i and j
                            current_euclidean_distance = math.sqrt((i[0] - j[0])**2 + ((i[1] - j[1])*4.0)**2)
                            # Store results of closest title to title i
                            if current_euclidean_distance < closest_index_to_i_val:
                                closest_index_to_i_val = current_euclidean_distance
                                closest_index_to_i_pos = j
 
                if closest_index_to_i_val < threshold:
                    # Add new cluster 
                    new_cluster_centers[index].append(((i[0] + closest_index_to_i_pos[0]) / 2, (i[1] + closest_index_to_i_pos[1]) / 2))
                    # Prevent combining j in future loops
                    clusters_to_skip.append(closest_index_to_i_pos)
                else:
                    new_cluster_centers[index].append(i)
        else:
            new_cluster_centers[index] = cluster_arr
    return new_cluster_centers

def create_y_divider(df: pd.DataFrame, clusters: dict, detection_range: int=10):
    divider_y_position = 0.0
    valid_y_dividers = [float("inf"), 0.0]
    all_title_y_points = []

    # Assign the beginning of divider_y_position to the largest X of all clusters
    for points in clusters.values():
        for point in points:
            all_title_y_points.append(point[1])
            if point[1] > divider_y_position:
                divider_y_position = point[1]

    # Continue creating dividers until we reach the page Y end.
    while divider_y_position > 0:
        valid_position = True
        closest_word_dist = float("inf")

        # Get closest cluster to divider_y_position
        point = None
        closest_title_point_dist = float("inf")

        # If no title points left to divide, break out of loop
        if len(all_title_y_points) <= 0:
            break

        
        for title_point in all_title_y_points:
            if closest_title_point_dist > abs(divider_y_position - title_point):
                closest_title_point_dist = abs(divider_y_position - title_point)
                point = title_point
            
        if point > divider_y_position:
            for _, word in df.iterrows():
                if word["Y"] >= (divider_y_position - detection_range) and word["Y"] <= (divider_y_position + detection_range):
                    valid_position = False
                    break
                if closest_word_dist > abs(divider_y_position - word["Y"]):
                    closest_word_dist = abs(divider_y_position - word["Y"])
        else:
            valid_position = False
                            
        if valid_position == True:
            if min(all_title_y_points) > divider_y_position:
                valid_position = False
        
        if valid_position == True:
            if closest_word_dist < closest_title_point_dist:
                valid_position = False
        
        if valid_position == True:
            valid_y_dividers.append(divider_y_position)
            all_title_y_points.remove(point)
        
        divider_y_position -= detection_range
        
    valid_y_dividers.sort(reverse=True)
    return valid_y_dividers

def create_x_divider(df: pd.DataFrame, clusters: dict, valid_y_dividers: list[float], detection_range: int=10):
    divider_x_position = 0.0
    valid_x_dividers = []
    all_title_x_points = []
    
    for points in clusters.values():
        for point in points:
            all_title_x_points.append(point)
            if point[0] > divider_x_position:
                divider_x_position = point[0]
                
    while divider_x_position > 0:
        valid_position = True
        closest_word_dist = float("inf")

        # Get closest cluster to divider_x_position
        point = None
        closest_title_point_dist = float("inf")

        if len(all_title_x_points) <= 0:
            break

        for i, _ in enumerate(valid_y_dividers):
            if i == len(valid_y_dividers) - 1:
                break
            j = i + 1
            
            for title_point in all_title_x_points:
                x_dist_val = abs(divider_x_position - title_point[0])
                if closest_title_point_dist > x_dist_val and title_point[1] <= valid_y_dividers[i] and title_point[1] >= valid_y_dividers[j]:
                    closest_title_point_dist = x_dist_val
                    point = title_point

            if point == None:
                break

            if divider_x_position > point[0]:
                for _, word in df.iterrows():
                    if word["Y"] <= valid_y_dividers[i] and word["Y"] >= valid_y_dividers[j]:
                        if word["X"] >= (divider_x_position - detection_range) and word["X"] <= (divider_x_position + detection_range):
                            valid_position = False
                            break
                        if closest_word_dist > abs(divider_x_position - word["X"]):
                            closest_word_dist = abs(divider_x_position - word["X"])
            else: 
                valid_position = False
                
            if valid_position == True:
                valid_title_x_points = [x[0] for x in all_title_x_points if x[0] <= valid_y_dividers[i] and x[0] >= valid_y_dividers[j]]
                if len(valid_title_x_points) > 0:
                    if min(valid_title_x_points) > divider_x_position:
                        valid_position = False
                else: 
                    valid_position = False
            
            if valid_position == True:
                valid_x_dividers.append((divider_x_position, valid_y_dividers[i], valid_y_dividers[j]))
                all_title_x_points.remove(point)
                
        divider_x_position -= detection_range
    return valid_x_dividers

##---Cluster each word to their closest title by their y-axis and assigning it's id to respective word---##
def cluster_text(df: pd.DataFrame, clusters: dict, y_dividers: list[float], x_dividers: list[tuple]) -> pd.DataFrame:
    clustered_df = []

    for row in df.values:
        smallest_distance = float('inf')
        closest_cluster = None

        for value, cluster_arr in clusters.items():
            word_x = row[-2]
            word_y = row[-1]

            # Look up for divider
            y_dividers_above = [y for y in y_dividers if y > word_y]
            if len(y_dividers_above) > 0:
                closest_above_y_point = min(y_dividers_above)
                
            # Look down for divider
            y_dividers_below = [y for y in y_dividers if y < word_y]
            if len(y_dividers_below) > 0:
                closest_below_y_point = max(y_dividers_below)

            x_dividers_right = [x for x in x_dividers if x[0] > word_x and x[1] == closest_above_y_point and x[2] == closest_below_y_point]
            if len(x_dividers_right) > 0:
                closest_right_x_point = min(x_dividers_right, key=lambda t: t[0], default=None)
            else:
                closest_right_x_point = (float("inf"), float("inf"), 0)

            x_dividers_left = [x for x in x_dividers if x[0] < word_x and x[1] == closest_above_y_point and x[2] == closest_below_y_point]
            if len(x_dividers_left) > 0:
                closest_left_x_point = max(x_dividers_left, key=lambda t: t[0], default=None)
            else:
                closest_left_x_point = (0, float("inf"), 0)
                
            for cluster in cluster_arr:
                cluster_x = cluster[0]
                cluster_y = cluster[1]

                if (closest_below_y_point <= cluster_y <= closest_above_y_point
                    and closest_below_y_point <= word_y <= closest_above_y_point
                    and closest_left_x_point[0] <= cluster_x <= closest_right_x_point[0]
                    and closest_left_x_point[0] <= word_x <= closest_right_x_point[0]):
                    
                
                    # Check if word is below cluster
                    if cluster_y <= word_y:
                        # Calculate distance
                        current_euclidean_distance = math.sqrt(((word_x - cluster_x))**2 + ((word_y - cluster_y))**2)
                        # Store results of closest title to word
                        if current_euclidean_distance < smallest_distance:
                            smallest_distance = current_euclidean_distance
                            closest_cluster = value
                        
        # Append closest cluster for this word
        if closest_cluster == None:
            clustered_df.append("UNKOWN")
        else:
            clustered_df.append(closest_cluster)
        
    df['cluster'] = clustered_df
            
    return df

def segmentate(df: pd.DataFrame, clusters: dict) -> dict:
    segmented_sections = {cluster: '' for cluster in clusters}
    
    for _, row in df.iterrows():
        segmented_sections[row["cluster"]] += row["original_word"] + " "

    return segmented_sections

def sub_segmentate_skills(segmented_sections: dict) -> list:
    return [item.strip() for item in re.split(r"\s*[•✓;,|\n\t]\s*", segmented_sections)]

def sub_segmentate_experience(segmented_sections: dict) -> list:
    job_pattern = re.compile(r'\b\d{4}\s*-\s*(current|present|\d{4})\b', re.IGNORECASE)
    split_on_years = re.split(job_pattern, segmented_sections)
    return [re.split(r"\s*[•✓;|\n]\s*", item) for item in split_on_years][0]


##---Job Listing Processing---##

def chunk_text(text: str) -> list[str]: return re.split(r"[\n:;]|\.\s", text)

def clean_chunks(chunks: list[str]) -> list[str]:
    cleaned_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            cleaned_chunks.append(chunk)
        
    return cleaned_chunks

def vectorize_chunks(chunks: list[str], vectorizer: TfidfVectorizer()) -> pd.DataFrame:
    vectorized_chunks = vectorizer.transform([chunk for chunk in chunks])
    vectorized_chunks_df = pd.DataFrame(vectorized_chunks.toarray(), columns=vectorizer.get_feature_names_out())
    return vectorized_chunks_df

def classify_chunks(vectorized_chunks: pd.DataFrame, rf: RandomForestClassifier()) -> list[int]: return rf.predict(vectorized_chunks)

##---Embeddings and Comparison---##

# Function to get BERT embeddings
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def create_similarity_matrix(resume_embeddings: list, job_listing_embeddings: list):
    similarity_matrix = []
    for i in job_listing_embeddings:
        similarity_arr = []
        for j in resume_embeddings:
            similarity_arr.append(cosine_similarity([i], [j])[0][0])
        similarity_matrix.append(similarity_arr)
    return similarity_matrix

def best_match_aggregation(similarity_matrix: list) -> (float, float):
    best_matches = [max(arr) for arr in similarity_matrix]
    aggregated_final_score = sum(best_matches) / len(best_matches)
    return aggregated_final_score, best_matches

def compare_resume_and_job_listing(resume_embeddings: list, job_listing_embeddings: list, report: bool=True) -> (float, float, list[float], float, list[float], float, list[float]):
    ##---EXPERIENCE---##
    experience_matrix = create_similarity_matrix(resume_embeddings["EXPERIENCE"], job_listing_embeddings["EXPERIENCE"])
    aggregated_experience_final_score, best_experience_matches = best_match_aggregation(experience_matrix)
    if report == True:
        print("##---EXPERIENCE---##")
        print(f"Total: {aggregated_experience_final_score}")
        print(f"Overall: {best_experience_matches}")
        print("\n")

    ##---EDUCATION---##
    education_matrix = create_similarity_matrix(resume_embeddings["EDUCATION"], job_listing_embeddings["EDUCATION"])
    aggregated_education_final_score, best_education_matches = best_match_aggregation(education_matrix)
    if report == True:
        print("##---EDUCATION---##")
        print(f"Total: {aggregated_education_final_score}")
        print(f"Overall: {best_education_matches}")
        print("\n")

    ##---SKILLS---##
    skills_matrix = create_similarity_matrix(resume_embeddings["SKILLS"], job_listing_embeddings["SKILLS"])
    aggregated_skills_final_score, best_skills_matches = best_match_aggregation(skills_matrix)
    if report == True:
        print("##---SKILLS---##")
        print(f"Total: {aggregated_skills_final_score}")
        print(f"Overall: {best_skills_matches}")
        print("\n")

    overall_similarity = (aggregated_skills_final_score + aggregated_education_final_score + aggregated_experience_final_score) / 3
    if report == True:
        print("##---RESULT---##")
        print(f"Total Similarity: {overall_similarity}")
    
    return overall_similarity, aggregated_experience_final_score, best_experience_matches, aggregated_education_final_score, best_education_matches, aggregated_skills_final_score, best_skills_matches

def compare(resume: str, job_listing: str, progress_callback=None):
    total_steps = 8
    current_step = 0

    ##---Vectorizing Resume---##
    resume_df, pages = vectorize_resume(resume_dir=resume,
                                   encoder=encoder,
                                   nlp=nlp)
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Vectorizing Resume")
    
    ##---Assigning Resume Clusters---##
    y_pred = rf_title_classifier.predict(resume_df.iloc[:, :-1])

    y_pred_df = pd.DataFrame(y_pred, columns=['label'])
    decoded_labels = resume_label_encoder.inverse_transform(y_pred_df)['label'].values 

    resume_df, cluster_centers = get_cluster_centers(df=resume_df, 
                                       predicted=decoded_labels, 
                                       original_words=resume_df["original_word"], 
                                       show_cluster_centers=False)
    
    resume_df = calculate_word_x_and_y_position(df=resume_df, 
                                            left=resume_df['left_distance'], 
                                            right=resume_df['right_distance'], 
                                            down=resume_df['bottom_distance'], 
                                            up=resume_df['top_distance'])
    combined_clusters = combine_close_cluster_titles(clusters=cluster_centers)
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Assigning Resume Clusters")

    ##---Dividing Resume---##
    y_dividers = create_y_divider(df=resume_df, 
                                    clusters=combined_clusters)
    
    x_dividers = create_x_divider(df=resume_df, 
                                    clusters=combined_clusters,
                                    valid_y_dividers=y_dividers)
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Dividing Resume")
    
    ##---Clustering---##
    resume_df = cluster_text(df=resume_df,
                         clusters=combined_clusters,
                         y_dividers=y_dividers,
                        x_dividers=x_dividers)
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Clustering")
    
    ##---Cleaning and Finalizing---##
    resume_sections = segmentate(df=resume_df, clusters=combined_clusters)

    resume_sections["SKILLS"] = sub_segmentate_skills(segmented_sections=resume_sections["SKILLS"])
    resume_sections["EXPERIENCE"] = sub_segmentate_experience(segmented_sections=resume_sections["EXPERIENCE"])
    resume_sections["EXPERIENCE"]
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Finalizing Resume Content")

    ##---Vectorizing Job Listing---##
    chunks = chunk_text(text=job_listing)
    chunks = clean_chunks(chunks=chunks)
    job_listing_vectorized_chunks_df = vectorize_chunks(chunks=chunks, 
                                                    vectorizer=vectorizer)
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Vectorizing Job Listing Sentences")

    ##---Classification---##
    results = classify_chunks(vectorized_chunks=job_listing_vectorized_chunks_df, 
                          rf=job_listing_tf_classifier)
    
    job_listing_sections = {"EDUCATION": [], "EXPERIENCE": [], "SKILLS":[], "RESPONSIBILITIES":[], "NOISE":[]}

    decoded_labels = job_listing_label_encoder.inverse_transform(results.reshape(-1,1))
    zipped_classified_chunks = zip(chunks, decoded_labels)

    for i in zipped_classified_chunks:
        job_listing_sections[i[1][0]].append(i[0])

    resume_sections["EDUCATION"] = [resume_sections["EDUCATION"].strip()]
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Classifying Sentences")

    ##---Embedding---##
    embedded_resume_sections = {}
    for section, texts in resume_sections.items():
        embedded_resume_sections[section] = np.array([get_bert_embedding(text, tokenizer, model) 
                                            for text in texts]
        )

    embedded_job_listing_sections = {}
    for section, texts in job_listing_sections.items():
        embedded_job_listing_sections[section] = np.array([get_bert_embedding(text, tokenizer, model) 
                                            for text in texts]
        )
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps, "Completed Embedding")
    
    ##---Comparing Resume and Job Listing---##
    (overall_similarity, 
    aggregated_experience_final_score, best_experience_matches, 
    aggregated_education_final_score, best_education_matches, 
    aggregated_skills_final_score, best_skills_matches) = compare_resume_and_job_listing(resume_embeddings=embedded_resume_sections,
                                                   job_listing_embeddings=embedded_job_listing_sections,
                                                   report=False)

    return (overall_similarity, job_listing_sections,
            aggregated_experience_final_score, best_experience_matches, 
            aggregated_education_final_score, best_education_matches, 
            aggregated_skills_final_score, best_skills_matches)

if __name__ == "__main__":
    resume = "Danielle_Brasseur.pdf"
    job_listing = """
        Full job description
        Are you a Data professional who has recently transitioned into Data Modelling??
        Your new company
        A leading Victorian State Government entity dedicated to protecting the people, places, and projects that enable the community to thrive is seeking a Junior Data Modeller for an initial 6‑month contract.This organisation plays a critical role in risk management and public-sector resilience and is continuing to strengthen its data capability to support strategic, analytic, and operational outcomes.

        Your new role
        As the Junior Data Modeller, you will support data and digital initiatives by assisting in mapping how front‑end systems relate to underlying data structures. Your contributions will enhance data quality, improve reporting accuracy, and support enterprise data modelling efforts. Key responsibilities include:

        Analyse front‑end systems and workflows to understand how fields map to backend databases.
        Contribute to logical and conceptual data models describing entities, relationships, and key business rules.
        Support alignment of application‑level data structures with enterprise data models and reporting requirements.
        Develop and maintain documentation such as data dictionaries, mapping documents, metadata records, and simple ER‑style diagrams.
        Capture and update data lineage, ownership information, and metadata in agreed repositories.
        Assist with data profiling activities to validate mappings and ensure data quality.
        Support testing of integrations and reporting outputs to ensure fields and relationships populate correctly.
        Identify basic data issues or inconsistencies and escalate them for remediation.
        Contribute to compliance, governance, and risk processes by following controls and reporting risks where needed.
        Maintain version control of documentation and promote consistency across artefacts.

        What you'll need to succeed
        To thrive in this role, you will bring technical curiosity, strong attention to detail, and a passion for understanding how systems and data fit together. Essential skills & experience:
        Exposure to relational databases and SQL through study, projects, internships, or early work experience.
        Bachelor degree in Data Science.
        Basic understanding of how front‑end applications interact with back‑end databases or APIs.
        Strong analytical and problem‑solving skills, with the ability to break down ambiguous problems.
        Clear written communication skills and a detail‑oriented approach to documentation.
        Demonstrated interest in systems, data structures, and end‑to‑end data flows.
        Data analysis, Data engineering, Data modelling experience
        Exposure in business intelligence or reporting & software development
        Familiarity with data modelling concepts (entities, relationships, normalisation, dimensional models)
        Reporting/visualisation tools (e.g., Power BI)
        At least 2 years of experience

        What you'll get in return
        A valuable opportunity to gain hands‑on experience in an enterprise‑level data environment within the public sector.
        Exposure to senior data professionals and a supportive, collaborative team culture.
        The chance to develop your skills in data modelling, documentation, analytics, and systems mapping.
        A meaningful role contributing to public safety, resilience, and community outcomes.
        Competitive daily rate and hybrid working flexibility.

        What you need to do now
        If you're interested in this role, click 'apply now' to forward an up-to-date copy of your CV to Prachi.Kalyanarora@Hays.com.au, or call us now.
        If this job isn't quite right for you, but you are looking for a new position, please contact us for a confidential discussion on your career.
        """
    print(compare(resume=resume, job_listing=job_listing))