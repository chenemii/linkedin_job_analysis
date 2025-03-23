#!/usr/bin/env python3
# LinkedIn Job Posting Analysis Script
# This script analyzes LinkedIn job postings data and generates HTML and PNG visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import networkx as nx
from wordcloud import WordCloud
import os
import warnings
import json

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def main():
    # Download NLTK resources
    print("Downloading required NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

    # Load data
    print("Loading LinkedIn job posting data...")
    try:
        df = pd.read_csv('postings.csv')
        print(f"Loaded {len(df)} job postings from CSV")
    except FileNotFoundError:
        try:
            df = pd.read_json('linkedin_job_postings.json')
            print(f"Loaded {len(df)} job postings from JSON")
        except FileNotFoundError:
            print(
                "Error: No data file found. Please ensure 'linkedin_job_postings.csv' "
                "or 'linkedin_job_postings.json' exists in the current directory."
            )
            exit(1)

    # Basic data overview
    print("\n## Job Posting Overview")
    print(f"Number of job postings: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")

    # Data cleaning
    print("\n## Cleaning job descriptions")
    # Convert description to string and replace NaN with empty string
    df['description'] = df['description'].astype(str).replace('nan', '')
    df['description_clean'] = df['description'].str.lower()
    df['description_clean'] = df['description_clean'].apply(
        lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
    df['description_clean'] = df['description_clean'].apply(
        lambda x: re.sub(r'\s+', ' ', x).strip())

    # Process text
    print("\n## Processing job descriptions and extracting keywords")
    df['tokens'] = df['description_clean'].apply(preprocess_text)
    df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))

    # Analyze job levels
    analyze_job_levels(df)

    # Analyze skills
    analyze_skills(df)

    # Analyze salaries
    analyze_salaries(df)

    # Analyze organization structures
    analyze_org_structures(df)

    # Generate HTML report
    generate_html_report()

    print("\nAnalysis complete! Check the generated files:")
    print(
        "- PNG visualizations: job_levels.png, job_terms_wordcloud.png, salary_by_level.png"
    )
    print("- HTML report: visualization_results.html")


def tokenize_simple(text):
    """Simple tokenization by splitting on spaces and removing non-alphanumeric characters"""
    return [
        token.strip()
        for token in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    ]


def preprocess_text(text):
    """Preprocess text: tokenize, remove stopwords, and lemmatize"""
    # Tokenize with simple approach
    tokens = tokenize_simple(text)

    # Get English stopwords
    stop_words = set(stopwords.words('english'))

    # Add custom stopwords for job postings
    custom_stopwords = {
        'experience', 'job', 'work', 'team', 'company', 'position', 'required',
        'skills', 'ability', 'requirements', 'qualifications', 'years', 'year',
        'must', 'applicant', 'candidate', 'opportunity', 'please', 'apply',
        'looking', 'role', 'include', 'including', 'etc', 'day', 'week', 'time'
    }
    stop_words.update(custom_stopwords)

    # Filter stopwords and short words
    tokens = [
        word for word in tokens
        if word.lower() not in stop_words and len(word) > 2
    ]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def analyze_job_levels(df):
    """Analyze and visualize job level distribution"""
    print("\n## Analyzing Job Levels")

    # Check which job level column exists
    job_level_col = None
    if 'job_level' in df.columns:
        job_level_col = 'job_level'
    elif 'formatted_experience_level' in df.columns:
        job_level_col = 'formatted_experience_level'

    if job_level_col is None:
        print("No job level column found. Skipping job level analysis.")
        return

    # Map experience levels to standardized format if needed
    if job_level_col == 'formatted_experience_level':
        # Create mapping dictionary based on typical LinkedIn values
        level_mapping = {
            'ENTRY_LEVEL': 'Entry',
            'ASSOCIATE': 'Entry',
            'MID_SENIOR': 'Mid',
            'MID_LEVEL': 'Mid',
            'SENIOR': 'Senior',
            'SENIOR_LEVEL': 'Senior',
            'DIRECTOR': 'Senior',
            'EXECUTIVE': 'Senior'
        }

        # Apply mapping, keeping original value if not in mapping
        df['job_level'] = df[job_level_col].apply(lambda x: level_mapping.get(
            str(x).upper(), str(x)) if pd.notna(x) else 'Unknown')
        job_level_col = 'job_level'

    # Filter out any missing values
    df_with_level = df[df[job_level_col].notna()]

    job_level_counts = df_with_level[job_level_col].value_counts()
    print("Job Level Distribution:")
    print(job_level_counts)

    # Determine order based on available categories
    standard_order = ['Entry', 'Mid', 'Senior']
    available_order = [
        level for level in standard_order if level in job_level_counts.index
    ]

    if not available_order:
        available_order = job_level_counts.index

    # Visualize job level distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=job_level_col,
                       data=df_with_level,
                       order=available_order)
    plt.title('Distribution of Job Levels')
    plt.xlabel('Job Level')
    plt.ylabel('Count')

    # Add percentage labels
    total = len(df_with_level)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + 5
        ax.annotate(percentage, (x, y), ha='center')

    plt.tight_layout()
    plt.savefig('job_levels.png')
    print("Job level distribution saved to 'job_levels.png'")


def get_skill_patterns():
    """Return a dictionary of skill patterns and their normalized names"""
    return {
        # Programming Languages
        r'\b(python|python programming|python developer)\b':
        'Python',
        r'\b(java|java programming|java developer)\b':
        'Java',
        r'\b(javascript|js|node\.js|nodejs|typescript|react|angular|vue\.js)\b':
        'JavaScript/Web',
        r'\b(c\+\+|cpp|c plus plus)\b':
        'C++',
        r'\b(c#|csharp|\.net|dotnet)\b':
        'C#/.NET',

        # Data Science & Analytics
        r'\b(sql|mysql|postgresql|oracle db|database)\b':
        'SQL/Database',
        r'\b(machine learning|ml|deep learning|dl|ai|artificial intelligence)\b':
        'Machine Learning/AI',
        r'\b(data science|data analysis|data analytics|statistics|statistical analysis)\b':
        'Data Science',
        r'\b(power bi|tableau|data visualization)\b':
        'Data Visualization',
        r'\b(excel|microsoft excel|spreadsheet)\b':
        'Excel',

        # Cloud & DevOps
        r'\b(aws|amazon web services)\b':
        'AWS',
        r'\b(azure|microsoft azure)\b':
        'Azure',
        r'\b(gcp|google cloud)\b':
        'Google Cloud',
        r'\b(docker|kubernetes|k8s|container)\b':
        'Containers',
        r'\b(ci/cd|jenkins|gitlab|github actions)\b':
        'CI/CD',

        # Software Development
        r'\b(git|github|version control)\b':
        'Git',
        r'\b(api|rest api|restful|web services)\b':
        'API Development',
        r'\b(agile|scrum|kanban)\b':
        'Agile/Scrum',
        r'\b(unit testing|test automation|selenium|pytest)\b':
        'Testing',

        # Web Development
        r'\b(html|css|web development)\b':
        'HTML/CSS',
        r'\b(react|reactjs|react\.js)\b':
        'React',
        r'\b(angular|angularjs)\b':
        'Angular',
        r'\b(node|nodejs|node\.js)\b':
        'Node.js',

        # Business & Professional
        r'\b(project management|program management)\b':
        'Project Management',
        r'\b(business analysis|business intelligence)\b':
        'Business Analysis',
        r'\b(leadership|team lead|team leadership)\b':
        'Leadership',
        r'\b(communication|interpersonal|presentation)\b':
        'Communication',

        # Design
        r'\b(ui/ux|user interface|user experience)\b':
        'UI/UX Design',
        r'\b(figma|sketch|adobe xd)\b':
        'Design Tools',

        # Marketing & Analytics
        r'\b(seo|search engine optimization)\b':
        'SEO',
        r'\b(google analytics|web analytics)\b':
        'Analytics',
        r'\b(digital marketing|content marketing|social media marketing)\b':
        'Digital Marketing',

        # Project Tools
        r'\b(jira|confluence|atlassian)\b':
        'Project Tools',
        r'\b(slack|teams|collaboration tools)\b':
        'Collaboration Tools'
    }


def extract_skills_from_text(text, skill_patterns):
    """Extract skills from text using regex patterns"""
    if pd.isna(text):
        return []

    text = str(text).lower()
    found_skills = set()

    for pattern, skill_name in skill_patterns.items():
        if re.search(pattern, text):
            found_skills.add(skill_name)

    return list(found_skills)


def analyze_skills(df):
    """Analyze and visualize skills distribution"""
    print("\n## Analyzing Skills in Job Descriptions")

    # Get skill patterns
    skill_patterns = get_skill_patterns()

    print(
        "\nExtracting skills from job descriptions and skills descriptions...")
    # Extract skills from both description and skills_desc columns
    df['skills_from_desc'] = df['description'].apply(
        lambda x: extract_skills_from_text(x, skill_patterns))
    df['skills_from_skills_desc'] = df['skills_desc'].apply(
        lambda x: extract_skills_from_text(x, skill_patterns))

    # Combine skills from both sources
    df['all_skills'] = df.apply(lambda row: list(
        set(row['skills_from_desc'] + row['skills_from_skills_desc'])),
                                axis=1)

    # Count total skills found
    all_skills = [
        skill for skills_list in df['all_skills'] for skill in skills_list
    ]
    skill_counts = Counter(all_skills)

    print(
        f"\nFound {len(skill_counts):,} unique skills across all job postings")
    print("\nTop 20 Most Common Skills:")
    for skill, count in skill_counts.most_common(20):
        percentage = (count / len(df)) * 100
        print(f"{skill}: {count:,} postings ({percentage:.1f}%)")

    # Create visualization of top skills
    plt.figure(figsize=(15, 8))
    top_n = 20
    skills_df = pd.DataFrame(skill_counts.most_common(top_n),
                             columns=['Skill', 'Count'])

    # Create bar plot
    sns.barplot(data=skills_df, x='Count', y='Skill', palette='viridis')
    plt.title(f'Top {top_n} Most In-Demand Skills', pad=20)
    plt.xlabel('Number of Job Postings')

    # Add percentage labels
    total_jobs = len(df)
    for i, v in enumerate(skills_df['Count']):
        percentage = (v / total_jobs) * 100
        plt.text(v, i, f' {percentage:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig('top_skills.png', dpi=300, bbox_inches='tight')
    print("\nSkills visualization saved as 'top_skills.png'")

    # Save detailed results to JSON
    results = {
        'total_job_postings': len(df),
        'total_unique_skills': len(skill_counts),
        'skill_frequencies': {
            skill: {
                'count': count,
                'percentage': (count / len(df)) * 100
            }
            for skill, count in skill_counts.items()
        },
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }

    with open('skills_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to 'skills_analysis_results.json'")


def analyze_salaries(df):
    """Analyze and visualize salary distribution"""
    print("\n## Analyzing Salary Distribution")

    salary_columns = {
        'min': ['min_salary'],
        'max': ['max_salary'],
        'med': ['med_salary', 'normalized_salary']
    }

    # Check if we have salary columns
    has_min_max = all(
        any(col in df.columns for col in salary_columns[k])
        for k in ['min', 'max'])
    has_med = any(col in df.columns for col in salary_columns['med'])

    if not (has_min_max or has_med):
        print("No salary columns found. Skipping salary analysis.")
        return

    # Create a copy to avoid modifying the original dataframe
    salary_df = df.copy()

    # Add avg_salary column
    if has_min_max:
        # Find which min column exists
        min_col = next(
            (col for col in salary_columns['min'] if col in df.columns), None)
        max_col = next(
            (col for col in salary_columns['max'] if col in df.columns), None)

        # Convert to numeric, coercing errors to NaN
        salary_df[min_col] = pd.to_numeric(salary_df[min_col], errors='coerce')
        salary_df[max_col] = pd.to_numeric(salary_df[max_col], errors='coerce')

        # Calculate average salary
        salary_df['avg_salary'] = (salary_df[min_col] + salary_df[max_col]) / 2
    elif has_med:
        # Use median salary
        med_col = next(
            (col for col in salary_columns['med'] if col in df.columns), None)
        salary_df['avg_salary'] = pd.to_numeric(salary_df[med_col],
                                                errors='coerce')

    # Drop rows with NaN salaries
    salary_df = salary_df[salary_df['avg_salary'].notna()]

    if len(salary_df) == 0:
        print("No valid salary data found. Skipping salary analysis.")
        return

    # Determine job level column
    job_level_col = 'job_level'
    if job_level_col not in salary_df.columns and 'formatted_experience_level' in salary_df.columns:
        # Create job_level column if it doesn't exist
        level_mapping = {
            'ENTRY_LEVEL': 'Entry',
            'ASSOCIATE': 'Entry',
            'MID_SENIOR': 'Mid',
            'MID_LEVEL': 'Mid',
            'SENIOR': 'Senior',
            'SENIOR_LEVEL': 'Senior',
            'DIRECTOR': 'Senior',
            'EXECUTIVE': 'Senior'
        }

        salary_df['job_level'] = salary_df['formatted_experience_level'].apply(
            lambda x: level_mapping.get(str(x).upper(), str(x))
            if pd.notna(x) else 'Unknown')

    if job_level_col not in salary_df.columns:
        print(
            "No job level column found for salary analysis. Using overall salary distribution."
        )
        # Display overall salary statistics
        salary_stats = salary_df['avg_salary'].agg(
            ['mean', 'median', 'min', 'max'])
        print("\nOverall Salary Statistics:")
        print(salary_stats)

        # Visualize overall salary distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(salary_df['avg_salary'], kde=True, bins=30)
        plt.title('Overall Salary Distribution')
        plt.xlabel('Salary')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('salary_distribution.png')
        print("Overall salary distribution saved to 'salary_distribution.png'")
        return

    # Filter for rows that have both salary and job level data
    valid_df = salary_df[salary_df[job_level_col].notna()]
    if len(valid_df) == 0:
        print(
            "No valid data with both salary and job level. Skipping job level salary analysis."
        )
        return

    # Display salary statistics by job level
    standard_levels = ['Entry', 'Mid', 'Senior']
    # Check which standard levels are present in the data
    present_levels = [
        level for level in standard_levels
        if level in valid_df[job_level_col].unique()
    ]

    # Use present levels if available, otherwise use all unique levels
    if not present_levels:
        present_levels = valid_df[job_level_col].unique()

    # Filter dataframe to only include the levels we want to analyze
    valid_df = valid_df[valid_df[job_level_col].isin(present_levels)]

    # Calculate statistics
    salary_stats = valid_df.groupby(job_level_col)['avg_salary'].agg(
        ['mean', 'median', 'min', 'max'])
    print("\nSalary Statistics by Job Level:")
    print(salary_stats)

    # Visualize salary distribution by job level
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=job_level_col,
                y='avg_salary',
                data=valid_df,
                order=present_levels)
    plt.title('Salary Distribution by Job Level')
    plt.xlabel('Job Level')
    plt.ylabel('Salary ($)')
    plt.tight_layout()
    plt.savefig('salary_by_level.png')
    print("Salary distribution chart saved to 'salary_by_level.png'")


def analyze_org_structures(df):
    """Analyze and classify organization structures in job descriptions"""
    print("\n## Analyzing Organization Structures in Job Descriptions")

    # Define indicator terms for different organization structures
    flat_indicators = [
        'collaborative', 'cross-functional', 'flat structure',
        'open communication', 'self-managed', 'autonomous', 'agile',
        'self-organizing', 'team-based', 'decentralized', 'horizontal',
        'peer collaboration', 'empowered teams', 'minimal hierarchy',
        'non-hierarchical', 'collective decision', 'informal',
        'flexible structure', 'distributed leadership', 'shared responsibility'
    ]

    hierarchical_indicators = [
        'reports to', 'supervision', 'hierarchy', 'managerial oversight',
        'layered', 'chain of command', 'direct report', 'supervisor',
        'management chain', 'organizational tiers', 'senior management',
        'middle management', 'reporting line', 'executive oversight',
        'vertical structure', 'escalation path', 'approval process',
        'subordinate', 'org chart', 'structured management'
    ]

    print("Classifying organization structures in job descriptions...")

    # Check for and handle skills column
    skills_col = None
    if 'required_skills' in df.columns:
        skills_col = 'required_skills'
        # Convert to string and handle NaN
        df[skills_col] = df[skills_col].astype(str).replace('nan', '')
    elif 'skills_desc' in df.columns:
        skills_col = 'skills_desc'
        # Convert to string and handle NaN
        df[skills_col] = df[skills_col].astype(str).replace('nan', '')

    # Apply classification to job postings
    if skills_col:
        df['Org_Structure'] = df.apply(lambda row: classify_org_structure(
            row['description'], row[skills_col]),
                                       axis=1)
    else:
        df['Org_Structure'] = df['description'].apply(classify_org_structure)

    # Display organization structure distribution
    org_structure_counts = df['Org_Structure'].value_counts()
    print("\nOrganization Structure Distribution:")
    print(org_structure_counts)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y='Org_Structure',
                       data=df,
                       order=org_structure_counts.index)
    plt.title('Distribution of Organization Structures')
    plt.xlabel('Count')
    plt.ylabel('Organization Structure')

    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_width() / total:.1f}%'
        x = p.get_width() + 5
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y))

    plt.tight_layout()
    plt.savefig('org_structures.png')
    print("Organization structure distribution saved to 'org_structures.png'")

    # Create a visualization of job level by organization structure if job_level exists
    if 'job_level' in df.columns:
        plt.figure(figsize=(12, 7))
        crosstab = pd.crosstab(df['Org_Structure'], df['job_level'])
        crosstab.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Job Levels by Organization Structure')
        plt.xlabel('Organization Structure')
        plt.ylabel('Count')
        plt.legend(title='Job Level')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('org_structure_by_level.png')
        print(
            "Organization structure by job level chart saved to 'org_structure_by_level.png'"
        )

        # Show percentage breakdown
        print("\nOrganization Structure by Job Level (%):")
        org_by_level = pd.crosstab(
            df['job_level'], df['Org_Structure'], normalize='index') * 100
        print(org_by_level.round(1))

    # Extract common keywords by organization structure
    print("\nExtracting characteristic words by organization structure...")
    for structure in df['Org_Structure'].unique():
        if structure == 'Unknown':
            continue

        structure_desc = df[df['Org_Structure'] ==
                            structure]['description'].tolist()
        if structure_desc:
            common_words = get_common_words(structure_desc)
            print(f"\nTop words in {structure} organization job descriptions:")
            for word, count in common_words:
                print(f"  {word}: {count}")


def classify_org_structure(description, skills=''):
    """
    Classify organizational structure based on text indicators in the job description
    and skills description.
    
    Args:
        description (str): Job description text
        skills (str): Skills description or requirements text
    
    Returns:
        str: Classified organizational structure (Flat, Hierarchical, or Unknown)
    """
    # Ensure inputs are strings
    description = str(description) if description is not None else ''
    skills = str(skills) if skills is not None else ''

    # Replace 'nan' string with empty string
    if description.lower() == 'nan':
        description = ''
    if skills.lower() == 'nan':
        skills = ''

    # Define indicator terms for different organization structures
    flat_indicators = [
        'collaborative', 'cross-functional', 'flat structure',
        'open communication', 'self-managed', 'autonomous', 'agile',
        'self-organizing', 'team-based', 'decentralized', 'horizontal',
        'peer collaboration', 'empowered teams', 'minimal hierarchy',
        'non-hierarchical', 'collective decision', 'informal',
        'flexible structure', 'distributed leadership', 'shared responsibility'
    ]

    hierarchical_indicators = [
        'reports to', 'supervision', 'hierarchy', 'managerial oversight',
        'layered', 'chain of command', 'direct report', 'supervisor',
        'management chain', 'organizational tiers', 'senior management',
        'middle management', 'reporting line', 'executive oversight',
        'vertical structure', 'escalation path', 'approval process',
        'subordinate', 'org chart', 'structured management'
    ]

    # Combine description and skills, convert to lowercase for analysis
    full_text = (description + ' ' + skills).lower()

    # Count indicators for each type of structure
    flat_count = sum(1 for indicator in flat_indicators
                     if indicator.lower() in full_text)
    hierarchical_count = sum(1 for indicator in hierarchical_indicators
                             if indicator.lower() in full_text)

    # Apply classification rules
    if flat_count > hierarchical_count and flat_count > 0:
        return 'Flat'
    elif hierarchical_count > flat_count and hierarchical_count > 0:
        return 'Hierarchical'
    elif flat_count == hierarchical_count and flat_count > 0:
        return 'Hybrid'  # Equal number of indicators for both structures
    else:
        # Check for specific reporting phrases using regex
        if re.search(r'report(?:s|ing)?\s+(?:to|directly)', full_text):
            return 'Hierarchical'  # Default to hierarchical if reporting relationship is mentioned
        return 'Hybrid'  # Changed from Unknown to Hybrid since many job postings seem to get categorized as Hybrid


def get_common_words(descriptions, top_n=15):
    """Get most common words from a list of descriptions without using nltk tokenizers"""
    # Combine all texts
    text = ' '.join(descriptions).lower()

    # Simple tokenization using regex to split on non-alphanumeric characters
    words = re.findall(r'\b[a-z]{3,}\b', text)

    # Remove stopwords
    # Use a simple list of common English stopwords instead of nltk
    stopwords_simple = {
        'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'it', 'with', 'for',
        'as', 'be', 'on', 'not', 'this', 'but', 'by', 'at', 'are', 'an',
        'from', 'or', 'have', 'you', 'will', 'can', 'your', 'we', 'our',
        'their', 'has', 'been', 'all', 'which', 'they', 'one', 'more', 'was',
        'who', 'would', 'about', 'may', 'should', 'could', 'when', 'what',
        'than', 'other', 'how', 'its', 'also', 'some', 'such', 'only', 'new',
        'very', 'any', 'these', 'his', 'her', 'then', 'there', 'were', 'into',
        'them', 'out', 'just', 'if', 'so', 'up', 'no', 'my', 'him', 'me', 'us',
        'do'
    }

    # Add job posting common words to filter out
    job_stopwords = {
        'experience', 'job', 'work', 'team', 'company', 'position', 'required',
        'skills', 'ability', 'requirements', 'qualifications', 'years', 'year',
        'must', 'applicant', 'candidate', 'opportunity', 'please', 'apply',
        'looking', 'role', 'include', 'including', 'etc', 'day', 'week', 'time'
    }

    stopwords_simple.update(job_stopwords)

    # Filter out stopwords
    words = [word for word in words if word not in stopwords_simple]

    # Count and return top words
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(),
                          key=lambda x: x[1],
                          reverse=True)
    return sorted_words[:top_n]


def generate_html_report():
    """Generate HTML report with visualizations"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LinkedIn Job Posting Analysis Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2 {
            color: #0077b5; /* LinkedIn color */
        }
        .visualization {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        p {
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <h1>LinkedIn Job Posting Analysis Results</h1>
    
    <p>This page displays the visualizations generated from analyzing LinkedIn job posting data.</p>
    
    <div class="visualization">
        <h2>Job Level Distribution</h2>
        <p>
            This chart shows the distribution of job levels (Entry, Mid, and Senior) 
            in the analyzed job postings.
        </p>
        <img src="job_levels.png" alt="Job Level Distribution">
    </div>
    
    <div class="visualization">
        <h2>Common Terms in Job Descriptions</h2>
        <p>
            This word cloud visualization displays the most common terms found in job descriptions. 
            The size of each word represents its frequency.
        </p>
        <img src="job_terms_wordcloud.png" alt="Word Cloud of Job Description Terms">
    </div>
    
    <div class="visualization">
        <h2>Salary Distribution by Job Level</h2>
        <p>
            This box plot shows the distribution of salaries across different job levels.
            The box represents the middle 50% of salaries, with the line inside showing the median.
        </p>
        <img src="salary_by_level.png" alt="Salary Distribution by Job Level">
    </div>
    """

    # Add organization structure visualizations if they were generated
    if os.path.exists("org_structures.png"):
        html_content += """
    <div class="visualization">
        <h2>Organization Structure Distribution</h2>
        <p>
            This chart shows the distribution of organization structures identified in the job descriptions.
        </p>
        <img src="org_structures.png" alt="Organization Structure Distribution">
    </div>
    """

    if os.path.exists("org_structure_by_level.png"):
        html_content += """
    <div class="visualization">
        <h2>Job Levels by Organization Structure</h2>
        <p>
            This chart shows the distribution of job levels within each organization structure type.
        </p>
        <img src="org_structure_by_level.png" alt="Job Levels by Organization Structure">
    </div>
    """

    # Add top skills visualization if it was generated
    if os.path.exists("top_skills.png"):
        html_content += """
    <div class="visualization">
        <h2>Top Skills in Job Postings</h2>
        <p>
            This chart shows the most frequently mentioned skills found in the job postings.
        </p>
        <img src="top_skills.png" alt="Top Skills">
    </div>
    """

    # Add overall salary distribution if it was generated instead of by-level
    if os.path.exists("salary_distribution.png"
                      ) and not os.path.exists("salary_by_level.png"):
        html_content += """
    <div class="visualization">
        <h2>Overall Salary Distribution</h2>
        <p>
            This histogram shows the distribution of salaries across all job postings.
        </p>
        <img src="salary_distribution.png" alt="Overall Salary Distribution">
    </div>
    """

    # Close the HTML file
    html_content += """
    <footer>
        <p>
            Generated using LinkedIn Job Posting Analysis Tool on """ + pd.Timestamp.now(
    ).strftime("%Y-%m-%d") + """
        </p>
    </footer>
</body>
</html>
"""

    # Write HTML content to file
    with open("visualization_results.html", "w") as f:
        f.write(html_content)

    print("HTML report saved to 'visualization_results.html'")


if __name__ == "__main__":
    main()
