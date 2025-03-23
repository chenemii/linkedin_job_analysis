#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import json
from linkedin_job_analysis import get_skill_patterns, extract_skills_from_text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Skill categories and their patterns
SKILL_CATEGORIES = {
    'Programming Languages':
    r'python|java|javascript|c\+\+|ruby|php|swift|kotlin|rust|golang|scala|typescript',
    'Databases':
    r'sql|mysql|postgresql|mongodb|oracle|redis|cassandra|elasticsearch',
    'Cloud & DevOps':
    r'aws|azure|gcp|docker|kubernetes|terraform|jenkins|git',
    'AI & ML':
    r'machine learning|deep learning|nlp|computer vision|ai|artificial intelligence|data science',
    'Web Development':
    r'react|angular|vue|node\.js|django|flask|spring|laravel',
    'Project Management':
    r'agile|scrum|kanban|waterfall|lean|prince2|pmp',
    'Soft Skills':
    r'communication|leadership|teamwork|problem.solving|analytical|critical.thinking',
    'Business Tools':
    r'excel|powerpoint|word|tableau|power.bi|looker|qlik',
    'Marketing':
    r'marketing|seo|sem|social media|content marketing|email marketing',
    'Finance':
    r'financial analysis|accounting|budgeting|forecasting|risk management',
    'Sales':
    r'sales|customer service|crm|salesforce|hubspot',
    'HR':
    r'hr|recruitment|talent management|employee relations|benefits administration',
    'Data Analysis':
    r'data analysis|statistics|r|spss|stata|matlab',
    'Design':
    r'ui|ux|user experience|user interface|wireframing|prototyping|figma|sketch',
    'Security':
    r'cybersecurity|network security|penetration testing|encryption|firewall'
}


def load_sp500_companies():
    """Load S&P 500 companies from the CSV file."""
    try:
        df = pd.read_csv('sp500_companies.csv')
        # Create a set of normalized company names for matching
        company_names = set()
        for name in df['Company Name']:
            # Remove common company suffixes and normalize
            normalized = re.sub(
                r'\s*(Inc\.|Corporation|Corp\.|Company|Co\.|Ltd\.|Limited|LLC)\.?\s*$',
                '',
                name,
                flags=re.IGNORECASE)
            normalized = normalized.lower().strip()
            company_names.add(normalized)
        return company_names
    except Exception as e:
        print(f"Error loading S&P 500 companies: {str(e)}")
        return set()


def normalize_company_name(name):
    """Normalize company name for comparison."""
    if not isinstance(name, str):
        return ""
    # Remove common suffixes and convert to lowercase
    name = name.lower()
    suffixes = [
        ' inc', ' corp', ' corporation', ' company', ' co', ' ltd', ' limited',
        ' llc', '.com'
    ]
    for suffix in suffixes:
        name = name.replace(suffix, '')
    # Remove special characters and extra spaces
    name = re.sub(r'[^\w\s]', '', name)
    name = ' '.join(name.split())
    return name


def is_sp500_company(company_name, sp500_companies):
    """Check if a company is in the S&P 500 list."""
    if not isinstance(company_name, str):
        return False
    normalized_name = normalize_company_name(company_name)
    return any(
        normalize_company_name(sp500) in normalized_name
        or normalized_name in normalize_company_name(sp500)
        for sp500 in sp500_companies)


def extract_skills_by_category(text):
    """Extract skills from text and categorize them."""
    if pd.isna(text):
        return {}

    text = str(text).lower()
    skills_by_category = {category: [] for category in SKILL_CATEGORIES}

    for category, pattern in SKILL_CATEGORIES.items():
        matches = re.finditer(f'\\b({pattern})\\b', text)
        skills = [match.group() for match in matches]
        if skills:
            skills_by_category[category].extend(skills)

    return {
        k: list(set(v))
        for k, v in skills_by_category.items() if v
    }  # Remove duplicates


def create_category_visualization(sp500_data, other_data, category,
                                  output_file):
    """Create visualization for a specific skill category."""
    plt.figure(figsize=(10, 6))

    # Prepare data
    all_skills = set(sp500_data.keys()) | set(other_data.keys())
    skills = sorted(all_skills)
    sp500_values = [sp500_data.get(skill, 0) for skill in skills]
    other_values = [other_data.get(skill, 0) for skill in skills]

    # Create bar chart
    x = np.arange(len(skills))
    width = 0.35

    plt.bar(x - width / 2,
            sp500_values,
            width,
            label='S&P 500',
            color='#2ecc71')
    plt.bar(x + width / 2,
            other_values,
            width,
            label='Other Companies',
            color='#3498db')

    plt.xlabel('Skills')
    plt.ylabel('Percentage of Job Postings')
    plt.title(f'{category} Skills: S&P 500 vs Other Companies')
    plt.xticks(x, skills, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def analyze_org_structure(df):
    """
    Analyze organizational structure based on job description keywords, contextual clues,
    job titles, and industry patterns.

    This analysis is based on several academic and professional sources:
    1. Mintzberg's Organizational Structures (1979):
       - Identifies basic organizational configurations including simple, machine bureaucracy,
         professional bureaucracy, divisionalized form, and adhocracy.
    
    2. Modern Organization Designs (Galbraith, 2014):
       - Matrix organizations
       - Network structures
       - Process-based organizations
    
    3. Digital Era Structures (Deloitte, 2016):
       - Team-based organizations
       - Agile structures
       - Remote/hybrid models
    
    4. Industry Standards and Best Practices:
       - Project Management Institute (PMI) organizational structures
       - Agile Alliance team structures
       - McKinsey's organizational design principles

    Returns:
        tuple: (structure_counts, structure_percentages) containing the distribution
               of organizational structures in the analyzed job postings.
    """
    # Organizational structure types and their indicators
    # Based on research from:
    # - "Images of Organization" by Gareth Morgan
    # - "Organization Design" by Jay Galbraith
    # - "Designing Organizations" by Jay R. Galbraith
    structure_keywords = {
        'Hierarchical': [  # Traditional bureaucratic structure (Weber, 1947)
            'hierarchical', 'traditional structure', 'chain of command',
            'top-down', 'reporting structure', 'management hierarchy',
            'organizational hierarchy', 'direct report', 'supervisor',
            'middle management', 'senior management', 'executive leadership',
            'reporting to', 'reports to', 'vertical structure',
            'levels of management', 'reporting hierarchy', 'management layers',
            'command chain', 'corporate hierarchy', 'management structure',
            'organizational chart', 'reporting relationship',
            'management team', 'leadership team', 'executive team',
            'senior leadership'
        ],
        'Functional': [  # Functional grouping (Taylor's Scientific Management)
            'functional organization', 'functional structure',
            'specialized departments', 'department head', 'functional teams',
            'specialized teams', 'functional division',
            'departmental structure', 'specialized units',
            'functional expertise', 'department-based',
            'functional leadership', 'center of excellence',
            'specialized group', 'functional area', 'domain expertise',
            'technical department', 'business function', 'functional manager',
            'department director', 'functional head', 'specialized practice',
            'practice area', 'expertise center'
        ],
        'Matrix': [  # Matrix structure (Galbraith, 1971)
            'matrix structure', 'matrix organization', 'cross-functional',
            'dual reporting', 'multiple reporting lines',
            'project-based structure', 'dotted line reporting',
            'functional reporting', 'matrix management', 'project matrix',
            'resource matrix', 'cross-departmental', 'interdepartmental',
            'multi-disciplinary teams', 'dual authority', 'multiple managers',
            'project and functional managers', 'matrix environment',
            'dual responsibility', 'project organization',
            'resource allocation', 'shared resources', 'matrix team'
        ],
        'Flat/Horizontal': [  # Modern flat structures (Valve, Gore)
            'flat structure', 'flat organization', 'horizontal structure',
            'non-hierarchical', 'self-managed', 'autonomous teams',
            'decentralized', 'peer-to-peer', 'collaborative environment',
            'open door policy', 'minimal hierarchy', 'empowered teams',
            'self-organizing', 'holacracy', 'few management layers',
            'direct communication', 'minimal bureaucracy', 'open organization',
            'lean organization', 'flat management', 'organic structure',
            'informal structure', 'flexible organization'
        ],
        'Team-Based': [  # Agile/Scrum team structures (Sutherland, Schwaber)
            'team-based', 'agile teams', 'squad model', 'pod structure',
            'scrum teams', 'product teams', 'feature teams', 'dedicated teams',
            'small teams', 'autonomous teams', 'self-managed teams',
            'cross-functional teams', 'agile squads', 'team-oriented',
            'team structure', 'team organization', 'collaborative teams',
            'agile organization', 'sprint teams', 'project teams',
            'delivery teams', 'development teams', 'product squads',
            'team-driven', 'team-led', 'team-focused'
        ],
        'Network': [  # Network organizations (Miles & Snow, 1986)
            'network structure', 'networked organization',
            'distributed organization', 'alliance network', 'partner network',
            'network model', 'interconnected units', 'network of teams',
            'collaborative network', 'strategic alliances',
            'partner ecosystem', 'network-based', 'distributed teams',
            'networked teams', 'external partnerships', 'partnership model',
            'ecosystem approach', 'collaborative model', 'alliance structure',
            'partner-based'
        ],
        'Process-Based': [  # Process-centric (Hammer & Champy, 1993)
            'process-based', 'process-oriented', 'workflow-based',
            'process organization', 'process management', 'process teams',
            'workflow teams', 'process structure', 'process-driven',
            'workflow structure', 'process-centric', 'end-to-end process',
            'process flow', 'workflow management', 'business process',
            'process improvement', 'continuous flow', 'process excellence',
            'operational excellence', 'lean process', 'process optimization',
            'value stream', 'process chain'
        ],
        'Divisional': [  # Divisional structure (Chandler, 1962)
            'business unit', 'division structure', 'product division',
            'regional division', 'business division', 'divisional structure',
            'profit center', 'business segment', 'operating unit',
            'strategic business unit', 'subsidiary', 'business group',
            'market division', 'geographic division', 'product-based division',
            'regional structure', 'market-based unit', 'business line',
            'product group', 'division head', 'business segment',
            'regional organization'
        ],
        'Project-Based': [  # Project organization (PMI standards)
            'project-based organization', 'project structure',
            'project-oriented', 'project organization', 'project teams',
            'project management office', 'project portfolio', 'project-driven',
            'project leadership', 'project governance', 'project coordination',
            'project-centric', 'program-based', 'program structure',
            'project framework'
        ],
        'Remote/Hybrid': [  # Modern distributed work (Post-2020 models)
            'remote-first', 'distributed team', 'virtual organization',
            'hybrid work', 'remote work', 'work from home', 'flexible work',
            'distributed workforce', 'virtual team', 'remote-friendly',
            'hybrid workplace', 'telework', 'hybrid model',
            'flexible workplace', 'distributed organization',
            'remote operations', 'virtual workplace', 'digital workplace'
        ]
    }

    # Context patterns based on organizational behavior research
    # Sources: Organizational Behavior (Robbins & Judge)
    context_patterns = {
        'Hierarchical': [  # Based on traditional management theory
            r'report(?:s|ing)?\s+(?:to|directly)\s+(?:the)?\s*(?:senior|chief|vp|director|manager|head)',
            r'(?:clear|defined)\s+reporting\s+(?:structure|hierarchy|line)',
            r'management\s+(?:structure|hierarchy|chain)',
            r'(?:multiple|several)\s+levels?\s+of\s+management',
            r'(?:senior|executive)\s+leadership\s+team',
            r'(?:clear|established)\s+chain\s+of\s+command'
        ],
        'Functional': [  # Based on specialization principles
            r'specialized\s+(?:department|team|unit|group)',
            r'functional\s+(?:expertise|specialization|area)',
            r'department\s+head',
            r'specialized\s+(?:role|function|responsibility)',
            r'center\s+of\s+excellence',
            r'functional\s+(?:group|practice|division)'
        ],
        'Matrix': [  # Based on matrix management literature
            r'work(?:ing)?\s+across\s+(?:multiple|different)\s+teams',
            r'collaborate\s+with\s+(?:multiple|different|various)\s+(?:teams|departments|functions)',
            r'matrix\s+(?:environment|organization|structure)',
            r'dual\s+reporting\s+(?:structure|relationship)',
            r'report(?:ing)?\s+to\s+both', r'multiple\s+stakeholders?'
        ],
        'Team-Based': [  # Based on Agile/Scrum frameworks
            r'agile\s+(?:methodology|framework|environment)',
            r'scrum\s+(?:methodology|framework|team)',
            r'(?:small|autonomous|self-managed)\s+teams?',
            r'team-based\s+(?:approach|structure|organization)',
            r'sprint\s+planning', r'agile\s+(?:ceremonies|practices)'
        ],
        'Process-Based': [  # Based on BPM and Lean principles
            r'end-to-end\s+process',
            r'process\s+(?:flow|management|improvement)',
            r'workflow\s+(?:based|oriented|driven)',
            r'process\s+(?:owner|leader|manager)', r'continuous\s+improvement',
            r'lean\s+(?:methodology|principles)'
        ],
        'Network': [  # Based on network organization theory
            r'network\s+of\s+(?:teams|partners|organizations)',
            r'strategic\s+(?:alliance|partnership)',
            r'collaborative\s+network', r'partner\s+ecosystem',
            r'ecosystem\s+(?:approach|model)',
            r'partnership\s+(?:structure|model)'
        ],
        'Project-Based': [  # Based on PMI PMBOK guidelines
            r'project\s+(?:governance|framework|structure)',
            r'program\s+management\s+office',
            r'project\s+portfolio\s+management',
            r'project\s+(?:coordination|organization)',
            r'program\s+(?:structure|framework)'
        ]
    }

    # Job title patterns based on common organizational roles
    # Sources: O*NET occupational database and LinkedIn job title standardization
    title_patterns = {
        'Hierarchical': [  # Traditional management titles
            r'chief|ceo|cfo|cto|coo|president|vice president|vp|director|manager|supervisor|team lead'
        ],
        'Functional': [  # Specialized role titles
            r'head of|director of|manager of|lead|specialist|expert|analyst|engineer'
        ],
        'Matrix': [  # Dual-reporting role titles
            r'project manager|program manager|product manager|scrum master|product owner'
        ],
        'Team-Based': [  # Agile/team role titles
            r'scrum master|agile coach|team lead|product owner|tech lead'
        ],
        'Process-Based': [  # Process management titles
            r'process manager|operations manager|process engineer|quality manager'
        ],
        'Project-Based': [  # Project management titles
            r'project manager|program manager|project lead|project coordinator'
        ]
    }

    # Industry-specific patterns based on industry research and standards
    # Sources: Industry reports from McKinsey, Deloitte, and Gartner
    industry_patterns = {
        'Technology': {  # Tech industry practices
            'Team-Based': r'agile|scrum|sprint|kanban|devops',
            'Matrix': r'product\s+(?:team|organization)|cross-functional'
        },
        'Manufacturing': {  # Manufacturing industry standards
            'Process-Based':
            r'lean|six sigma|continuous improvement|quality control',
            'Hierarchical':
            r'plant manager|production supervisor|operations manager'
        },
        'Financial': {  # Financial sector structures
            'Divisional': r'business unit|profit center|trading desk',
            'Matrix': r'product group|client team'
        },
        'Healthcare': {  # Healthcare organization models
            'Functional': r'clinical|medical|nursing|specialty',
            'Matrix': r'care team|medical staff'
        }
    }

    # Initialize counters
    structure_counts = {
        structure: 0
        for structure in structure_keywords.keys()
    }
    structure_counts['Other'] = 0

    # Analyze each job posting
    for _, row in df.iterrows():
        if not isinstance(row['description'], str):
            structure_counts['Other'] += 1
            continue

        desc = str(row['description']).lower()
        title = str(row.get('title', '')).lower()
        found_structures = set()

        # Check for keyword matches
        for structure, keywords in structure_keywords.items():
            if any(keyword in desc for keyword in keywords):
                found_structures.add(structure)

        # Check for context pattern matches
        for structure, patterns in context_patterns.items():
            if any(re.search(pattern, desc) for pattern in patterns):
                found_structures.add(structure)

        # Check job title patterns
        for structure, patterns in title_patterns.items():
            if any(re.search(pattern, title) for pattern in patterns):
                found_structures.add(structure)

        # Check industry-specific patterns
        for industry, patterns in industry_patterns.items():
            if industry.lower() in desc:
                for structure, pattern in patterns.items():
                    if re.search(pattern, desc):
                        found_structures.add(structure)

        # Handle Remote/Hybrid separately as it can coexist with other structures
        has_remote = 'Remote/Hybrid' in found_structures
        found_structures.discard('Remote/Hybrid')

        # Assign structure based on priority if any structures were found
        if found_structures:
            # Priority order for resolving multiple structures
            priority_order = [
                'Matrix',  # Most specific/complex
                'Network',
                'Process-Based',
                'Project-Based',
                'Divisional',
                'Team-Based',
                'Functional',
                'Flat/Horizontal',
                'Hierarchical'  # Most general/basic
            ]

            # Try to assign based on priority
            assigned = False
            for priority_structure in priority_order:
                if priority_structure in found_structures:
                    structure_counts[priority_structure] += 1
                    assigned = True
                    break

            # If no priority structure found but we have structures, use any of them
            if not assigned and found_structures:
                structure_counts[next(iter(found_structures))] += 1
        else:
            # Try to infer structure from job level and reporting relationships
            if re.search(r'senior|lead|manager|director|vp|chief|head', title):
                structure_counts['Hierarchical'] += 1
            elif re.search(r'specialist|analyst|engineer|developer', title):
                structure_counts['Functional'] += 1
            else:
                structure_counts['Other'] += 1

        # Add Remote/Hybrid count if applicable
        if has_remote:
            structure_counts['Remote/Hybrid'] += 1

    # Calculate percentages (excluding Remote/Hybrid from total since it can coexist)
    total = sum(structure_counts.values()) - structure_counts['Remote/Hybrid']
    structure_percentages = {
        k: (v / total) * 100
        for k, v in structure_counts.items()
    }

    return structure_counts, structure_percentages


def create_org_structure_visualization(
        sp500_data, other_data, output_file='org_structure_comparison.png'):
    """
    Create a bar chart comparing organizational structures between S&P 500 and other companies.
    """
    plt.figure(figsize=(12, 6))

    structures = list(sp500_data[0].keys())
    x = np.arange(len(structures))
    width = 0.35

    # Convert counts to percentages for comparison
    sp500_pct = [sp500_data[1][struct] for struct in structures]
    other_pct = [other_data[1][struct] for struct in structures]

    plt.bar(x - width / 2,
            sp500_pct,
            width,
            label='S&P 500 Companies',
            color='#2ecc71')
    plt.bar(x + width / 2,
            other_pct,
            width,
            label='Other Companies',
            color='#3498db')

    plt.xlabel('Organizational Structure')
    plt.ylabel('Percentage of Job Postings')
    plt.title(
        'Organizational Structure Distribution: S&P 500 vs Other Companies')
    plt.xticks(x, structures, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()

    return output_file


def generate_html_report(results):
    """Generate an HTML report with all analysis results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>S&P 500 vs Other Companies Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .visualization {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>S&P 500 vs Other Companies Analysis</h1>
        
        <div class="section">
            <h2>Organizational Structure Distribution</h2>
            <p>Analysis of organizational structures mentioned in job postings:</p>
            <div class="visualization">
                <img src="org_structure_comparison.png" alt="Organizational Structure Distribution">
            </div>
            <table>
                <tr>
                    <th>Structure Type</th>
                    <th>S&P 500 Companies (%)</th>
                    <th>Other Companies (%)</th>
                </tr>
    """

    # Add rows for each structure type
    for structure in results['sp500_org_structure'][0].keys():
        sp500_pct = results['sp500_org_structure'][1][structure]
        other_pct = results['other_org_structure'][1][structure]
        html_content += f"""
                <tr>
                    <td>{structure}</td>
                    <td>{sp500_pct:.1f}%</td>
                    <td>{other_pct:.1f}%</td>
                </tr>
        """

    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Skills Analysis</h2>
            <div class="visualization">
                <img src="skills_comparison.png" alt="Skills Distribution">
            </div>
        </div>
    </body>
    </html>
    """

    with open('analysis_report.html', 'w') as f:
        f.write(html_content)


def extract_skills(df):
    """Extract skills from job descriptions and skills_desc."""
    skill_patterns = {
        'Programming': [
            r'python|java|javascript|c\+\+|ruby|php|swift|kotlin|golang|rust',
            r'typescript|scala|perl|r programming|matlab|shell scripting'
        ],
        'Web Development': [
            r'html|css|react|angular|vue|node\.js|django|flask|spring|asp\.net',
            r'web development|frontend|backend|full stack|restful api'
        ],
        'Database': [
            r'sql|mysql|postgresql|mongodb|oracle|redis|elasticsearch',
            r'database|nosql|cassandra|dynamodb|neo4j'
        ],
        'Cloud': [
            r'aws|azure|gcp|google cloud|cloud computing|kubernetes|docker',
            r'containerization|microservices|serverless|cloud native'
        ],
        'AI & ML': [
            r'machine learning|artificial intelligence|deep learning|neural networks',
            r'nlp|computer vision|tensorflow|pytorch|scikit-learn|data science'
        ],
        'DevOps': [
            r'ci/cd|jenkins|git|github|gitlab|bitbucket|devops|sre',
            r'infrastructure as code|ansible|terraform|puppet|chef'
        ],
        'Security': [
            r'cybersecurity|information security|network security|encryption',
            r'penetration testing|security audit|compliance|authentication'
        ],
        'Project Management': [
            r'agile|scrum|kanban|jira|confluence|project management',
            r'product management|roadmap|sprint planning|stakeholder'
        ],
        'Soft Skills': [
            r'communication|leadership|teamwork|problem solving|analytical',
            r'collaboration|time management|critical thinking|presentation'
        ]
    }

    # Initialize counters
    skill_counts = {category: 0 for category in skill_patterns.keys()}
    total_postings = len(df)

    # Process each job posting
    for _, row in df.iterrows():
        description = str(row['description']).lower()
        skills_desc = str(row['skills_desc']).lower()
        combined_text = f"{description} {skills_desc}"

        for category, patterns in skill_patterns.items():
            if any(re.search(pattern, combined_text) for pattern in patterns):
                skill_counts[category] += 1

    # Convert to percentages
    skill_percentages = {
        k: (v / total_postings) * 100
        for k, v in skill_counts.items()
    }

    return skill_counts, skill_percentages


def create_skills_visualization(sp500_data,
                                other_data,
                                output_file='skills_comparison.png'):
    """Create visualization comparing skills between S&P 500 and other companies."""
    plt.figure(figsize=(12, 6))

    categories = list(sp500_data[0].keys())
    x = np.arange(len(categories))
    width = 0.35

    sp500_pct = [sp500_data[1][cat] for cat in categories]
    other_pct = [other_data[1][cat] for cat in categories]

    plt.bar(x - width / 2,
            sp500_pct,
            width,
            label='S&P 500 Companies',
            color='#2ecc71')
    plt.bar(x + width / 2,
            other_pct,
            width,
            label='Other Companies',
            color='#3498db')

    plt.xlabel('Skill Categories')
    plt.ylabel('Percentage of Job Postings')
    plt.title('Skills Distribution: S&P 500 vs Other Companies')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()

    return output_file


def analyze_sp500_skills(postings_file='postings.csv'):
    """Main function to analyze skills in S&P 500 companies vs others."""
    print("Loading S&P 500 companies...")
    sp500_companies = load_sp500_companies()

    print("Loading job postings data...")
    df = pd.read_csv(postings_file)

    # Filter for S&P 500 companies
    sp500_postings = df[df['company_name'].apply(
        lambda x: is_sp500_company(x, sp500_companies))]
    other_postings = df[~df['company_name'].
                        apply(lambda x: is_sp500_company(x, sp500_companies))]

    print(f"Found {len(sp500_postings)} job postings from S&P 500 companies")
    print(f"Found {len(other_postings)} job postings from other companies")

    # Analyze organizational structure
    print("Analyzing organizational structure...")
    sp500_org_structure = analyze_org_structure(sp500_postings)
    other_org_structure = analyze_org_structure(other_postings)

    # Create visualization
    create_org_structure_visualization(sp500_org_structure,
                                       other_org_structure)

    # Extract and analyze skills
    print("Analyzing skills distribution...")
    sp500_skills = extract_skills(sp500_postings)
    other_skills = extract_skills(other_postings)

    # Create skills visualization
    create_skills_visualization(sp500_skills, other_skills)

    # Generate report
    results = {
        'sp500_org_structure': sp500_org_structure,
        'other_org_structure': other_org_structure,
        'sp500_skills': sp500_skills,
        'other_skills': other_skills
    }

    generate_html_report(results)
    print("Analysis complete! Check analysis_report.html for results.")


if __name__ == "__main__":
    analyze_sp500_skills()
