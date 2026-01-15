import xml.etree.ElementTree as ET
import pandas as pd
import re

def clean_html(text):
    """Removes HTML tags like <p> and <a> for cleaner Excel viewing."""
    if not text: return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def process_pali_reference_requests(xml_file, full_output, final_output):
    # 1. Initialize storage
    questions = {}  
    answers_pool = {} 
    
    cite_regex = re.compile(
        r'\b(MN|DN|SN|AN|KhP|Dhp|Ud|It|Sn|Vv|Pv|Thag|Thig|SF)\s?\d+' 
        r'|\b[DMS] [i-v] \d+' 
        r'|\b[\w\u0100-\u017F]+ Sutta\b'
        r'|suttacentral\.net|accesstoinsight\.org|dhammatalks\.org'
        , re.IGNORECASE
    )

    print("Step 1: Parsing XML...")
    context = ET.iterparse(xml_file, events=('end',))
    
    for event, elem in context:
        if elem.tag == 'row':
            post_type = elem.get('PostTypeId')
            
            # Filter for Questions with BOTH tags
            if post_type == '1': 
                tags = elem.get('Tags', '')
                if 'pali-canon' in tags and 'reference-request' in tags:
                    questions[elem.get('Id')] = {
                        'Id': elem.get('Id'),
                        'Title': elem.get('Title'),
                        'QuestionBody': elem.get('Body'),
                        'AcceptedId': elem.get('AcceptedAnswerId'),
                        'Tags': tags
                    }
            
            # Store Answers for those questions
            elif post_type == '2':
                parent_id = elem.get('ParentId')
                if parent_id in questions: 
                    if parent_id not in answers_pool:
                        answers_pool[parent_id] = []
                    answers_pool[parent_id].append({
                        'Id': elem.get('Id'),
                        'AnswerBody': elem.get('Body'),
                        'Score': int(elem.get('Score', 0))
                    })
        
        elem.clear()

    print("Step 2: Processing content and citations...")
    final_rows = []
    
    for q_id, q_info in questions.items():
        answers = answers_pool.get(q_id, [])
        if not answers:
            continue
            
        # Select best answer
        best_answer = None
        acc_id = q_info['AcceptedId']
        if acc_id:
            best_answer = next((a for a in answers if a['Id'] == acc_id), None)
        if not best_answer:
            best_answer = max(answers, key=lambda x: x['Score'])

        # Check for citation
        has_cite = 1 if cite_regex.search(best_answer['AnswerBody']) else 0
        
        final_rows.append({
            'PostId': q_info['Id'],
            'Title': q_info['Title'],
            'Question_Text': clean_html(q_info['QuestionBody']),
            'Best_Answer_Text': clean_html(best_answer['AnswerBody']),
            'Tags': q_info['Tags'],
            'pali_canon_cite': has_cite
        })

    # --- Step 3: Exporting ---
    df_full = pd.DataFrame(final_rows)
    
    # Save File 1: Full dataset (includes 0s and 1s)
    df_full.to_excel(full_output, index=False)
    
    # Save File 2: Filtered (Cite=1 only) and remove the cite column
    df_filtered = df_full[df_full['pali_canon_cite'] == 1].copy()
    df_filtered = df_filtered.drop(columns=['pali_canon_cite'])
    df_filtered.to_excel(final_output, index=False)
    
    print("-" * 40)
    print(f"Success!")
    print(f"1. Full File: {full_output} ({len(df_full)} rows)")
    print(f"2. Filtered File (Cite=1 Only): {final_output} ({len(df_filtered)} rows)")
    print("-" * 40)

# --- EXECUTION ---
xml_path = r"C:\Users\rohan\ws\git\TFM\data\data-q&a\src\Posts.xml"
file_full = "PaliCanon_All_Requests.xlsx"
file_filtered = "PaliCanon_Cited_Only.xlsx"

process_pali_reference_requests(xml_path, file_full, file_filtered)