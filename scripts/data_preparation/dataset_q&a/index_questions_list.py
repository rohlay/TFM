import xml.etree.ElementTree as ET
import pandas as pd

def filter_pali_canon_posts(xml_file, output_csv):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    filtered_data = []

    for row in root.findall('row'):
        # Filter 1: PostTypeId="1" (Questions only)
        # Filter 2: Tags containing "pali-canon"
        post_type = row.get('PostTypeId')
        tags = row.get('Tags', '')
        
        if post_type == "1" and 'pali-canon' in tags and 'reference-request' in tags:
            # Extract relevant fields
            post_id = row.get('Id')
            title = row.get('Title')
            accepted_answer_id = row.get('AcceptedAnswerId')
            creation_date = row.get('CreationDate')
            score = row.get('Score')
            
            filtered_data.append({
                'Id': post_id,
                'Title': title,
                'Tags': tags,
                'AcceptedAnswerId': accepted_answer_id,
                'Score': score,
                'CreationDate': creation_date
            })

    # Create DataFrame and export to Excel
    df = pd.DataFrame(filtered_data)
    #df.to_excel(output_csv, index=False, encoding='utf-8-sig')
    df.to_excel(output_csv)
    print(f"Extraction complete. {len(df)} questions saved to {output_csv}")

# Usage
posts_xml = r"C:\Users\rohan\ws\git\TFM\data\data-q&a\src\Posts.xml"
filter_pali_canon_posts(posts_xml, 'PaliCanon_Questions.xlsx')