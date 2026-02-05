import pandas as pd
import ast
import re
from pathlib import Path
import os

def load_author_harmonization(harmonization_file_path):
    """Load author harmonization mapping from Excel file.
    
    Args:
        harmonization_file_path (str): Path to Harmonize_Authors_Arrays.xlsx
    
    Returns:
        dict: Mapping from author variations to canonical author names
    """
    harmonization_df = pd.read_excel(harmonization_file_path)
    author_mapping = {}
    
    for _, row in harmonization_df.iterrows():
        canonical_author = row['Author']
        # Parse the string representation of the list
        try:
            associated_authors = ast.literal_eval(row['Associated Unique Authors'])
        except (ValueError, SyntaxError):
            # If parsing fails, try treating it as a string and splitting
            associated_authors = [row['Associated Unique Authors']]
        
        # Map all variations to the canonical name
        for author_variation in associated_authors:
            author_mapping[author_variation.strip()] = canonical_author
    
    return author_mapping


def harmonize_authors(df, author_mapping, harmonization_df=None):
    """Harmonize author names using the mapping.
    
    Args:
        df (pd.DataFrame): DataFrame with 'author' column
        author_mapping (dict): Mapping from author variations to canonical names
        harmonization_df (pd.DataFrame, optional): Original harmonization table for fuzzy matching
    
    Returns:
        pd.DataFrame: DataFrame with harmonized 'author' column
    """
    df = df.copy()
    
    # Get canonical authors for fuzzy matching
    canonical_authors = {}
    if harmonization_df is not None:
        for _, row in harmonization_df.iterrows():
            canonical = row['Author']
            canonical_authors[canonical.lower()] = canonical
    
    # Create harmonized author column
    def map_author(author):
        if pd.isna(author):
            return author
        author_str = str(author).strip()
        
        # Try exact match first
        if author_str in author_mapping:
            return author_mapping[author_str]
        
        # Try case-insensitive match
        for variation, canonical in author_mapping.items():
            if variation.lower() == author_str.lower():
                return canonical
        
        # If no exact match found, return original
        # Note: All common variations should be in the harmonization table
        return author_str
    
    df['author_harmonized'] = df['author'].apply(map_author)
    return df


def extract_date_from_filename(filename):
    """Extract date from metadata analysis filename (format: metadata_analysis_YYYYMMDD.xlsx).
    
    Args:
        filename (str): Filename or path
    
    Returns:
        str: Date string in YYYYMMDD format, or None if not found
    """
    match = re.search(r'metadata_analysis_(\d{8})', filename)
    if match:
        return match.group(1)
    return None


def create_organization_score_files(
    metadata_file_path, 
    harmonization_file_path=None,
    output_dir=None
):
    """Create score aggregation files for each organization from metadata analysis.
    
    This function reads a metadata analysis Excel file, groups datasets by author/organization,
    and creates separate Excel files for ONLY organizations in the harmonization table.
    
    Args:
        metadata_file_path (str): Path to the metadata_analysis Excel file
        harmonization_file_path (str, optional): Path to Harmonize_Authors_Arrays.xlsx
        output_dir (str, optional): Directory where organization score files will be saved.
                                   If None, creates open_data_ag_[DATE] folder.
    
    Returns:
        None: Creates Excel files in the output directory
    """
    # Read the metadata analysis file
    df = pd.read_excel(metadata_file_path)
    
    # Load harmonization mapping if provided
    author_mapping = {}
    canonical_authors = set()
    
    if harmonization_file_path and os.path.exists(harmonization_file_path):
        print(f"Loading author harmonization from {harmonization_file_path}...")
        author_mapping = load_author_harmonization(harmonization_file_path)
        print(f"Loaded {len(author_mapping)} author mappings")
        
        # Get list of canonical author names from harmonization table
        harmonization_df = pd.read_excel(harmonization_file_path)
        canonical_authors = set(harmonization_df['Author'].tolist())
        print(f"Will create files for {len(canonical_authors)} canonical organizations only")
        
        # Harmonize authors (pass harmonization_df for fuzzy matching)
        df = harmonize_authors(df, author_mapping, harmonization_df)
        # Use harmonized author for grouping
        author_col = 'author_harmonized'
    else:
        author_col = 'author'
        print("No harmonization file provided, using original author names")
    
    # Determine output directory
    if output_dir is None:
        # Extract date from metadata filename
        date_str = extract_date_from_filename(metadata_file_path)
        if date_str:
            output_dir = f"_results/open_data_ag_{date_str}"
        else:
            output_dir = "_results/open_data_ag"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Score columns to aggregate
    score_columns = [
        "dateninhalt_score",
        "methodik_score",
        "datenqualitaet_score",
        "geographie_score",
        "tag_qualitaet_score",
        "referenz_score",
    ]
    
    # Group by author/organization (using harmonized column if available)
    org_count = 0
    skipped_count = 0
    skipped_authors = []
    
    for author, group_df in df.groupby(author_col):
        # Skip if harmonization is used and this author is not in canonical list
        if canonical_authors and author not in canonical_authors:
            skipped_count += len(group_df)
            skipped_authors.append((author, len(group_df)))
            continue
        
        # Create a summary DataFrame with score counts
        summary_data = []
        
        # For each score value (1, 2, 3), count occurrences for each score type
        for score_value in [1, 2, 3]:
            row_data = {"Unnamed: 0": score_value}
            
            for score_col in score_columns:
                # Count how many datasets have this score value for this score type
                count = (group_df[score_col] == score_value).sum()
                row_data[score_col] = count
            
            summary_data.append(row_data)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Create filename from author name (sanitize for filesystem)
        # Replace problematic characters and create a safe filename
        safe_filename = author.replace("/", "_").replace("\\", "_").strip()
        filename = f"{safe_filename}_scores.xlsx"
        filepath = os.path.join(output_dir, filename)
        
        # Save to Excel
        summary_df.to_excel(filepath, index=False)
        print(f"Created: {filepath} ({len(group_df)} datasets)")
        org_count += 1
    
    print(f"\nCompleted! Created files for {org_count} organizations.")
    if author_col == 'author_harmonized':
        print(f"(Harmonized from {df['author'].nunique()} original unique authors)")
        if skipped_count > 0:
            print(f"\n⚠️  WARNING: Skipped {skipped_count} datasets from {len(skipped_authors)} organizations not in harmonization table:")
            for author, count in sorted(skipped_authors, key=lambda x: x[1], reverse=True):
                print(f"   - {author} ({count} datasets)")


def create_score_table(metadata_file_path, harmonization_file_path=None, output_dir=None):
    """Create a Score_Tabelle.xlsx with average scores per organization.
    
    Only includes organizations that are in the harmonization table.
    Uses all 6 score columns: dateninhalt, methodik, datenqualität, geographie, tag_qualität, referenz.
    Calculates average for each score type, then sums them (theoretical max: 18 for scale 1-3).
    
    Args:
        metadata_file_path (str): Path to the metadata_analysis Excel file
        harmonization_file_path (str, optional): Path to Harmonize_Authors_Arrays.xlsx
        output_dir (str, optional): Directory where Score_Tabelle will be saved.
                                   If None, uses same logic as create_organization_score_files.
    
    Returns:
        None: Creates Score_Tabelle.xlsx in the output directory
    """
    # Read the metadata analysis file
    df = pd.read_excel(metadata_file_path)
    
    # Load harmonization mapping if provided
    author_col = 'author'
    canonical_authors = set()
    
    if harmonization_file_path and os.path.exists(harmonization_file_path):
        print(f"Loading author harmonization for score table...")
        author_mapping = load_author_harmonization(harmonization_file_path)
        
        # Get list of canonical author names
        harmonization_df = pd.read_excel(harmonization_file_path)
        canonical_authors = set(harmonization_df['Author'].tolist())
        
        df = harmonize_authors(df, author_mapping, harmonization_df)
        author_col = 'author_harmonized'
    
    # Determine output directory
    if output_dir is None:
        date_str = extract_date_from_filename(metadata_file_path)
        if date_str:
            output_dir = f"_results/open_data_ag_{date_str}"
        else:
            output_dir = "_results/open_data_ag"
    
    # Use all 6 score columns for Score_Tabelle
    all_6_scores = [
        "dateninhalt_score",
        "methodik_score",
        "datenqualitaet_score",
        "geographie_score",
        "tag_qualitaet_score",
        "referenz_score",
    ]
    
    # Calculate total score per organization
    # Method: Calculate avg for each of the 6 scores, then sum them
    results = []
    for author in canonical_authors:
        author_data = df[df[author_col] == author]
        
        if len(author_data) == 0:
            continue
        
        # Calculate average for each of the 6 scores
        score_avgs = []
        for col in all_6_scores:
            if col in author_data.columns:
                avg = author_data[col].mean()
                score_avgs.append(avg)
        
        # Sum the 6 averages to get total score
        if len(score_avgs) == 6:
            total_score = sum(score_avgs)
            results.append({
                'Zuständige Stelle': author,
                'total_score': total_score
            })
    
    # Create DataFrame and sort by total_score descending
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('total_score', ascending=False).reset_index(drop=True)
    
    # Save to Excel
    output_path = os.path.join(output_dir, "Score_Tabelle.xlsx")
    result_df.to_excel(output_path, index=True)
    
    print(f"\nCreated Score_Tabelle: {output_path}")
    print(f"  Organizations: {len(result_df)}")
    print(f"  Top score: {result_df['total_score'].max():.3f} ({result_df.iloc[0]['Zuständige Stelle']})")
    print(f"  Bottom score: {result_df['total_score'].min():.3f} ({result_df.iloc[-1]['Zuständige Stelle']})")
    print(f"  Average score: {result_df['total_score'].mean():.3f}")
    print(f"  (Theoretical max: 18 using all 6 scores with scale 1-3)")


if __name__ == "__main__":
    # Find the newest metadata_analysis file
    results_dir = Path("_results")
    metadata_files = list(results_dir.glob("metadata_analysis_*.xlsx"))
    
    if not metadata_files:
        print("No metadata_analysis files found in _results directory")
        exit(1)
    
    # Sort by modification time and get the newest
    newest_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
    print(f"Using newest file: {newest_file}")
    
    # Find harmonization file
    harmonization_file = results_dir / "Harmonize_Authors_Arrays.xlsx"
    if not harmonization_file.exists():
        print(f"Warning: Harmonization file not found at {harmonization_file}")
        harmonization_file = None
    
    # Create organization score files
    create_organization_score_files(
        str(newest_file),
        harmonization_file_path=str(harmonization_file) if harmonization_file else None
    )
    
    # Create Score_Tabelle
    create_score_table(
        str(newest_file),
        harmonization_file_path=str(harmonization_file) if harmonization_file else None
    )
