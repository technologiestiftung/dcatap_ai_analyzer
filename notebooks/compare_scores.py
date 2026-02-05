import pandas as pd
from pathlib import Path


def compare_score_tables(old_score_table_path, new_score_table_path):
    """Compare two Score_Tabelle files and show differences.
    
    Args:
        old_score_table_path: Path to old Score_Tabelle.xlsx
        new_score_table_path: Path to new Score_Tabelle.xlsx
    
    Returns:
        DataFrame with comparison results
    """
    old_df = pd.read_excel(old_score_table_path)
    new_df = pd.read_excel(new_score_table_path)
    
    # Merge on organization name
    comparison = pd.merge(
        old_df,
        new_df,
        on='Zuständige Stelle',
        how='outer',
        suffixes=('_old', '_new')
    )
    
    # Calculate change
    comparison['change'] = comparison['total_score_new'] - comparison['total_score_old']
    comparison['change_pct'] = (comparison['change'] / comparison['total_score_old'] * 100).round(2)
    
    # Sort by absolute change (biggest changes first)
    comparison['abs_change'] = comparison['change'].abs()
    comparison = comparison.sort_values('abs_change', ascending=False)
    
    return comparison


def compare_individual_org_files(old_folder, new_folder):
    """Compare individual organization score files.
    
    Args:
        old_folder: Path to old organization scores folder
        new_folder: Path to new organization scores folder
    
    Returns:
        Dict with detailed comparison results per organization
    """
    old_files = {f.stem.replace('_scores', ''): f for f in Path(old_folder).glob('*_scores.xlsx')}
    new_files = {f.stem.replace('_scores', ''): f for f in Path(new_folder).glob('*_scores.xlsx')}
    
    all_orgs = set(old_files.keys()) | set(new_files.keys())
    
    comparisons = {}
    
    for org in sorted(all_orgs):
        org_data = {'organization': org}
        
        if org not in old_files:
            org_data['status'] = 'NEW'
            org_data['old_datasets'] = 0
            new_df = pd.read_excel(new_files[org])
            org_data['new_datasets'] = len(new_df)
        elif org not in new_files:
            org_data['status'] = 'REMOVED'
            old_df = pd.read_excel(old_files[org])
            org_data['old_datasets'] = len(old_df)
            org_data['new_datasets'] = 0
        else:
            org_data['status'] = 'EXISTING'
            old_df = pd.read_excel(old_files[org])
            new_df = pd.read_excel(new_files[org])
            
            org_data['old_datasets'] = len(old_df)
            org_data['new_datasets'] = len(new_df)
            org_data['dataset_change'] = org_data['new_datasets'] - org_data['old_datasets']
            
            # Compare score distributions
            score_cols = [col for col in old_df.columns if col.endswith('_score')]
            
            org_data['score_changes'] = {}
            for score_col in score_cols:
                if score_col in old_df.columns and score_col in new_df.columns:
                    old_avg = old_df[score_col].mean()
                    new_avg = new_df[score_col].mean()
                    change = new_avg - old_avg
                    org_data['score_changes'][score_col] = {
                        'old_avg': old_avg,
                        'new_avg': new_avg,
                        'change': change
                    }
        
        comparisons[org] = org_data
    
    return comparisons


def print_score_table_comparison(comparison_df):
    """Print formatted comparison of Score_Tabelle."""
    print("\n" + "="*80)
    print("SCORE_TABELLE COMPARISON")
    print("="*80)
    
    # Summary stats
    total_orgs = len(comparison_df)
    new_orgs = comparison_df['total_score_old'].isna().sum()
    removed_orgs = comparison_df['total_score_new'].isna().sum()
    existing_orgs = total_orgs - new_orgs - removed_orgs
    
    print(f"\nOrganizations: {total_orgs} total")
    print(f"  - Existing in both: {existing_orgs}")
    print(f"  - New in latest: {new_orgs}")
    print(f"  - Removed from latest: {removed_orgs}")
    
    if existing_orgs > 0:
        existing = comparison_df[comparison_df['total_score_old'].notna() & comparison_df['total_score_new'].notna()]
        avg_change = existing['change'].mean()
        print(f"\nAverage change for existing orgs: {avg_change:+.3f}")
        print(f"  - Improved: {(existing['change'] > 0).sum()} orgs")
        print(f"  - Declined: {(existing['change'] < 0).sum()} orgs")
        print(f"  - Unchanged: {(existing['change'] == 0).sum()} orgs")
    
    # Top improvements
    print("\n" + "-"*80)
    print("TOP 10 IMPROVEMENTS (biggest score increases):")
    print("-"*80)
    improved = comparison_df[comparison_df['change'] > 0].head(10)
    for idx, row in improved.iterrows():
        print(f"  {row['Zuständige Stelle']:45s} "
              f"{row['total_score_old']:6.3f} → {row['total_score_new']:6.3f} "
              f"({row['change']:+.3f}, {row['change_pct']:+.1f}%)")
    
    # Top declines
    print("\n" + "-"*80)
    print("TOP 10 DECLINES (biggest score decreases):")
    print("-"*80)
    declined = comparison_df[comparison_df['change'] < 0].head(10)
    for idx, row in declined.iterrows():
        print(f"  {row['Zuständige Stelle']:45s} "
              f"{row['total_score_old']:6.3f} → {row['total_score_new']:6.3f} "
              f"({row['change']:+.3f}, {row['change_pct']:+.1f}%)")
    
    # New organizations
    if new_orgs > 0:
        print("\n" + "-"*80)
        print(f"NEW ORGANIZATIONS ({new_orgs}):")
        print("-"*80)
        new = comparison_df[comparison_df['total_score_old'].isna()]
        for idx, row in new.iterrows():
            print(f"  {row['Zuständige Stelle']:45s} {row['total_score_new']:6.3f}")
    
    # Removed organizations
    if removed_orgs > 0:
        print("\n" + "-"*80)
        print(f"REMOVED ORGANIZATIONS ({removed_orgs}):")
        print("-"*80)
        removed = comparison_df[comparison_df['total_score_new'].isna()]
        for idx, row in removed.iterrows():
            print(f"  {row['Zuständige Stelle']:45s} {row['total_score_old']:6.3f}")


def print_individual_org_comparison(comparisons, show_details=True):
    """Print detailed comparison of individual organization files."""
    print("\n" + "="*80)
    print("INDIVIDUAL ORGANIZATION FILE COMPARISON")
    print("="*80)
    
    new_orgs = [o for o, d in comparisons.items() if d['status'] == 'NEW']
    removed_orgs = [o for o, d in comparisons.items() if d['status'] == 'REMOVED']
    existing_orgs = [o for o, d in comparisons.items() if d['status'] == 'EXISTING']
    
    print(f"\nTotal organizations: {len(comparisons)}")
    print(f"  - New: {len(new_orgs)}")
    print(f"  - Removed: {len(removed_orgs)}")
    print(f"  - Existing: {len(existing_orgs)}")
    
    if show_details and existing_orgs:
        print("\n" + "-"*80)
        print("DETAILED CHANGES FOR EXISTING ORGANIZATIONS:")
        print("-"*80)
        
        for org in existing_orgs:
            data = comparisons[org]
            print(f"\n{org}:")
            print(f"  Datasets: {data['old_datasets']} → {data['new_datasets']} "
                  f"({data['dataset_change']:+d})")
            
            if data.get('score_changes'):
                print(f"  Score changes:")
                for score_name, score_data in data['score_changes'].items():
                    change = score_data['change']
                    print(f"    {score_name:25s}: "
                          f"{score_data['old_avg']:.3f} → {score_data['new_avg']:.3f} "
                          f"({change:+.3f})")


def save_comparison_to_excel(comparison_df, output_path):
    """Save comparison results to Excel file."""
    # Clean up the dataframe for export
    export_df = comparison_df[[
        'Zuständige Stelle',
        'total_score_old',
        'total_score_new',
        'change',
        'change_pct'
    ]].copy()
    
    export_df.columns = [
        'Organization',
        'Old Score',
        'New Score',
        'Change',
        'Change %'
    ]
    
    export_df = export_df.sort_values('Change', ascending=False)
    
    export_df.to_excel(output_path, index=False)
    print(f"\n✓ Saved comparison to: {output_path}")


if __name__ == "__main__":
    # Paths
    old_score_table = Path("_results/Score_Tabelle.xlsx")
    new_score_table = Path("_results/open_data_ag_20251204/Score_Tabelle20251204.xlsx")
    
    old_org_folder = Path("_results/open_data_ag")
    new_org_folder = Path("_results/open_data_ag_20251204")
    
    output_comparison = Path("_results/Score_Comparison.xlsx")
    
    # Compare Score_Tabelle
    print("Comparing Score_Tabelle files...")
    comparison_df = compare_score_tables(old_score_table, new_score_table)
    print_score_table_comparison(comparison_df)
    
    # Compare individual organization files
    print("\n\nComparing individual organization files...")
    org_comparisons = compare_individual_org_files(old_org_folder, new_org_folder)
    print_individual_org_comparison(org_comparisons, show_details=True)
    
    # Save comparison to Excel
    save_comparison_to_excel(comparison_df, output_comparison)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
