#!/usr/bin/env python3
"""
Batch update interactive scripts to only save HTML
"""
import os
import re

def update_save_function(file_path):
    """Update the save_interactive_plot function to only save HTML"""

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Simple function to replace
        new_save_function = '''def save_interactive_plot(plot, filename):
    """Save the interactive plot as HTML only"""
    # Always save as HTML (this always works)
    html_filename = filename.replace('.png', '.html').replace('.svg', '.html')
    hv.save(plot, html_filename)
    print(f"Interactive plot saved as '{html_filename}'")'''

        # Find the function and replace it (this is a simple approach)
        # Look for the function definition and replace everything until the next function or end
        pattern = r'def save_interactive_plot\(.*?\n(?:.*\n)*?(?=def |\Z)'

        if 'def save_interactive_plot' in content:
            # Replace with new function
            content = re.sub(pattern, new_save_function + '\n\n\n', content, flags=re.MULTILINE)

            with open(file_path, 'w') as f:
                f.write(content)

            print(f"✅ Updated {file_path}")
        else:
            print(f"⚠️  No save_interactive_plot function found in {file_path}")

    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")

def main():
    scripts_to_update = [
        "/home/matthias/ballpushing_utils/src/PCA/interactive_scatterplot_matrix_pca.py",
        "/home/matthias/ballpushing_utils/src/PCA/interactive_scatterplot_matrix_temporal_individual.py",
        "/home/matthias/ballpushing_utils/src/PCA/interactive_scatterplot_matrix_temporal_pca.py"
    ]

    for script in scripts_to_update:
        if os.path.exists(script):
            update_save_function(script)
        else:
            print(f"⚠️  File not found: {script}")

if __name__ == "__main__":
    main()
