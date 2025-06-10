
# Script 2: description_generator.py
"""
Loads saved SHAP values (.npz) per video and uses a language model via OpenRouter to generate explanations.
Aggregates and writes an Excel file with columns: video_path, tag, description.
"""
import os
import numpy as np
import pandas as pd
import openai

# Configure OpenRouter endpoint and API key for Deepseek v3
openai.api_base = os.getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1')
openai.api_key = os.getenv('OPENROUTER_API_KEY')

def load_shap(shap_file):
    data = np.load(shap_file)
    return data['indices'], data['shap']


def summarize_shap(shap_vals):
    """Convert shap array into a summary text for the LM prompt."""
    stats = np.mean(np.abs(shap_vals), axis=(1,2,3))
    top_idxs = np.argsort(stats)[-3:][::-1]
    return f"Top influential frames: {list(top_idxs)}; mean abs SHAP: {stats[top_idxs].round(3).tolist()}"


def generate_description(shap_summary, tag):
    prompt = (
        f"We have a video classified as {tag}. "
        f"SHAP summary: {shap_summary}. "
        "In 2-3 sentences, explain why the model considers it real or fake, focusing on the key visual features."
    )
    response = openai.ChatCompletion.create(
        model='deepseek/deepseek-r1-0528:free',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


def main(shap_dir, video_dir, tags_csv, output_excel='results.xlsx'):
    tags_df = pd.read_csv(tags_csv)
    records = []

    for _, row in tags_df.iterrows():
        fname, tag = row['video_filename'], row['tag']
        shap_file = os.path.splitext(fname)[0] + '_shap.npz'
        shap_path = os.path.join(shap_dir, shap_file)
        if not os.path.exists(shap_path):
            print(f"Warning: missing SHAP file for {fname}")
            continue

        indices, shap_vals = load_shap(shap_path)
        summary = summarize_shap(shap_vals)
        desc = generate_description(summary, tag)

        records.append({
            'video_path': os.path.join(video_dir, fname),
            'tag': tag,
            'description': desc
        })

    df = pd.DataFrame(records)
    df.to_excel(output_excel, index=False)
    print(f"Results written to {output_excel}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shap_dir', required=True, help='Dir with .npz SHAP files')
    parser.add_argument('--video_dir', required=True, help='Original video directory')
    parser.add_argument('--tags', required=True, help='CSV mapping video filenames to tags')
    parser.add_argument('--out', default='results.xlsx', help='Output Excel file')
    args = parser.parse_args()
    main(args.shap_dir, args.video_dir, args.tags, args.out)

# example usage
"""python description_generator.py \
  --shap_dir /path/to/shap_output_dir \
  --video_dir /path/to/your/videos_dir \
  --tags video_tags.csv \
  --out explanation_results.xlsx
"""
