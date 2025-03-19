![](https://img.shields.io/badge/Built%20with%20%E2%9D%A4%EF%B8%8F-at%20Technologiestiftung%20Berlin-blue)

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# TSB DCATAP AI Analyzer

# **How to Use This Notebook**

This notebook analyzes open government dataset metadata with a Large Language Model (LLM) and saves the enriched results to an Excel file. Below is a brief guide on how to run it.

---

### 1. Prerequisites

1. **Environment Setup**  
   - Make sure you have a Python 3 environment with the necessary libraries installed. You can install all required packages from `requirements.txt` using:
     ```bash
     pip install -r requirements.txt
     ```

2. **OpenAI API Key**  
   - Set your OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY=YOUR_OPENAI_KEY
     ```
   - The notebook will check for `OPENAI_API_KEY` and will raise an error if it’s not found.

---

### 2. Running the Notebook

1. **Navigate to the `/notebooks` directory**  
   ```bash
   cd notebooks
   ```
2. **Open the `llm_assessment.ipynb` (or similarly named) notebook**  
   - Select the notebook to run it step by step.

---

### 3. What the Notebook Does

1. **Retrieve Metadata**  
   - Pulls a full list of datasets from the Berlin open data API (CKAN) and saves it as a Parquet file (`metadata.parquet`).

2. **Filter or Inspect Data (Optional)**  
   - You can optionally filter the metadata by tags, publisher, or any other criteria before analysis.

3. **Semantic Analysis with LLM**  
   - For each dataset (title, description, etc.), the notebook calls an OpenAI model to generate semantic insights:
     - `dateninhalt_score`, `methodik_score`, `datenqualitaet_score`, `geographie_score`, `tag_qualitaet_score`, `referenz_score`
     - Human-readable text assessments for content, methods, data quality, geography, tag quality and reference.

4. **Combine and Save**  
   - The original metadata is combined with the new LLM-generated columns into a single DataFrame.
   - This final enriched DataFrame is saved to an Excel file in the `/_results` folder, named with a current date stamp (e.g., `metadata_analysis_YYYYMMDD.xlsx`).

---

### 4. Adjusting the Number of Datasets

- If you want to test only on a small subset to save time or API credits, update the line:
  ```python
  num_datasets_to_analyze = 10  # or None for all datasets
  ```
  This slices the DataFrame so only a certain number of rows are sent to the LLM.

---

### 5. Output

- **Location:**  
  An Excel file (e.g., `metadata_analysis_20250101.xlsx`) in `../_results`.
- **Contents:**  
  Original metadata columns + LLM scores (`dateninhalt_score`, `methodik_score`, `datenqualitaet_score`, `geographie_score`, `tag_qualitaet_score`, `referenz_score`) + textual summaries.


## Contributing

Before you create a pull request, write an issue so we can discuss your changes.

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/malte-b"><img src="https://avatars.githubusercontent.com/u/27922183?v=4?s=64" width="64px;" alt="Malte Barth"/><br /><sub><b>Malte Barth</b></sub></a><br /><a href="https://github.com/technologiestiftung/template-default/commits?author=malte-b" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Content Licensing

Texts and content available as [CC BY](https://creativecommons.org/licenses/by/3.0/de/).

## Credits

<table>
  <tr>
    <td>
      Made by <a href="https://citylab-berlin.org/de/start/">
        <br />
        <br />
        <img width="200" src="https://logos.citylab-berlin.org/logo-citylab-color.svg" alt="Link to the CityLAB Berlin website" />
      </a>
    </td>
    <td>
      A project by <a href="https://www.technologiestiftung-berlin.de/">
        <br />
        <br />
        <img width="150" src="https://logos.citylab-berlin.org/logo-technologiestiftung-berlin-de.svg" alt="Link to the Technologiestiftung Berlin website" />
      </a>
    </td>
    <td>
      Supported by <a href="https://www.berlin.de/rbmskzl/">
        <br />
        <br />
        <img width="80" src="https://logos.citylab-berlin.org/logo-berlin-senatskanzelei-de.svg" alt="Link to the Senate Chancellery of Berlin"/>
      </a>
    </td>
  </tr>
</table>

## Related Projects
https://github.com/tifa365/dcatapde_ai_analyzer
https://github.com/machinelearningZH/ogd_ai-analyzer/tree/main