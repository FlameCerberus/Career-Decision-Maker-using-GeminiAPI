# Smart Decision Assistant: Let AI Guide Your Career
![Smart Decision Assistant Logo](https://github.com/FlameCerberus/Career-Decision-Maker-using-GeminiAPI/blob/main/Images/HEADERIMG.svg)

## Introduction
Welcome to the Smart Decision Assistant, an innovative Streamlit app designed to guide your career decisions with the power of AI. Our app leverages GeminiAI to analyze your resume and match you with the best job opportunities.

## Features

### Resume Analysis
- **Input**: Upload your resume file or enter your resume details manually.
- **Output**: Get a summarized version of your resume in JSON format, ready for comparison with job descriptions.

### Job Compatibility
- **Input**: Enter the job details you're interested in.
- **Output**: Discover how compatible the job is with your resume information according to the GeminiAI LLM

### Job Matching
- **Input**: Choose the job website for job scraping and specify the number of jobs to be compared to.
- **Output**: GeminiAI generates a table showing the percentage match with each job.

## How to Use
1. **Start the App**: Launch the Smart Decision Assistant through Streamlit or [click here](https://career-decision-maker-using-geminiapi.streamlit.app)
2. **Upload Your Resume**: Easily upload your resume file or input the text manually.
3. **Enter Job Details**: Input the job description for a compatibility check.
4. **Scrape Jobs**: Let the app scrape job listings for matching.
5. **Review Matches**: Check out your job matches and percentages in a neatly formatted table.

## Visual Guide
![Upload your Resume file](https://github.com/FlameCerberus/Career-Decision-Maker-using-GeminiAPI/assets/96816249/1b956e18-de22-4fbd-8590-40e3d1552bea)
*Upload your resume file by dragging it or selecting it in your file explorer.*

---

![Job Comparison](https://github.com/FlameCerberus/Career-Decision-Maker-using-GeminiAPI/assets/96816249/c5a9bc49-e9c0-4346-a069-7070bb0e38bb)
*Input the job description you want to have compared to. In this example, an automation software development job description is compared to the resume of an electronic engineer graduate.*

---

![Job Search](https://github.com/FlameCerberus/Career-Decision-Maker-using-GeminiAPI/assets/96816249/d36b6602-009b-4707-b847-c7cbb6da90d8)
*Input the number of jobs wanted to be listed in the table and select the job scraping website. The application would output the table along with a clickable link on the job title to redirect the user to the website. The resume of an electronic engineer graduate was used and the table output is as shown in the image above.*


## Getting Started
To get started with the Smart Decision Assistant:
1. Clone the repo: `git clone https://github.com/FlameCerberus/Career-Decision-Maker-using-GeminiAPI`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run MainStreamlit.py`

## Contributing
We welcome contributions to the Smart Decision Assistant and please process for submitting pull requests when trying to contribute to this project.

## Acknowledgments
- GeminiAI for their free API usage.
- All contributors who have helped shape this project.
