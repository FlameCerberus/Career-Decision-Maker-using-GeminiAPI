# %%
# Import libraries
import pandas as pd
import os
import streamlit as st
import requests
import csv
from jobspy import scrape_jobs
import pandas as pd
import google.generativeai as genai
from PyPDF2 import PdfReader
import streamlit as st
import json
import time
import numpy as np


st.set_page_config(page_title="Smart Decision Assistant", page_icon="🧊", layout="wide")

# Getting Image path
gif_path = os.path.join(os.getcwd(), 'Images/Logo.gif')


# Google API setup
geminiapi = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=geminiapi)

# %%
def gemini_ai(education_saved, work_experience_saved, projects_saved, skills_saved, job_desc):

   # Gemini set up configuration
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {   "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]
    model = genai.GenerativeModel('gemini-pro')

    context = [
    {
        "role": "user",
        "parts": "You are the best career advisor which shares opinions on how the skills and job description given to you matches and what you can highlight during the interview. Given details and job description from user, generate how much a match of the job is and you opinions on the job match. If the job does not match with most of the user details, disagree and give opinions on why it does not match. Do not say feel free to ask as you will be giving a one time advice"
    },
    {
        "role": "model",
        "parts": "Yes, I am the best career advisor, I will generate a reply on giving opinions whether the user's detail and background studies compared to the job description given. If the job is not matching, I will explain why. If the job does match, I will explain why"
    },
    {
        "role": "user",
        "parts": """Education: Universiti Tenaga Nasional (UNITEN) Oct. 2018 – Feb. 2023
Bachelor of Electrical and Electronics Engineering (Hons.)
▪ CGPA: 3.61. First Class Honours
▪ Courses: Microprocessor Systems, Artificial Intelligence, Digital Signal Processing, Data Communication and
Network, Radio Frequency & Microwave Engineering, Computer Organization and Architecture, VLSI Design
Kolej Matrikulasi Kelantan Jun. 2017 – Jun. 2018
Malaysian Matriculation (Science PST Module 1)
▪ CGPA: 3.78\n Work Experience: Test Engineer at Semi,  I developed and executed test plans for electronic components, ensuring product quality and reliability. I also collaborated with design teams to enhance product performance and efficiency.Tenaga Nasional Berhad Jun. 2022 – Sep. 2022
Electrical Engineer Intern
▪ Analyzed and simplified the single line diagram of power distribution in AutoCAD for Kelantan area,
focusing on creating one- way line configurations, used by the technicians to easily identify flow of electricity
▪ Enhanced my knowledge and skills through hands-on experience and close collaboration with technicians
during on-site visits, providing me with a deep understanding of TNB's power distribution system \nProjects: IoT-based Tracking System and Health Monitoring for the Elderly with Alzheimer
▪ Developed an IoT wearable device prototype using C++ language in VSCode for ESP32 microcontroller
programming, making use of libraries and resources for sensor integration and enhanced functionality
▪ Integrated the ThingSpeak cloud platform to streamline data storage and analysis in CSV format, optimizing data
analysis visualization and ease of use, while using the cloud platform for real-time abnormal data detection
▪ Implemented an alert system that connects ThingSpeak cloud with the Telegram bot API, to notify the user and
send the alert along with the data and location of the wearer triggered it to the user/caretaker through Telegram.
Smart Irrigation System using PIC18 Microcontroller
▪ Designed a system incorporating multiple moisture sensors and pump areas to operate simultaneously by
utilizing polled input for efficient irrigation management in Assembly language using MPLAB software
▪ Implemented interrupt service routines and analog-to-digital converters I/O with appropriate voltage
resolution for accurate moisture detection and to ensure smooth flow of the microcontroller’s program
Mitsubishi RV-M1 Robotic Arm based Mini-Production Line
▪ Object rejection used servo motor, IR sensor and Arduino on the production line using ArduinoIDE software
▪ Mitsubishi Robot Arm was programmed to pick up the accepted object and place it into the designated place \nSkills: Programming Language: C, C++, Python, Assembly, MySQL, MATLAB
Hardware: Oscilloscope, Signal Wave Generator, Arduino, ESP32, PIC18 Microcontroller
Design Programs: AutoCAD, Proteus, Simulink, LTSpice, Microwind, MPLAB, MikroC
Language Proficiency: Malay (Native), English (Fluent) \nJob Description: Job Responsibilities: Join our cutting-edge team and be the driving force behind the next technological revolution. Embrace the challenge of creating groundbreaking innovations and industry-leading solutions that shape the future.

Unleash your passion for technology, innovation, and vision, and transform your aspirations into reality.

 

We seek graduates who have passion, a relentless drive for innovation, and have the vision to look to the future, and the inspiration to make that vision a reality. We have various exciting technical openings for EE graduates. Depending on your areas of interest, you could explore fields such as analog design, digital design, circuit design, embedded system, firmware, product engineering, quality engineer, knowledge engineer, engineering project management in R&D, Manufacturing, or Materials Engineering Departments.

As a graduate, you will first learn about our Businesses, Products, Solutions, various platforms, systems, equipment, machines, processes and etc.
Based on the assignment, you will be applying your Engineering knowledge in your day to day work which involves designing, evaluating, testing and troubleshoot and solve engineering related issues.
EE Engineers will need to collaborate with various stakeholders and customers within and outside the company.
EE Engineers will drive for continuous improvements and efficiency.
Our experienced mentors are here to guide you every step of the way, providing invaluable insights and support. Additionally, the company has various platforms for you to empower you to acquire the necessary skills or technologies to excel in your field.
Be a part of a team that dares to push boundaries, redefine possibilities, and shape the future of technology.
 

Job Qualification

Candidate must possess at least a Bachelor's Degree, in Electrical, Electronics, Mechatronics or equivalent.
No work experience required.
To secure the job, here are some of the qualities to differentiate you from others: attention to detail, analytical mindset, good interpersonal skills, a growth mindset, and curiosity in problem-solving."""
    },
    {
        "role": "model",
        "parts": """Based on the information you have provided, your skills and experience are a good match for the job description. Your education in Electrical and Electronics Engineering, along with your experience as an Electrical Engineer Intern, have given you the necessary knowledge and skills to be successful in this role. You have experience in analyzing and simplifying single line diagrams of power distribution, as well as hands-on experience with electrical systems. You have also developed projects in IoT-based tracking systems, smart irrigation systems, and robotic arms, which demonstrates your ability to apply your engineering knowledge to practical applications.

Here are some specific examples of how your skills and experience match the job description:

Your experience in analyzing and simplifying single line diagrams of power distribution is directly relevant to the job description, which requires candidates to have knowledge of electrical systems.
Your hands-on experience with electrical systems is also a valuable asset, as the job description requires candidates to be able to design, evaluate, test, and troubleshoot electrical systems.
Your project in developing an IoT-based tracking system for the elderly with Alzheimer's demonstrates your ability to apply your engineering knowledge to a real-world problem. This project required you to use your skills in programming, hardware design, and data analysis.
Your project in developing a smart irrigation system using a PIC18 microcontroller demonstrates your ability to design and implement a complete engineering system. This project required you to use your skills in programming, hardware design, and control systems.
Your project in developing a Mitsubishi RV-M1 Robotic Arm based mini-production line demonstrates your ability to work with robotics and automation. This project required you to use your skills in programming, hardware design, and control systems.
Overall, your skills and experience are a good match for the job description, and you should be well-qualified for this role. I would encourage you to apply for the position and highlight your skills and experience in your resume and cover letter."""
    },
    {
        "role": "user",
        "parts": f"Education: {education_saved}\n Work Experience: {work_experience_saved}\n Projects: {projects_saved}\n Skills: {skills_saved}\n\n Job Description: {job_desc}"
    }
]
    



    response = model.generate_content(context)
    return response




def gemini_ai_resume(extracted_resume):
    


    
    # Gemini set up configuration
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {   "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    model = genai.GenerativeModel('gemini-pro')

    contextResume = [
    {
        "role": "user",
        "parts": "You are a Resume Summarizer where you will only OUTPUT 4 MAIN POINTS which are Education, Work Experience, Projects, and Skills using JSON FORMAT [{}]. YOUR REPLY WILL START WITH THE OUTPUT OF THE JSON. NO NEED TO SAY HERE ARE THE JSON OUTPUT ETC "
    },    
    {
        "role": "model",
        "parts": "Yes, I will give a very short summarizer where i will only OUTPUT 4 MAIN POINTS which are Education, Work Experience, Projects, and Skills using JSON FORMAT [{}] and will start with the json "
    },
#     {
#         "role": "user",
#         "parts": """
#         EDUCATION Abang Amirulluqman Farhan bin Abang Kilat
# ❖ abangamirulluq99@gmail.com
# ❖ 018-2649557
# ❖ 1747, Taman Sabariah, Jalan Pengkalan Chepa, 15400, Kota Bharu, Kelantan

# Universiti Tenaga Nasional (UNITEN) Oct. 2018 – Feb. 2023
# Bachelor of Electrical and Electronics Engineering (Hons.)
# ▪ CGPA: 3.61. First Class Honours
# ▪ Courses: Microprocessor Systems, Artificial Intelligence , Digital Signal Processing, Data Communication and
# Network, Radio Frequency & Microwave Engineering, Computer Organization and Architecture, VLSI Design
# Kolej Matrikulasi Kelantan Jun. 2017 – Jun. 2018
# Malaysian Matriculation (Science PST Module 1) ▪ CGPA: 3.78 WORK EXPERIENCE
# Tenaga Nasional Berhad Jun. 2022 – Sep. 2022
# Electrical Engineer Intern
# ▪ Analyzed and simplified the single line diagram of power distribution in AutoCAD for Kelantan area , focusing on creating one - way line configurations , used by the technicians to easily identify flow of electricity
# ▪ Enhanced my knowledge and skills through hands -on experience and close collaboration with technicians
# during on-site visits, providing me with a deep understanding of TNB's power distribution system
# PROJECTS
# IoT-based Tracking System and Health Monitoring for the Elderly with Alzheimer
# ▪ Developed an IoT wearable device prototype using C++ language in VSCode for ESP32 microcontroller programming, making use of libraries and resources for sensor integration and enhanced functionality
# ▪ Integrated the ThingSpeak cloud platform to streamline data storage and analysis in CSV format, optimizing data analysis visualization and ease of use , while using the cloud platform for real -time abnormal data detection
# ▪ Implemented an alert system that connects ThingSpeak cloud with the Telegram bot API, to notify the user and send the alert along with the data and location of the wearer triggered it to the user/caretaker through Telegram.
# Smart Irrigation System using PIC18 Microcontroller
# ▪ Designed a system incorporating multiple moisture sensors and pump areas to operate simultaneously by utilizing polled input for efficient irrigation management in Assembly language using MPLAB software
# ▪ Implemented interrupt service routines and analog -to-digital converters I/O with appropriate voltage resolution for accurate moisture detection and to ensure smooth flow of the microcontroller’s program
# Mitsubishi RV -M1 Robotic Arm based Mini -Production Line
# ▪ Object rejection used servo motor, IR sensor and Arduino on the production line using ArduinoIDE software
# ▪ Mitsubishi Robot Arm was programmed to pick up the accepted object and place it into the designated place
# CERTIFICATIONS AND ACHIEVEMENTS
# ▪ Enrolled in AI & Machine Learning Competence for Industry 4.0 Certification by SHRDC Exp. Mar. 2024
# ▪ Volunteer for Community Service with 3ESC in collaboration with I -WELD Dec. 2022
# ▪ English Trainer role for "Soar with Reading and Writing" Program Aug. 2017
# SKILLS
# Programming Language: C, C++, Python, Assembly , MySQL, MATLAB
# Hardware: Oscilloscope, Signal Wave Generator, Arduino, ESP32, PIC18 Microcontroller
# Design Programs: AutoCAD, Proteus, Simulink , LTSpice, Microwind, MPLAB, MikroC
# Language Proficiency: Malay (Native), English (Fluent)
#         """
#     }, 
    {
        "role": "user",
        "parts": f"Here is my resume: {extracted_resume}"
    }
    
    ]
    response = model.generate_content(contextResume)
    return response

def gemini_ai_job_matcher(education, work_experience, projects, skills, description):
   # Gemini set up configuration
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {   "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]
    model = genai.GenerativeModel('gemini-pro')

    context = [
    {
        "role": "user",
        "parts": "You are the best career advisor. You will give me percentage only answer when comparing my background details compared to the job description"
    },
    {
        "role": "model",
        "parts": "Yes, I am the best career advisor, I will generate a reply by giving only percentage value on how matching the jobs compared to your details. Do put more emphasis on the work experience required compared to the user background"
    },
    {
        "role": "user",
        "parts": """Education: Universiti Tenaga Nasional (UNITEN) Oct. 2018 – Feb. 2023
Bachelor of Electrical and Electronics Engineering (Hons.)
▪ CGPA: 3.61. First Class Honours
▪ Courses: Microprocessor Systems, Artificial Intelligence, Digital Signal Processing, Data Communication and
Network, Radio Frequency & Microwave Engineering, Computer Organization and Architecture, VLSI Design
Kolej Matrikulasi Kelantan Jun. 2017 – Jun. 2018
Malaysian Matriculation (Science PST Module 1)
▪ CGPA: 3.78\n Work Experience: Test Engineer at Semi,  I developed and executed test plans for electronic components, ensuring product quality and reliability. I also collaborated with design teams to enhance product performance and efficiency.Tenaga Nasional Berhad Jun. 2022 – Sep. 2022
Electrical Engineer Intern
▪ Analyzed and simplified the single line diagram of power distribution in AutoCAD for Kelantan area,
focusing on creating one- way line configurations, used by the technicians to easily identify flow of electricity
▪ Enhanced my knowledge and skills through hands-on experience and close collaboration with technicians
during on-site visits, providing me with a deep understanding of TNB's power distribution system \nProjects: IoT-based Tracking System and Health Monitoring for the Elderly with Alzheimer
▪ Developed an IoT wearable device prototype using C++ language in VSCode for ESP32 microcontroller
programming, making use of libraries and resources for sensor integration and enhanced functionality
▪ Integrated the ThingSpeak cloud platform to streamline data storage and analysis in CSV format, optimizing data
analysis visualization and ease of use, while using the cloud platform for real-time abnormal data detection
▪ Implemented an alert system that connects ThingSpeak cloud with the Telegram bot API, to notify the user and
send the alert along with the data and location of the wearer triggered it to the user/caretaker through Telegram.
Smart Irrigation System using PIC18 Microcontroller
▪ Designed a system incorporating multiple moisture sensors and pump areas to operate simultaneously by
utilizing polled input for efficient irrigation management in Assembly language using MPLAB software
▪ Implemented interrupt service routines and analog-to-digital converters I/O with appropriate voltage
resolution for accurate moisture detection and to ensure smooth flow of the microcontroller’s program
Mitsubishi RV-M1 Robotic Arm based Mini-Production Line
▪ Object rejection used servo motor, IR sensor and Arduino on the production line using ArduinoIDE software
▪ Mitsubishi Robot Arm was programmed to pick up the accepted object and place it into the designated place \nSkills: Programming Language: C, C++, Python, Assembly, MySQL, MATLAB
Hardware: Oscilloscope, Signal Wave Generator, Arduino, ESP32, PIC18 Microcontroller
Design Programs: AutoCAD, Proteus, Simulink, LTSpice, Microwind, MPLAB, MikroC
Language Proficiency: Malay (Native), English (Fluent) \nJob Description: Job Responsibilities: Join our cutting-edge team and be the driving force behind the next technological revolution. Embrace the challenge of creating groundbreaking innovations and industry-leading solutions that shape the future.

Unleash your passion for technology, innovation, and vision, and transform your aspirations into reality.

 

We seek graduates who have passion, a relentless drive for innovation, and have the vision to look to the future, and the inspiration to make that vision a reality. We have various exciting technical openings for EE graduates. Depending on your areas of interest, you could explore fields such as analog design, digital design, circuit design, embedded system, firmware, product engineering, quality engineer, knowledge engineer, engineering project management in R&D, Manufacturing, or Materials Engineering Departments.

As a graduate, you will first learn about our Businesses, Products, Solutions, various platforms, systems, equipment, machines, processes and etc.
Based on the assignment, you will be applying your Engineering knowledge in your day to day work which involves designing, evaluating, testing and troubleshoot and solve engineering related issues.
EE Engineers will need to collaborate with various stakeholders and customers within and outside the company.
EE Engineers will drive for continuous improvements and efficiency.
Our experienced mentors are here to guide you every step of the way, providing invaluable insights and support. Additionally, the company has various platforms for you to empower you to acquire the necessary skills or technologies to excel in your field.
Be a part of a team that dares to push boundaries, redefine possibilities, and shape the future of technology.
 

Job Qualification

Candidate must possess at least a Bachelor's Degree, in Electrical, Electronics, Mechatronics or equivalent.
No work experience required.
To secure the job, here are some of the qualities to differentiate you from others: attention to detail, analytical mindset, good interpersonal skills, a growth mindset, and curiosity in problem-solving."""
    },
    {
        "role":"model",
        "parts":"90%"
    },
    {
        "role": "user",
        "parts": f"Education: {education}\n Work Experience: {work_experience}\n Projects: {projects}\n Skills: {skills}\n\n Job Description: {description}"
    }
]

    response = model.generate_content(context)
    return response.text


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdf_file as f:
        reader = PdfReader(f)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            text += reader.pages[page_num].extract_text()
    return text


def removeJSON(json_string):

    formatted_string = json.dumps(json_string)
    formatted_string = formatted_string.replace('{', '').replace('}', '').replace('"', '').replace(':', '').replace(',', '').replace('[', '').replace(']','')

    return formatted_string



def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

    yield pd.DataFrame(
        np.random.randn(5, 10),
        columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    )

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def main():

    #st.sidebar.write("<img src='logo.gif' >", unsafe_allow_html=True)
    #st.sidebar.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2FnaDdiaTZxbWpweXpmNTBidjV5cGdzcGEwajRyMmpwYng5amljbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tYCYOxKldsfPPDqrWG/giphy.gif", caption='Your Awesome GIF', use_column_width=True)
        
    
    
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()





    st.image('Images/HEADERIMG.svg', use_column_width=True,output_format='SVG')



    st.sidebar.image(gif_bytes)

    st.sidebar.markdown("""


                        

                        **<h2>BITWISE: Smart Decision Assistant</h2>** 

                        Welcome to the Smart Decision Assistant, where cutting-edge AI meets personalized career guidance. Our mission is to empower you with insights and data-driven advice, illuminating the path to your dream job.

                        **Why Choose Smart Decision Assistant?**

                        - **Tailored Recommendations:** Get career suggestions that align with your skills, interests, and goals.
                        - **Real-time Job Postings:** Stay ahead with real-time data on job posting comparison.
                        - **Skill Analysis:** Identify your strengths and discover areas for growth to boost your employability.

                        **How to use Smart Decision Assistant**

                        1. Choose the method to upload your background 🌌
                        2. Upload your Resume/Background 👔
                        3. Compare your background to a job description 💼
                        4. Generate a table by inputting your wanted job title for job recommendations 📑
                        <br>
                        **<h2>About</h2>**
                        Welcome to Smart Decision Assistant, a tool powered by Gemini-1.0 Pro designed to compare job descriptions and job postings compatibility compared to your background.
                        <br>
                        Job seekers are always in a hustle of searching for jobs and requires a lot of time and energy to surf job postings throughout different websites. Hence, the AI is designed to reduce the load on the jobseekers by listing the perfect job for them using AI.
                        
                        Created by: [Abang Amirulluqman Farhan](https://www.linkedin.com/in/abang-amirulluqman-farhan-abang-kilat-073206226)
                        ⭐ Star my work on GitHub: [![GitHub stars](https://img.shields.io/github/stars/FlameCerberus/Career-Decision-Maker-using-GeminiAPI.svg?style=social)](https://github.com/FlameCerberus/Career-Decision-Maker-using-GeminiAPI)

                        **<h2>FAQs</h2>**

                        **<h2>How does the Decision Maker AI works?</h2>**
                        When you enter your resume, Gemini AI is used to summarize your resume. Then, for the job description the AI is comparing to the resume or user background uploaded to the application.
                        <br>
                        The job search works by scraping the information from the jobsites selected by the user. All the job description scraped is fed into the AI for comparison one by one.
                        **<h2>Is the AI 100% accurate in its job comparison?</h2>**
                        The AI are not 100% accurate as it uses Gemini-1.0 Pro which uses Large Language Models (LLMs) to generate the output. The LLM itself are very powerful and accurate but are sometimes prone to hallucinations (Generating inaccurate or irrelevant content). The output should just be a reference or starting point for your career search. Use the application at the risk of some results might be innacurate.
                        <br>
                        **<h2>Why do i keep getting error?</h2>**
                        The errors might happen due to the AI not outputting the results in a certain format which is not compatible with the application. This happens due to the hallucinations that might happen. A solution to this is try re-running the application and re-inputting your background
            
                                
""", 
unsafe_allow_html=True)


    ######################## Title and description #########################
    st.markdown("<h1 style='text-align: center;'>Smart Decision Assistant: Let AI Guide Your Career Choice</h1><br><br>", unsafe_allow_html=True)
    #st.markdown("Our platform AI provides you with the necessary resources and data to help you make clear and informed career choices that are in line with your professional objectives and market trends")


    ##################### GPT RESUME #####################

    

    if "resume" not in st.session_state:
        st.session_state.resume = False
    if "education_saved" not in st.session_state:
        st.session_state.education_saved = ""
    if "work_experience_saved" not in st.session_state:
        st.session_state.work_experience_saved = ""
    if "projects_saved" not in st.session_state:
        st.session_state.projects_saved = ""
    if "skills_saved" not in st.session_state:
        st.session_state.skills_saved = ""

    col1, col2, col3 = st.columns(3)

    with col1:
        selection = st.selectbox("Choose upload background method", ["Resume Scanner","Manual"], 0)


    if selection == "Manual":
        st.title("Enter your details")

        education = st.text_area("Education:", value=st.session_state.education_saved)
        work_experience = st.text_area("Work Experience:", value=st.session_state.work_experience_saved)
        projects = st.text_area("Projects:", value=st.session_state.projects_saved)
        skills = st.text_area("Skills:", value=st.session_state.skills_saved)
        
        if st.button("Submit"):
            if education.strip() and work_experience.strip() and projects.strip() and skills.strip():
                st.success("Successfully saved!")
            else:
                st.error("Error: There are missing data in one of the variables", icon="🚨")

    elif selection == "Resume Scanner":
    
        st.title("Resume Scanner")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        
        if st.session_state.resume == False:
            if uploaded_file is not None:
                text = extract_text_from_pdf(uploaded_file)
                #st.text("Extracted Text:")
                #st.write(text)
                summarized_result = gemini_ai_resume(text)


                #st.write("Summarized:")
                #st.write(summarized_result.text)

                cleaned_data = summarized_result.text.replace("json", "").replace("`", "")
                data = json.loads(cleaned_data)

                education_json = data.get("Education")
                work_experience_json = data.get("Work Experience") or data.get("WorkExperience") or data.get("Work_Experience")
                projects_json = data.get("Projects")
                skills_json = data.get("Skills")

                education_text = removeJSON(education_json)
                work_experience_text = removeJSON(work_experience_json)
                projects_text = removeJSON(projects_json)
                skills_text = removeJSON(skills_json)

                st.session_state.education_saved = education_text
                st.session_state.work_experience_saved = work_experience_text
                st.session_state.projects_saved = projects_text
                st.session_state.skills_saved = skills_text
                st.session_state.resume = True

                with st.expander("See Information"):
                    st.subheader("Education")
                    st.write(education_text)

                    st.subheader("Work Experience")
                    st.write(work_experience_text)

                    st.subheader("Projects")
                    st.write(projects_text)

                    st.subheader("Skills")
                    st.write(skills_text)
                    #st.write("**Work Experience:**\n", work_experience_text)
                    #st.write("**Projects:**\n", projects_text)
                    #st.write("**Skills:**", skills_text)



            
            else:
                st.warning('Please Upload Your Resume in PDF Format', icon="⚠️")
            
    education = st.session_state.education_saved
    work_experience = st.session_state.work_experience_saved
    projects = st.session_state.projects_saved
    skills = st.session_state.skills_saved


    ####### Job Comparison #######
    st.title('Job Description Comparison')
    job_desc = st.text_area("Job Description:")
    submit_button = st.button('Generate AI opinion and summary')
    
    # Processing user input
    if job_desc is not None and submit_button:
        with st.spinner("Generating opinion"):
            opinion_result = gemini_ai(st.session_state.education_saved, st.session_state.work_experience_saved, st.session_state.projects_saved, st.session_state.skills_saved, job_desc)
            with st.container(border=True):
                #st.write_stream(opinion_result.text)
                st.write(opinion_result.text)
            st.toast('Succesfully Generated Opinion', icon='🤖')

        

    #################################################################################################################################################################
            

    
    ####### Job Search Automation Part (Scraping) #######
    st.title('Job Search Match')
    job_desc = st.text_area("Input your desired Job Title")
    
    column1, column2, column3 = st.columns(3)

    with column1:  
        num_job = st.number_input('Enter the number of jobs to list', value=5, min_value = 1, max_value = 15)       #  Number of jobs to list in the html table
    with column1:
        options = st.multiselect('Platform to find the job listings', ['Indeed','LinkedIn','Test'], default='Indeed')   # Multiselect options for the job scrape
    site_names = [option.lower() for option in options]                                                             # Formatting for job scrape
    if st.button('Find Jobs') and num_job > 0 and num_job < (num_job + 1):
        with st.spinner('Searching for jobs...'):


            jobs = scrape_jobs(     # Job scraping
                site_name=site_names,
                search_term=job_desc,
                location="Malaysia",
                results_wanted=num_job,
                hours_old=240,  # Only LinkedIn is hour specific, others round up to days old
                country_indeed='Malaysia'  # Only needed for indeed / glassdoor
            )
            jobs.to_csv("jobss.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)
            df = pd.read_csv("jobss.csv")
            df['match_rate'] = (df['title'] + ' ' + df['description']).apply(lambda x: gemini_ai_job_matcher(education, work_experience, projects, skills, x))
            df = df[["title", "company", "match_rate", "job_url"]]
            df = df.sort_values(by='match_rate', ascending=False, na_position='last')
            df['Job Title'] = df.apply(lambda row: make_clickable(row['title'], row['job_url']), axis=1)
            df = df.rename(columns={'company': 'Company', 'match_rate': 'Match Rate'})
            st.header("Jobs")
        # Convert DataFrame to HTML with render_links=True and escape=False
            
            #st.dataframe(df) debug

            df = df[['Job Title', 'Company', 'Match Rate']]
            html = df.to_html(escape=False, index=False)
            styled_html = f"""
                            <style>
                            table {{
                                margin: auto;
                                text-align: center;
                                background-color: dark;
                                border-collapse: collapse;
                            }}
                            th, td {{
                                border: 1px solid white;
                                padding: 8px; 
                                color: white; 
                            }}
                            th {{
                            background-color: #bd93f9; /* COLOUR HERE */
                        }}
                            </style>{html}
                            """

            st.markdown(styled_html, unsafe_allow_html=True)



    #################################################################################################################################################################


def make_clickable(job_title, job_url):
    return f'<a target="_blank" href="{job_url}">{job_title}</a>'

# Function to create a number input with inline "+" and "-" buttons and manual entry




if __name__ == '__main__':
    main()