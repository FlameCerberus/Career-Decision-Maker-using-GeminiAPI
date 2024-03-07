# Import libraries
import os
import streamlit as st
import requests
import csv

import pandas as pd
import google.generativeai as genai

# Initialize OpenAI api client
genai.configure(api_key=st.session_state.geminiapi)

def gemini_ai(education, work_experience, projects, skills, description):
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




def main():

    education = st.session_state.education_saved
    work_experience = st.session_state.work_experience_saved
    projects = st.session_state.projects_saved
    skills = st.session_state.skills_saved

    st.text(education)
    st.text(work_experience)
    st.text(projects)
    st.text(skills)
    
    st.title('Job Search Matcher Automata')
    job_desc = st.text_area("Job Title")
    if st.button('Find Jobs'):
        jobs = scrape_jobs(
        site_name=["indeed"],
        search_term=job_desc,
        location="Malaysia",
        results_wanted=10,
        hours_old=240, # (only linkedin is hour specific, others round up to days old)
        country_indeed='Malaysia'  # only needed for indeed / glassdoor
        )
        jobs.to_csv("jobss.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)

        # selected_columns = ["title", "company", "job_url", "match rate"]
        df = pd.read_csv("jobss.csv")
        df.loc[0:4, 'match_rate'] = df.loc[0:4, 'description'].apply(lambda x: gemini_ai(education, work_experience, projects, skills, x))
        df = df[["title", "company", "job_url", "match_rate"]]
        df = df.sort_values(by='match_rate', ascending=False, na_position='last')

        st.header("Jobs")
        st.dataframe(
            df,
            hide_index=True,
            column_config=
            {
                "job_url": st.column_config.LinkColumn("Link")
            }
        )
        

    


    

        


if __name__ == '__main__':
    main()
