# %%
# Import libraries
import pandas as pd
import os
import streamlit as st

from PyPDF2 import PdfReader
import google.generativeai as genai
import os
import streamlit as st
import json

# Initialize Gemini api client
genai.configure(api_key=st.session_state.geminiapi)

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
        "parts": "You are the resume summarizer where you will ONLY state Education, Work Experienced, Projects, and Skills and you will be outputing JSON format only/"
    },    
    {
        "role": "model",
        "parts": "Yes, I will give a very short summarizer FOR ONLY Education, Work Experienced, Projects, and Skills JSON csv format"
    },
    {
        "role": "user",
        "parts": """
        EDUCATION Abang Amirulluqman Farhan bin Abang Kilat
❖ abangamirulluq99@gmail.com
❖ 018-2649557
❖ 1747, Taman Sabariah, Jalan Pengkalan Chepa, 15400, Kota Bharu, Kelantan

Universiti Tenaga Nasional (UNITEN) Oct. 2018 – Feb. 2023
Bachelor of Electrical and Electronics Engineering (Hons.)
▪ CGPA: 3.61. First Class Honours
▪ Courses: Microprocessor Systems, Artificial Intelligence , Digital Signal Processing, Data Communication and
Network, Radio Frequency & Microwave Engineering, Computer Organization and Architecture, VLSI Design
Kolej Matrikulasi Kelantan Jun. 2017 – Jun. 2018
Malaysian Matriculation (Science PST Module 1) ▪ CGPA: 3.78 WORK EXPERIENCE
Tenaga Nasional Berhad Jun. 2022 – Sep. 2022
Electrical Engineer Intern
▪ Analyzed and simplified the single line diagram of power distribution in AutoCAD for Kelantan area , focusing on creating one - way line configurations , used by the technicians to easily identify flow of electricity
▪ Enhanced my knowledge and skills through hands -on experience and close collaboration with technicians
during on-site visits, providing me with a deep understanding of TNB's power distribution system
PROJECTS
IoT-based Tracking System and Health Monitoring for the Elderly with Alzheimer
▪ Developed an IoT wearable device prototype using C++ language in VSCode for ESP32 microcontroller programming, making use of libraries and resources for sensor integration and enhanced functionality
▪ Integrated the ThingSpeak cloud platform to streamline data storage and analysis in CSV format, optimizing data analysis visualization and ease of use , while using the cloud platform for real -time abnormal data detection
▪ Implemented an alert system that connects ThingSpeak cloud with the Telegram bot API, to notify the user and send the alert along with the data and location of the wearer triggered it to the user/caretaker through Telegram.
Smart Irrigation System using PIC18 Microcontroller
▪ Designed a system incorporating multiple moisture sensors and pump areas to operate simultaneously by utilizing polled input for efficient irrigation management in Assembly language using MPLAB software
▪ Implemented interrupt service routines and analog -to-digital converters I/O with appropriate voltage resolution for accurate moisture detection and to ensure smooth flow of the microcontroller’s program
Mitsubishi RV -M1 Robotic Arm based Mini -Production Line
▪ Object rejection used servo motor, IR sensor and Arduino on the production line using ArduinoIDE software
▪ Mitsubishi Robot Arm was programmed to pick up the accepted object and place it into the designated place
CERTIFICATIONS AND ACHIEVEMENTS
▪ Enrolled in AI & Machine Learning Competence for Industry 4.0 Certification by SHRDC Exp. Mar. 2024
▪ Volunteer for Community Service with 3ESC in collaboration with I -WELD Dec. 2022
▪ English Trainer role for "Soar with Reading and Writing" Program Aug. 2017
SKILLS
Programming Language: C, C++, Python, Assembly , MySQL, MATLAB
Hardware: Oscilloscope, Signal Wave Generator, Arduino, ESP32, PIC18 Microcontroller
Design Programs: AutoCAD, Proteus, Simulink , LTSpice, Microwind, MPLAB, MikroC
Language Proficiency: Malay (Native), English (Fluent)
        """
    }, 
    {
        "role": "model",
        "parts": """
        {
  "Education": [
    {
      "Institution": "Universiti Tenaga Nasional (UNITEN)",
      "Degree": "Bachelor of Electrical and Electronics Engineering (Hons.)",
      "CGPA": "3.61",
      "Honors": "First Class Honours",
      "Duration": "Oct.2018–Feb.2023"
    },
    {
      "Institution": "Kolej Matrikulasi Kelantan",
      "Degree": "Malaysian Matriculation (Science PST Module 1)",
      "CGPA": "3.78",
      "Duration": "Jun.2017–Jun.2018"
    }
  ],
  "Work Experience": [
    {
      "Company": "Tenaga Nasional Berhad",
      "Position": "Electrical Engineer Intern",
      "Duration": "Jun.2022–Sep.2022",
      "Responsibilities": [
        "Analyzed and simplified the single line diagram of power distribution in AutoCAD for Kelantan area, focusing on creating one-way line configurations, used by the technicians to easily identify flow of electricity",
        "Enhanced my knowledge and skills through hands-on experience and close collaboration with technicians during on-site visits, providing me with a deep understanding of TNB's power distribution system"
      ]
    }
  ],
  "Projects": [
    {
      "Title": "IoT-based Tracking System and Health Monitoring for the Elderly with Alzheimer",
      "Description": [
        "Developed an IoT wearable device prototype using C++ language in VSCode for ESP32 microcontroller programming, making use of libraries and resources for sensor integration and enhanced functionality",
        "Integrated the ThingSpeak cloud platform to streamlined data storage and analysis in CSV format, optimizing data analysis visualization and ease of use, while using the cloud platform for real-time abnormal data detection",
        "Implemented an alert system that connects ThingSpeak cloud with the Telegram bot API, to notify the user and send the alert along with the data and location of the wearer triggeredit to the user/caretaker through Telegram."
      ]
    },
    {
      "Title": "Smart Irrigation System using PIC18 Microcontroller",
      "Description": [
        "Designed a system incorporating multiple moisture sensors and pump areas to operate simultaneously by utilizing polled input for efficient irrigation management in Assembly language using MPLAB software",
        "Implemented interrupt service routines and analog-to-digital converters I/O with appropriate voltage resolution for accurate moisture detection and to ensure smooth flow of the microcontroller’s program"
      ]
    },
    {
      "Title": "Mitsubishi RV-M1 Robotic Armbased Mini-Production Line",
      "Description": [
        "Object rejection used servomotor, IR sensor and Arduino on the production line using Arduino IDE software",
        "Mitsubishi Robot Arm was programmed to pick up the accepted object and place it into the designated place"
      ]
    }
  ],
  "Skills": {
    "Programming Languages": [
      "C",
      "C++",
      "Python",
      "Assembly",
      "MySQL",
      "MATLAB"
    ],
    "Hardware": [
      "Oscilloscope",
      "Signal Wave Generator",
      "Arduino",
      "ESP32",
      "PIC18 Microcontroller"
    ],
    "Design Programs": [
      "AutoCAD",
      "Proteus",
      "Simulink",
      "LTSpice",
      "Microwind",
      "MPLAB",
      "MikroC Language Proficiency"
    ],
    "Language Proficiency": [
      "Malay (Native)",
      "English (Fluent)"
    ]
  },
  "Certifications and Achievements": [
    {
      "Title": "AI & Machine Learning Competence for Industry 4.0 Certification",
      "Organization": "SHRDCExp",
      "Date": "Mar.2024"
    },
    {
      "Title": "Volunteer for Community Service with 3ESC",
      "Organization": "In collaboration with I-WELD",
      "Date": "Dec.2022"
    },
    {
      "Title": "English Trainer role for \"Soar with Reading and Writing\" Program",
      "Organization": "3ESC",
      "Date": "Aug.2017"
    }
  ]
}
        """
    },
    {
        "role": "user",
        "parts": f"Here is my resume: {extracted_resume}"
    }
    
    ]
    response = model.generate_content(contextResume)
    return response


st.set_page_config(
    page_title='YP AI-08 GPT Streamlit',
    page_icon = ''
)
col1, col2, col3 = st.columns(3)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdf_file as f:
        reader = PdfReader(f)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            text += reader.pages[page_num].extract_text()
    return text


def main():
    
    if "education_saved" not in st.session_state:
        st.session_state.education_saved = ""
    if "work_experience_saved" not in st.session_state:
        st.session_state.work_experience_saved = ""
    if "projects_saved" not in st.session_state:
        st.session_state.projects_saved = ""
    if "skills_saved" not in st.session_state:
        st.session_state.skills_saved = ""
        
    st.title("Input your details")


    


    education = st.text_area("Education:", value=st.session_state.education_saved)
    work_experience = st.text_area("Work Experience:", value=st.session_state.work_experience_saved)
    projects = st.text_area("Projects:", value=st.session_state.projects_saved)
    skills = st.text_area("Skills:", value=st.session_state.skills_saved)
    
    # Button to submit and save
    if st.button("Submit"):
            st.write("Your Details")
            st.write("### Education:")
            st.write(education)
            st.session_state.education_saved = education

            st.write("### Work Experience:")
            st.write(work_experience)
            st.session_state.work_experience_saved = work_experience

            st.write("### Projects:")
            st.write(projects)
            st.session_state.projects_saved = projects

            st.write("### Skills:")
            st.write(skills)
            st.session_state.skills_saved = skills

            st.sidebar.success('Information upload Success!')

    job_desc = st.text_area("Job Description:")
    submit_button = st.button('Generate AI opinion and summary')



    # Processing user input
    if job_desc and submit_button:
        with st.spinner("Generating opinion"):
            opinion_result = gemini_ai(st.session_state.education_saved, st.session_state.work_experience_saved, st.session_state.projects_saved, st.session_state.skills_saved, job_desc)
            
        st.write(opinion_result.text)


    

    st.title("Resume Scanner")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        st.text("Extracted Text:")
        st.write(text)
        summarized_result = gemini_ai_resume(text)
        st.write("Summarized:")
        st.write(summarized_result.text)


        json_string = summarized_result.text.strip('```').strip()
        json_string2 = summarized_result.text.strip('json').strip()
        print(json_string2)
        #print(summarized_result.text)
        data = json.loads(json_string2)

        # data = json.load(summarized_result.text)
        
        


        education = data["Education"]
        work_experience = data["Work Experience"]
        projects = data["Projects"]
        skills = data["Skills"]



        st.write("LAST BUT NOT LEAST JSON FIXED:")
        st.write(education)
        st.write(work_experience)
        st.write(projects)
        st.write(skills)


        





if __name__ == "__main__":
    main()
























# %%
