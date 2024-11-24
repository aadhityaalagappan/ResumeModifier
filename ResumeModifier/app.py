import streamlit as st
from fastapi import HTTPException
from langchain_groq import ChatGroq
import PyPDF2
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize ChatGroq
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile",
)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def modify_resume(resume_text, job_description):
    try:
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPPED TEXT FROM WEBSITE:
            {job_description}
            ### INSTRUCTION:
            The scrapped text is from the career page of the website.
            Your job is to extract the job postings and return them in JSON format containing the following keys
            `role`, `experience`, `location`, `company`, `description`, `skills` from the entire text including basic qualifications, what you will bring, preferred qualifications.
            Only return the valid JSON
            ### VALID JSON(NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | llm
        response = chain_extract.invoke({"job_description": job_description})

        output_parser = JsonOutputParser()
        json_response = output_parser.parse(response.content)

        skills = json_response["skills"]

        prompt_template_resume = PromptTemplate(
            input_variables=["resume_text", "job_description", "skills"],
            template="""
            You are a professional resume writer. Your task is to tailor the following resume to align with the provided job description.

               ### Input:
                **Resume:**
               {resume_text}

                **Job Description:**
               {job_description}

               ### Instructions:
               Please modify the resume to better match the job description. Focus on the following:
               1. Highlight relevant skills and experiences that align with the job requirements.
               2. Use similar keywords and phrases from the job description in the resume.
               3. Reorder or emphasize sections to prioritize the most relevant information.

               4. Add a summary related to the job description at the start of the resume.
               5. Modify the skills section relevant to the job description. Please remove all the irrelevant skills specific to the job role and add the necessary skills related to the job description. Add the programming languages and other skills based on the job description and categorize it and remove unnecessary programming skills
               6. Dont add the End Note from the model
               7. Try to include all the technical programming skills mentioned in the job description into your resume
               8. Modify the projects and the paragraphs in the experience section to match with the job description
               9. Modify experience to match with the job description
               10. Increase the ATS Score
               11. Include all {skills} and use this skills in order to update the skills, experience and projects section

                ### Output:
               Provide the updated resume in a clean, professional format that aligns with industry standards. Focus on clarity and relevance.

               Please modify and format the resume to better match the job description. Use the following structure for your output:

               ---
           **[Your Name]**  
           📞 **Phone:** [Your Phone Number] | **Location:** [City, State]  
           📧 **Email:** [Your Email Address]  
           🔗 **LinkedIn:** [Your LinkedIn URL]  
           💻 **GitHub:** [Your GitHub URL]  

           ---

           ### **Summary**  
           [Provide a concise and professional summary of skills and experience aligned to the job description.]

           ---

           ### **Education**  
           - **[Degree and Major]**  
           [University Name], [City, State]  
           *[Start Date - End Date]*  

           ---

           ### **Skills**  
           [Use bullet points to categorize and list relevant skills.]

           ---

           ### **Experience**  
           #### **[Job Title]**  
           *[Company Name, Location]*  
           *[Start Date – End Date]*  
           - [Responsibility/achievement bullet point 1]  
           - [Responsibility/achievement bullet point 2]  
           - [Responsibility/achievement bullet point 3]  

           ---

           ### **Projects**  
           - **[Project Name]**  
            [Brief description of the project, technologies used, and outcomes.]  

           ---

           ### **Achievements**  
           - **[Achievement Title]**, [Event Name]  
           *[Date]*  

           ---

           ### Output:
           Provide the updated and formatted resume following the structure above. Ensure the formatting is professional, consistent, and adheres to industry standards.

            """
        )

        chain_jd = prompt_template_resume | llm
        response_resume = chain_jd.invoke(
            {
                "resume_text": resume_text,
                "job_description": job_description,
                "skills": skills,
            }
        )
        return response_resume.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit UI
st.title("Resume Modifier")

st.header("Upload your Resume")
uploaded_file = st.file_uploader("Choose Resume", type=["pdf"])

st.header("Enter Job Description")
job_desc = st.text_area("Job Description", height=300)

if st.button("Generate Resume"):
    if uploaded_file and job_desc:
        with st.spinner("Processing..."):
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                modified_resume = modify_resume(resume_text, job_desc)

                st.success("Resume modified successfully!")
                st.markdown("---")
                st.header("Modified Resume")
                st.markdown(modified_resume, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
    else:
        st.error("Please upload a resume and provide a job description.")