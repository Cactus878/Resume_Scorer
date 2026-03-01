import streamlit as st
import tempfile
import pdfplumber
import rscorer as rs
import time

st.set_page_config(layout="wide")

def progress_callback(step: int, total_steps: int, message: str = ""):
    progress = step / total_steps
    progress_bar.progress(progress)
    status_text.text(message)

def display_matched_requirements(results):
    for score, requirement in results:
        if score >= 0.80:
            badge_class = "badge-green"
        elif score >= 0.70:
            badge_class = "badge-yellow"
        else:
            badge_class = "badge-red"

        st.markdown(
            f"""
            <div class="custom-badge {badge_class}">
                {round(score*100)}% | {requirement}
            </div>
            """,
            unsafe_allow_html=True
        )

@st.cache_data
def render_pdf_first_page(file_bytes):
    with pdfplumber.open(file_bytes) as pdf:
        page = pdf.pages[0]
        page_image = page.to_image(resolution=200)
        return page_image.original
    
st.markdown("""
<style>
    .custom-badge {
        display: block;
        padding: 10px 14px;
        font-size: 15px;
        border-radius: 10px;
        margin-bottom: 8px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.7)
    }

    .badge-green {
    background-color: #1E4620;
    color: #A6F4A6;
    }

    .badge-yellow {
        background-color: #4A3E0B;
        color: #FFEB99;
    }

    .badge-red {
        background-color: #601010;
        color: #FFB3B3;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    title = st.title("RESUME SCORER")

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    
    text_area = st.text_area("Paste Job Listing Here")

    button = st.button("Compare", width="stretch")

left, right = st.columns([0.3, 0.7])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
            
        with left:
            st.image(render_pdf_first_page(tmp_file_path), use_container_width=True)

if button:

    if uploaded_file == None:
        with st.sidebar:
            st.badge("No Valid Upload File", icon=":material/check:", color="red")

    if text_area == "":
        with st.sidebar:
            st.badge("No Pasted Job Listing", icon=":material/check:", color="red")
    
    if uploaded_file and text_area:
        with right:
            results_container = st.container(height=564)
            with results_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                (overall_similarity, job_listing_sections,
                aggregated_experience_final_score, best_experience_matches, 
                aggregated_education_final_score, best_education_matches, 
                aggregated_skills_final_score, best_skills_matches) = rs.compare(resume=tmp_file_path, job_listing=text_area, progress_callback=progress_callback)

                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

                experience_results = list(zip(best_experience_matches, job_listing_sections["EXPERIENCE"]))
                education_results = list(zip(best_education_matches, job_listing_sections["EDUCATION"]))
                skills_results = list(zip(best_skills_matches, job_listing_sections["SKILLS"]))

                st.header(f"""
                    {round(overall_similarity*100)}%
                """)
                st.subheader("Overall Requirements Match")

                # --- Required Experience ---
                with st.container(border=True):
                    st.subheader("Required Experience")
                    display_matched_requirements(experience_results)
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # --- Required Education ---
                with st.container(border=True):
                    st.subheader("Required Education")
                    display_matched_requirements(education_results)
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # --- Required Skills ---
                with st.container(border=True):
                    st.subheader("Required Skills")
                    display_matched_requirements(skills_results)
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    