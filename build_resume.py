from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
out=r'C:\Users\USER\Downloads\covid-19-chest-x-ray-multiclass-classification\Paul_Ajibola_Kasi_Academy_Resume.docx'
d=Document(); sec=d.sections[0]; sec.top_margin=Inches(.52); sec.bottom_margin=Inches(.52); sec.left_margin=Inches(.68); sec.right_margin=Inches(.68)
styles=d.styles
styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(9); styles['Normal'].font.color.rgb=RGBColor(45,55,65)
styles['Normal'].paragraph_format.space_after=Pt(2)
for s in ['Heading 1']:
 st=styles[s]; st.font.name='Aptos Display'; st.font.size=Pt(11); st.font.bold=True; st.font.color.rgb=RGBColor(12,73,92); st.paragraph_format.space_before=Pt(6); st.paragraph_format.space_after=Pt(2)
p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('PAUL SEGUN AJIBOLA'); r.bold=True; r.font.size=Pt(19); r.font.color.rgb=RGBColor(12,73,92)
p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(5)
p.add_run('Lagos State, Nigeria  |  +234 813 913 2063  |  paulajibola6@gmail.com\n').font.size=Pt(8)
p.add_run('GitHub: github.com/Paul-Ajibola  |  LinkedIn: linkedin.com/in/paul-ajibola/  |  Portfolio: paul-ajibola.netlify.app').font.size=Pt(8)
def heading(t): d.add_heading(t,level=1)
def para(t): d.add_paragraph(t)
def bullet(t):
 p=d.add_paragraph(style='List Bullet'); p.paragraph_format.left_indent=Inches(.18); p.paragraph_format.first_line_indent=Inches(-.12); p.paragraph_format.space_after=Pt(1); p.add_run(t)
heading('PROFILE')
para('Early-career engineer and data professional with experience building reliable data workflows, machine learning systems, and AI products. Led a 12-person team digitizing historical clinical and epidemiological records into a relational database, implemented data quality checks, and used analytics to support disease surveillance. Brings a Doctor of Veterinary Medicine background, practical Python and SQL skills, and a strong interest in the systems and infrastructure that power digital services.')
heading('RELEVANT SKILLS')
para('Programming and data: Python, SQL, MongoDB, relational databases, data cleaning, transformation and validation, exploratory and statistical analysis, dashboards and visualization. Machine learning: PyTorch, scikit-learn, Pandas, NumPy, computer vision, time-series forecasting, clustering and ensemble methods. Delivery: Agile/Scrum, Jira, GitHub, product roadmapping, sprint planning, backlog prioritization, cross-functional coordination and technical presentations.')
heading('DATA AND TECHNICAL EXPERIENCE')
p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('AI Product Engineering Intern | HNG'); r.bold=True; p.add_run('  |  Late Sep–Oct 2026')
bullet('Collaborated on the development of amp.ziki.dev, including its Zedu tasks experience.')
bullet('Implemented fixes to the zedu.chat application.')
bullet('Built and deployed Turbo Race, a browser game hosted on start.ng: https://turbo-race.start.ng/')
p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('AI Backend Engineer | FlyRank AI, Remote'); r.bold=True; p.add_run('  |  Jun 2026–Present')
bullet('Built a project scraper (github.com/Paul-Ajibola/flyrank-ai-project-scrapper) and an authentication project (github.com/Paul-Ajibola/flyrank-project-auth), and implemented a basic CRUD application.')
bullet('Created a support triage agent as a capstone project and deployed it as a web application: https://support-triage-agent-suhw.onrender.com/')

p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('Data Scientist (Volunteer) | Centre for Control and Prevention of Zoonoses (CCPZ)'); r.bold=True
p.add_run('  |  Jan 2026–Present')
bullet('Led a 12-person team digitizing veterinary clinical and epidemiological records from 1977–2001 and organizing them in a relational database for efficient access and analysis.')
bullet('Set up automated data quality assurance checks to improve the reliability of data ingestion.')
bullet('Applied machine learning and spatial analysis to identify disease outbreak patterns and support public health decisions.')
bullet('Collaborated on testing and benchmarking a medical diagnostic device with a health technology partner.')
p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('Data Analyst, HNG 11 Finalist | HNG Tech, Remote'); r.bold=True; p.add_run('  |  Oct–Dec 2025')
bullet('Cleaned, transformed, and analyzed consumer application datasets to identify customer behavior patterns.')
bullet('Built interactive user behavior dashboards and presented analytical findings and forecast models to engineering and product teams during sprint planning.')

heading('PRODUCT MANAGEMENT EXPERIENCE')
p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('Product Manager (Contract) | Waterbrooks Technologies, Remote'); r.bold=True; p.add_run('  |  Aug–Dec 2025')
bullet('Coordinated engineering teams delivering data-informed product features and aligned product roadmaps with user telemetry to improve onboarding and delivery focus.')
p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('Product Manager | Spur Nation Group HQ, Remote'); r.bold=True; p.add_run('  |  Oct 2024–Dec 2025')
bullet('Managed the lifecycle of data-intensive enterprise platforms, aligning engineering, design, and market operations on feature planning and delivery.')
bullet('Set product priorities for the DryKlin mobile app MVP, using user feedback to guide requirements and collaborating with UI/UX and engineering through delivery.')
p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('Product Manager Intern | HNG Tech, Remote'); r.bold=True; p.add_run('  |  Jun–Aug 2024')
bullet('Contributed within a 20+ member product team to market research, product vision, strategy, and technical requirements for Delve, an AI-powered gamified language-learning MVP.')
bullet('Researched market opportunities and facilitated Agile delivery for the HNG Tech Boilerplate, authoring user stories and prioritizing backlog items for a standardized software development framework.')
bullet('Contributions supported a 45% reduction in iteration workflow and delivery of a functional, scalable Delve MVP; boilerplate work identified a 30% market gap and contributed to 15% less development time and 40% better delivery efficiency.')
heading('TECHNICAL TRAINING AND PROJECTS')

p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('Applied Deep Learning Lab | WorldQuant University'); r.bold=True; p.add_run('  |  Oct–Dec 2025')
bullet('Engineered 12 end-to-end projects using multilayer perceptrons, CNNs, recurrent networks (LSTM/GRU), autoencoders, and generative adversarial networks.')
p=d.add_paragraph(); p.paragraph_format.space_after=Pt(0); r=p.add_run('Applied Data Science Lab | WorldQuant University'); r.bold=True; p.add_run('  |  May–Sep 2025')
bullet('Built eight machine learning frameworks covering ARIMA/GARCH forecasting, PCA, K-means clustering, random forests, and gradient boosting.')
heading('EDUCATION & CERTIFICATIONS')
para('Doctor of Veterinary Medicine (DVM) | Feb 2016–Jun 2023')
para('Oracle Data Science Professional  |  ALX Artificial Intelligence Career Essentials  |  Machine Learning Foundations  |  WorldQuant University Applied Data Science and Deep Learning Lab')
# prevent split on key section headings
for p in d.paragraphs:
 if p.style.name.startswith('Heading'): p.paragraph_format.keep_with_next=True
# metadata
d.core_properties.title='Paul Segun Ajibola Resume - Kasi Academy Digital Infrastructure Internship'
d.core_properties.subject='Tailored resume for Kasi Academy digital infrastructure program'
d.save(out)
print(out)










