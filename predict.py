import joblib

def predict_postings(postings: list, loaded_model):
    pipeline = loaded_model['pipeline']
    threshold = loaded_model['threshold']
    prob = pipeline.predict_proba(postings)
    decision = prob[:,1] >= threshold
    
    return prob[:,1], decision


if __name__ == "__main__":
    
    job_row = [
    2,  # job_id
    "Customer Service - Cloud Video Production",  # title
    "NZ, , Auckland",  # location
    "Success",  # department
    "",  # salary_range
    # company_profile
    """90 Seconds, the worlds Cloud Video Production Service.90 Seconds is the worlds Cloud Video Production Service enabling brands and agencies to get high quality online video content shot and produced anywhere in the world. 90 Seconds makes video production fast, affordable, and all managed seamlessly in the cloud from purchase to publish. http://90#URL_fbe6559afac620a3cd2c22281f7b8d0eef56a73e3d9a311e2f1ca13d081dd630#90 Seconds removes the hassle, cost, risk and speed issues of working with regular video production companies by managing every aspect of video projects in a beautiful online experience. With a growing global network of over 2,000 rated video professionals in over 50 countries managed by dedicated production success teams in 5 countries, 90 Seconds provides a 100% success guarantee.90 Seconds has produced almost 4,000 videos in over 30 Countries for over 500 Global brands including some of the worlds largest including Paypal, L’Oreal, Sony and Barclays and has offices in Auckland, London, Sydney, Tokyo and Singapore.http://90#URL_fbe6559afac620a3cd2c22281f7b8d0eef56a73e3d9a311e2f1ca13d081dd630# | http://90#URL_e2ad0bde3f09a0913a486abdbb1e6ac373bb3310f64b1fbcf550049bcba4a17b# | http://90#URL_8c5dd1806f97ab90876d9daebeb430f682dbc87e2f01549b47e96c7bff2ea17e# """,
    # description
    """Organised - Focused - Vibrant - Awesome!Do you have a passion for customer service? Slick typing skills? Maybe Account Management? ...And think administration is cooler than a polar bear on a jetski? Then we need to hear you! We are the Cloud Video Production Service and opperating on a glodal level. Yeah, it's pretty cool. Serious about delivering a world class product and excellent customer service.Our rapidly expanding business is looking for a talented Project Manager to manage the successful delivery of video projects, manage client communications and drive the production process. Work with some of the coolest brands on the planet and learn from a global team that are representing NZ is a huge way!We are entering the next growth stage of our business and growing quickly internationally.  Therefore, the position is bursting with opportunity for the right person entering the business at the right time. 90 Seconds, the worlds Cloud Video Production Service - http://90#URL_fbe6559afac620a3cd2c22281f7b8d0eef56a73e3d9a311e2f1ca13d081dd630#90 Seconds is the worlds Cloud Video Production Service enabling brands and agencies to get high quality online video content shot and produced anywhere in the world. Fast, affordable, and all managed seamlessly in the cloud from purchase to publish. 90 Seconds removes the hassle, cost, risk and speed issues of working with regular video production companies by managing every aspect of video projects in a beautiful online experience.  With a growing network of over 2,000 rated video professionals in over 50 countries and dedicated production success teams in 5 countries guaranteeing video project success 100%. It's as easy as commissioning a quick google adwords campaign.90 Seconds has produced almost 4,000 videos in over 30 Countries for over 500 Global brands including some of the worlds largest including Paypal, L'oreal, Sony and Barclays and has offices in Auckland, London, Sydney, Tokyo &amp; Singapore.Our Auckland office is based right in the heart of the Wynyard Quarter Innovation Precinct - GridAKL! """,
    # requirements
    """What we expect from you:Your key responsibility will be to communicate with the client, 90 Seconds team and freelance community throughout the video production process including, shoot planning, securing freelance talent, managing workflow and the online production management system.  The aim is to manage each video project effectively so that we produce great videos that our clients love.Key attributesClient focused - excellent customer service and communication skillsOnline - oustanding computer knowledge and experience using online software and project management toolsOrganised - manage workload and able to multi-task100% attention to detailMotivated - self-starter with a passion for doing excellent work and achieving great resultsAdaptable - show initiative and think on your feet as this is a constantly evolving atmosphereFlexible - fast turnaround work and after hours availabilityEasy going &amp; upbeat - dosen't get bogged down and loves the challengeSense of Humour - have a laugh and know that working in a startup takes guts!Ability to deliver - including meeting project deadlines and budgetAttitude is more important than experience at 90 Seconds, however previous experience in customer service and/or project management is beneficialPlease view our platform / website at #URL_395a8683a907ce95f49a12fb240e6e47ad8d5a4f96d07ebbd869c4dd4dea1826# and get a clear understand about what we do before reaching out.""",
    # benefits
    """What you will get from usThrough being part of the 90 Seconds team you will gain:experience working on projects located around the world with an international brandexperience working with a variety of clients and on a large range of projectsopportunity to drive and grow production function and teama positive working environment with a great teamPay$40,000-$55,000Applying for this role with a VIDEOBeing a video business, we understand that one of the quickest ways that we can assess your suitability for this role, and one of the quickest ways that you can apply for it, is for you to submit a 60-90 second long video telling us about yourself, your experience and why you think you would be perfect for the role.  It’s not about being a filmmaker or making a really creative video.  A simple video filmed with a smart phone or web cam will be fine. Please also include where you are based and when you can start.You can upload the video onto YouTube or Vimeo (or similar) as a Draft or Live link.APPLICATIONS DUE by 5pm on Wednesday 18th July 2014 - Once you have a video ready, apply for this role via the following link together with a cover letter and your CV. After we have watched your video and get an idea of your suitability for the role, we will email the shortlisted candidates""",
    0,  # telecommuting
    1,  # has_company_logo
    0,  # has_questions
    "Full-time",  # employment_type
    "Not Applicable",  # required_experience
    "",  # required_education
    "Marketing and Advertising",  # industry
    "Customer Service",  # function
    0,  # fraudulent
    ]
    
    job_row_2 = [
    3,  # job_id
    "Commissioning Machinery Assistant (CMA)",  # title
    "US, IA, Wever",  # location
    "",  # department
    "",  # salary_range

    # company_profile
    """Valor Services provides Workforce Solutions that meet the needs of companies across the Private Sector, with a special focus on the Oil &amp; Gas Industry. Valor Services will be involved with you throughout every step of the hiring process and remain in contact with you all the way through the final step of signing of the employment contract with your new employer. Valor Services was founded with the vision of employing the unique skills, experiences, and qualities of America’s finest veterans to provide Private Sector companies with precise and concerted value-added services – and America’s finest Veterans with an optimized career opportunity.We are eager to get the word out to veterans that there are ample opportunities for employment in the private sector and that you are the ideal candidates to fill those positions. Valor Services Your Success is Our Mission. ™ """,

    # description
    """Our client, located in Houston, is actively seeking an experienced Commissioning Machinery Assistant that possesses strong supervisory skills and has an attention to detail. A strong dedication to safety is a must. The ideal candidate will execute all activities while complying with quality requirements and health, environmental, and safety regulations.""",

    # requirements
    """Implement pre-commissioning and commissioning procedures for rotary equipment.Execute all activities with subcontractor’s assigned crew that pertains to the discipline.Ensure effective utilization of commissioning manpower and consumables.Ensure the execution of vendor specialists' field activities with the assigned resources from the sub-contractor per vendor’s representative plans.Carry out equipment inspections with client representatives and ensure proper certification is produced.Prepare forms for all pending tests and submit signed certificates for final hand over to the certification engineer for QA and QC.Coordinate in the field with vendor representatives.Keep records of all activities.Ensure that safety practices are strictly followed during the execution of activities.Report progress and constraints to the mechanical supervisor.Possible authorization by site manager to receive or issue a Permit To Work according to project Permit To Work procedures.Assist supervisor to expedite pending punch-list items in accordance with the commissioning manager’s priorities.Assist supervisor to coordinate and supervise construction-support activities during pre-commissioning and commissioning activities.Company Overview:Our client is a premiere engineering, construction, and procurement company that executes large-scale projects internationally.""",

    "",  # benefits
    0,  # telecommuting
    1,  # has_company_logo
    0,  # has_questions
    "",  # employment_type
    "",  # required_experience
    "",  # required_education
    "",  # industry
    "",  # function
    0,  # fraudulent
    ]
    
    text_fields_1 = [job_row[1], job_row[5], job_row[6], job_row[7], job_row[8]]
    text_fields_2 = [job_row_2[1], job_row_2[5], job_row_2[6], job_row_2[7], job_row_2[8]]
    postings = [" ".join(text_fields_1), " ".join(text_fields_2)]
    loaded_model = joblib.load("fake_job_detector.joblib")
    
    prob, decision = predict_postings(postings, loaded_model)
    print(prob)
    print(decision)
    