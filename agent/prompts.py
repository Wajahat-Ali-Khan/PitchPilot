from langchain.prompts import PromptTemplate

extract_prompt = PromptTemplate(
    input_variables=["idea"],
    template=(
        "You are a structured extraction assistant. Given the startup idea below, return a JSON object ONLY "
        "with keys: category, problem, solution. Keep values short (one or two sentences each). "
        "Startup idea: {idea}\n\nRespond with JSON only."
    ),
)

market_research_prompt = PromptTemplate(
    input_variables=["category", "idea"],
    template=(
        "You are a concise market researcher. For the startup category: '{category}' and idea: '{idea}', "
        "return a JSON object ONLY with keys: market_size_estimate, key_trends (list), top_competitors (list, short names). "
        "market_size_estimate: short string with a numeric estimate and region (e.g., 'Global ~$X billion'). "
        "key_trends: array of 3 short trend bullets. top_competitors: array of up to 5 names. Respond with JSON only."
    ),
)

business_model_prompt = PromptTemplate(
    input_variables=["category", "problem", "solution", "market_summary"],
    template=(
        "You are a startup strategist. Using these inputs, suggest 3 concise business model options. "
        "Return JSON ONLY with key business_models which is a list of objects {name, one_line_description, revenue_streams (list)}.\n\n"
        "category: {category}\nproblem: {problem}\nsolution: {solution}\nmarket_summary: {market_summary}"
    ),
)

slides_prompt = PromptTemplate(
    input_variables=["category", "problem", "solution", "market_summary", "business_models"],
    template=(
        "Generate 8-10 investor-ready pitch deck slides as a JSON array. Each slide should be an object with keys: title, content. "
        "Titles should be classic deck titles (e.g., 'Problem', 'Solution', 'Market', 'Business Model', 'Competition', 'Go-to-Market', 'Traction', 'Financials', 'Team'). "
        "Content should be concise, investor-ready bullet paragraphs (2-6 lines). Use the inputs: category: {category}; problem: {problem}; solution: {solution}; "
        "market_summary: {market_summary}; business_models: {business_models}.\n\nReturn JSON array only."
    ),
)
