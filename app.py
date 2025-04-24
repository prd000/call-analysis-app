import os
import flask
from dotenv import load_dotenv
import google.generativeai as genai
from pyairtable import Api
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Configure Flask app
app = flask.Flask(__name__)

# --- Configuration --- 
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Airtable Column Names (Update if needed) ---
OUTCOME_COLUMN = 'Call Outcome'
REP_COLUMN = 'Rep(s)'
CALL_TYPE_COLUMN = 'Call Type'
TRANSCRIPT_COLUMN = 'JSON Transcript'

# --- Error Handling for Config --- 
if not all([AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# --- Initialize Services ---
try:
    airtable_api = Api(AIRTABLE_API_KEY)
    airtable_table = airtable_api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
except Exception as e:
    print(f"Error initializing Airtable API: {e}") 
    airtable_table = None # Ensure it's None if init fails

try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') 
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_model = None # Ensure it's None if init fails


def build_airtable_formula(outcome_filter, reps_filter, call_type_filter):
    """Builds an Airtable filter formula string based on provided filters."""
    filters = []
    # --- Handle Single Select Filters --- 
    if outcome_filter:
        safe_outcome = outcome_filter.replace("'", "\\'")
        filters.append(f"{{{OUTCOME_COLUMN}}} = '{safe_outcome}'") 
    if call_type_filter:
        safe_call_type = call_type_filter.replace("'", "\\'")
        filters.append(f"{{{CALL_TYPE_COLUMN}}} = '{safe_call_type}'")

    # --- Handle Reps Multi-Select Filter --- 
    rep_conditions = []
    if isinstance(reps_filter, list) and len(reps_filter) > 0:
        for rep in reps_filter:
            if rep: # Ensure rep string is not empty
                safe_rep = rep.strip().replace("'", "\\'")
                rep_conditions.append(f"FIND('{safe_rep}', ARRAYJOIN({{{REP_COLUMN}}})) > 0")
        
        if len(rep_conditions) == 1:
            filters.append(rep_conditions[0])
        elif len(rep_conditions) > 1:
            filters.append(f"OR({ ', '.join(rep_conditions) })")
    
    # --- Combine all filters with AND --- 
    if not filters:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return f"AND({ ', '.join(filters) })"

def fetch_and_process_airtable(formula, n_records):
    """Fetches data using a formula, processes it, and returns a sample DataFrame."""
    if not airtable_table:
        raise ConnectionError("Airtable client not initialized.")
        
    required_fields = list(set([TRANSCRIPT_COLUMN, OUTCOME_COLUMN, REP_COLUMN, CALL_TYPE_COLUMN]))
    all_records = airtable_table.all(fields=required_fields, formula=formula)
    
    if not all_records:
        return pd.DataFrame(), "No records found matching filters."

    data = [{'transcript': r['fields'].get(TRANSCRIPT_COLUMN),
             'outcome': r['fields'].get(OUTCOME_COLUMN),
             'rep': r['fields'].get(REP_COLUMN), 
             'call_type': r['fields'].get(CALL_TYPE_COLUMN)} 
            for r in all_records if 'fields' in r and r['fields'].get(TRANSCRIPT_COLUMN)]
    
    if not data:
        return pd.DataFrame(), "No records with transcripts found after filtering."

    df = pd.DataFrame(data)
    df.dropna(subset=['transcript'], inplace=True) # Only drop if transcript is missing

    if df.empty:
         return pd.DataFrame(), "No valid records with transcripts found after cleaning."

    # Select the requested number of records (0 means all)
    if n_records > 0:
         if len(df) > n_records:
            df_sample = df.tail(n_records)
         else:
            df_sample = df 
    else: 
        df_sample = df 
        
    return df_sample, f"{len(df_sample)} records fetched."

# --- Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    return flask.render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the analysis request, potentially comparing two sets of filters."""
    if not gemini_model:
         return flask.jsonify({"error": "Gemini client not initialized. Check API key and logs."}), 500
    if not airtable_table:
         return flask.jsonify({"error": "Airtable client not initialized. Check API key/Base ID/Table Name and logs."}), 500

    try:
        request_data = flask.request.json
        user_request = request_data.get('prompt', '')
        # Get Counts for A and B
        transcript_count_a = request_data.get('transcript_count_a', 0) 
        transcript_count_b = request_data.get('transcript_count_b', 0) 
        # Get Filters for Set A
        filter_a_outcome = request_data.get('filter_a_outcome', '')
        filter_a_reps = request_data.get('filter_a_rep', [])     
        filter_a_call_type = request_data.get('filter_a_call_type', '')
        # Get Filters for Set B
        filter_b_outcome = request_data.get('filter_b_outcome', '')
        filter_b_reps = request_data.get('filter_b_rep', [])      
        filter_b_call_type = request_data.get('filter_b_call_type', '')

        if not user_request:
            return flask.jsonify({"error": "No prompt provided"}), 400
            
        # Validate transcript counts
        try:
            n_transcripts_a = int(transcript_count_a) if transcript_count_a is not None else 0
            if n_transcripts_a < 0:
                n_transcripts_a = 0
            n_transcripts_b = int(transcript_count_b) if transcript_count_b is not None else 0
            if n_transcripts_b < 0:
                n_transcripts_b = 0
        except (ValueError, TypeError):
             return flask.jsonify({"error": "Invalid value provided for transcript count."}), 400

        # --- Build Formulas and Fetch Data --- 
        formula_a = build_airtable_formula(filter_a_outcome, filter_a_reps, filter_a_call_type)
        formula_b = build_airtable_formula(filter_b_outcome, filter_b_reps, filter_b_call_type)

        print(f"Set A Formula: {formula_a}")
        print(f"Set B Formula: {formula_b}")
        
        try:
             # Pass the specific count for Set A
             df_sample_a, status_a = fetch_and_process_airtable(formula_a, n_transcripts_a)
             
             # Determine if Set B is active (at least one filter is set)
             is_set_b_active = bool(filter_b_outcome or filter_b_reps or filter_b_call_type)
             df_sample_b = pd.DataFrame() # Initialize empty
             status_b = "Set B filters not active."

             if is_set_b_active:
                 # Pass the specific count for Set B
                 df_sample_b, status_b = fetch_and_process_airtable(formula_b, n_transcripts_b)
             
             print(f"Set A Status: {status_a}")
             print(f"Set B Status: {status_b}")

             if df_sample_a.empty and (not is_set_b_active or df_sample_b.empty):
                  return flask.jsonify({"error": "No records found matching the specified filters for either Set A or Set B."}), 500
             elif df_sample_a.empty and is_set_b_active:
                 # Proceed with only Set B if Set A is empty
                 print("Warning: Set A returned no data. Proceeding with Set B only.")
                 # (Consider if the prompt needs adjustment here)
                 pass 
             elif df_sample_b.empty and is_set_b_active:
                 # Proceed with only Set A if Set B was active but returned no data
                 print("Warning: Set B filters were active but returned no data. Proceeding with Set A only.")
                 is_set_b_active = False # Treat as single set analysis now

        except Exception as e:
            print(f"Error fetching/processing Airtable data: {e}")
            return flask.jsonify({"error": f"Failed to retrieve or process data from Airtable: {e}"}), 500

        # --- Prepare data and construct prompt for Gemini --- 
        
        # Helper to format data for prompt
        def format_data_for_prompt(df):
             return "\n\n".join([f"Transcript: {row['transcript']}\nOutcome: {row['outcome']}" 
                                for index, row in df.iterrows()])
        
        # Helper to describe filters
        def describe_filters(outcome, reps, call_type):
            desc = []
            if outcome: desc.append(f"Outcome='{outcome}'")
            if reps: desc.append(f"Rep(s)='{', '.join(reps)}'")
            if call_type: desc.append(f"Type='{call_type}'")
            return ', '.join(desc) if desc else "None"

        sample_data_text_a = format_data_for_prompt(df_sample_a)
        filters_desc_a = describe_filters(filter_a_outcome, filter_a_reps, filter_a_call_type)

        # Construct the prompt based on whether Set B is active
        if is_set_b_active and not df_sample_b.empty:
            sample_data_text_b = format_data_for_prompt(df_sample_b)
            filters_desc_b = describe_filters(filter_b_outcome, filter_b_reps, filter_b_call_type)
            
            prompt_for_gemini = f"""
            Analyze and compare the two sets of sales call transcripts (Set A and Set B) provided below. 
            User Request: {user_request}
            
            Focus on the user's request, highlighting differences and similarities between the two sets based on their content and outcomes.
            Be specific in your findings and clearly label insights related to Set A vs Set B.

            --- SET A --- 
            Filters Used: {filters_desc_a}
            Data ({len(df_sample_a)} records):
            {sample_data_text_a}
            
            --- SET B ---
            Filters Used: {filters_desc_b}
            Data ({len(df_sample_b)} records):
            {sample_data_text_b}
            --- END SETS ---
            
            Comparison Analysis:
            """
        else:
            # Standard single-set analysis prompt
            prompt_for_gemini = f"""
            Analyze the following sales call transcripts and their associated outcomes (Set A).
            User Request: {user_request}
            
            Filters Used for Set A: {filters_desc_a}

            Based on the data provided below ({len(df_sample_a)} records), identify specific phrases, keywords, topics, 
            or conversational patterns (from either the prospect or the sales rep) that 
            appear to correlate with different outcomes (e.g., Closed Won, Closed Lost). 
            
            Focus on finding actionable insights, like 'If X is said, Y outcome is more likely'.
            Be specific in your findings.

            Data:
            {sample_data_text_a}
            """
        
        # 3. Call Gemini API
        try:
            # Add safety settings if needed, e.g., to block harmful content
            # safety_settings = { ... }
            response = gemini_model.generate_content(prompt_for_gemini) #, safety_settings=safety_settings)
            analysis_result = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Consider specific error handling (e.g., RateLimitError, BlockedPromptException)
            return flask.jsonify({"error": f"Failed to get analysis from Gemini: {e}"}), 500

        # 4. Return result
        return flask.jsonify({"analysis": analysis_result})

    except Exception as e:
        # Catch-all for unexpected errors in the route
        print(f"Unexpected error in /analyze: {e}")
        return flask.jsonify({"error": "An unexpected server error occurred."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # ---- Configuration for Render (Production) ----
    # Use host='0.0.0.0' to make it accessible
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get("PORT", 5001)) 
    # When running with gunicorn, it handles workers/threading, debug should be False
    app.run(host='0.0.0.0', port=port, debug=False)
    # ---------------------------------------------

    # ---- Configuration for Local Development ----
    # Use a fixed port and enable debug mode
    # Comment this out and uncomment the Render config above for production!
    # app.run(host='0.0.0.0', port=5001, debug=True)
    # --------------------------------------------- 