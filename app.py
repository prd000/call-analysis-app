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

# --- Error Handling for Config --- 
if not all([AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# --- Initialize Services ---
try:
    airtable_api = Api(AIRTABLE_API_KEY)
    airtable_table = airtable_api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
except Exception as e:
    # Handle Airtable connection errors during startup if necessary
    # For now, we let it fail loudly if keys are wrong or service is down
    print(f"Error initializing Airtable API: {e}") 
    # Depending on the desired behaviour, you might exit or disable Airtable features.

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use the specific experimental model identifier requested by the user
    gemini_model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25') 
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # Handle Gemini configuration errors

# --- Routes ---
@app.route('/')
def index():
    """Renders the main page."""
    return flask.render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the analysis request."""
    try:
        request_data = flask.request.json
        user_request = request_data.get('prompt', '')
        transcript_count = request_data.get('transcript_count', 0) # Default to 0 (All) if not provided
        filter_outcome = request_data.get('filter_outcome', '') # String from single select
        filter_reps = request_data.get('filter_rep', [])      # Array from multi-select
        filter_call_type = request_data.get('filter_call_type', '') # String from single select

        if not user_request:
            return flask.jsonify({"error": "No prompt provided"}), 400
            
        # Validate transcript_count
        try:
            # Treat negative counts as 0 (All)
            n_transcripts = int(transcript_count) if transcript_count is not None else 0
            if n_transcripts < 0:
                n_transcripts = 0
        except (ValueError, TypeError):
             return flask.jsonify({"error": "Invalid value provided for transcript count."}), 400

        # ---- Build Airtable Filter Formula ----
        # Use the exact column names provided by the user
        outcome_column_name = 'Call Outcome' 
        rep_column_name = 'Rep(s)' # Multiple Select field
        call_type_column_name = 'Call Type' # Single Select field
        transcript_column_name = 'JSON Transcript' # Assuming this is still correct
        
        filters = []
        # --- Handle Single Select Filters --- 
        if filter_outcome:
            safe_outcome = filter_outcome.replace("'", "\\'")
            filters.append(f"{{{outcome_column_name}}} = '{safe_outcome}'") 
        if filter_call_type:
            safe_call_type = filter_call_type.replace("'", "\\'")
            filters.append(f"{{{call_type_column_name}}} = '{safe_call_type}'")

        # --- Handle Reps Multi-Select Filter --- 
        rep_conditions = []
        if isinstance(filter_reps, list) and len(filter_reps) > 0:
            for rep in filter_reps:
                if rep: # Ensure rep string is not empty
                    safe_rep = rep.strip().replace("'", "\\'")
                    # Formula for Multiple Select: Check if the value exists in the field
                    rep_conditions.append(f"FIND('{safe_rep}', ARRAYJOIN({{{rep_column_name}}})) > 0")
            
            if len(rep_conditions) == 1:
                filters.append(rep_conditions[0])
            elif len(rep_conditions) > 1:
                 # Combine multiple rep conditions with OR
                filters.append(f"OR({ ', '.join(rep_conditions) })")
        
        # --- Combine all filters with AND --- 
        airtable_formula = None
        if len(filters) == 1:
            airtable_formula = filters[0]
        elif len(filters) > 1:
            # Combine outcome/type/rep filters with AND
            airtable_formula = f"AND({ ', '.join(filters) })"
        
        print(f"Using Airtable filter formula: {airtable_formula}") # For debugging
        # ----------------------------------------

        # 1. Fetch data from Airtable using the filter formula
        try:
            # Fetch records matching the formula
            # Only request the fields we absolutely need
            required_fields = [transcript_column_name, outcome_column_name, rep_column_name, call_type_column_name]
            # Remove duplicates just in case outcome/rep column is same as transcript
            required_fields = list(set(required_fields)) 
            
            all_records = airtable_table.all(fields=required_fields, formula=airtable_formula)
             
            # Note: Airtable returns ALL records if formula is None/empty
            #       or if the formula is invalid (it doesn't raise an error here usually)
            
            if not all_records:
                return flask.jsonify({"error": "No records found matching the specified filters."}), 500
            
            # Convert to Pandas DataFrame
            # Ensure we extract the correct fields now
            data = [{'transcript': r['fields'].get(transcript_column_name),
                     'outcome': r['fields'].get(outcome_column_name),
                     'rep': r['fields'].get(rep_column_name), # Extract rep field
                     'call_type': r['fields'].get(call_type_column_name)} # Extract call type field
                    for r in all_records if 'fields' in r and r['fields'].get(transcript_column_name)]
            
            if not data:
                 return flask.jsonify({"error": "No records with transcripts found."}), 500

            df = pd.DataFrame(data)
            df.dropna(inplace=True) # Remove rows where transcript or outcome might be missing

            if df.empty:
                 return flask.jsonify({"error": "No valid records with transcripts and outcomes found after cleaning."}), 500

            # Select the requested number of records (0 means all)
            if n_transcripts > 0:
                 if len(df) > n_transcripts:
                    df_sample = df.tail(n_transcripts)
                 else:
                    df_sample = df # Use all if less than requested number exist
            else: # n_transcripts is 0, use all records
                df_sample = df 

            if df_sample.empty:
                # This case might happen if n_transcripts > 0 but df was empty initially
                return flask.jsonify({"error": "No records available to sample."}), 500
            
        except Exception as e:
            print(f"Error fetching/processing Airtable data: {e}")
            return flask.jsonify({"error": f"Failed to retrieve or process data from Airtable: {e}"}), 500

        # 2. Prepare data and construct prompt for Gemini using the filtered and sampled records
        sample_data_text = "\n\n".join([f"Transcript: {row['transcript']}\nOutcome: {row['outcome']}" 
                                       for index, row in df_sample.iterrows()]) 
        
        prompt_for_gemini = f"""
        Analyze the following sales call transcripts and their associated outcomes. 
        
        Based on the data provided below, identify specific phrases, keywords, topics, 
        or conversational patterns (from either the prospect or the sales rep) that 
        appear to correlate with different outcomes (e.g., Closed Won, Closed Lost). 
        
        Focus on finding actionable insights, like 'If X is said, Y outcome is more likely'.
        Be specific and concise in your findings. No fluff, just insights.

        User Request: {user_request}
        
        Data:
        {sample_data_text}
        """

        # 3. Call Gemini API
        try:
            response = gemini_model.generate_content(prompt_for_gemini)
            analysis_result = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Consider more specific error handling based on Gemini API errors
            return flask.jsonify({"error": f"Failed to get analysis from Gemini: {e}"}), 500

        # 4. Return result
        return flask.jsonify({"analysis": analysis_result})

    except Exception as e:
        # Catch-all for unexpected errors in the route
        print(f"Unexpected error in /analyze: {e}")
        return flask.jsonify({"error": "An unexpected server error occurred."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible
    # Get port from environment variable (Render provides this) or default to 5001 for local dev
    port = int(os.environ.get("PORT", 5001))
    # When running with gunicorn, it handles workers/threading, debug should be False
    app.run(host='0.0.0.0', port=port, debug=False) 