import streamlit as st
import pandas as pd
import joblib

# Set the page configuration
st.set_page_config(page_title="Loan Prediction Chatbot", layout="centered")

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Initialize session state variables to keep track of user input
if 'step' not in st.session_state:
    st.session_state['step'] = 0
    st.session_state['user_data'] = {}
    st.session_state['messages'] = []

def next_step():
    st.session_state['step'] += 1

def reset():
    st.session_state['step'] = 0
    st.session_state['user_data'] = {}
    st.session_state['messages'] = []

# Define the conversational flow
steps = [
    {"text": "Please enter your ID:", "key": "ID", "type": "text_input", "validation": lambda x: x.isdigit(), "error": "ID must be a number."},
    {"text": "Please enter your age:", "key": "Age", "type": "number_input", "params": {"min_value": 16, "max_value": 130, "value": 25}},
    {"text": "Please enter your gender (M/F/O):", "key": "Gender", "type": "text_input", "validation": lambda x: x in ["M", "F", "O"], "error": "Gender must be M, F, or O."},
    {"text": "Please enter your experience (years):", "key": "Experience", "type": "number_input", "params": {"min_value": 0, "max_value": 50, "value": 1}},
    {"text": "Please enter your income (in 1000s) per annum:", "key": "Income", "type": "number_input", "params": {"min_value": 0, "max_value": 500, "value": 49}},
    {"text": "Please enter your ZIP Code:", "key": "ZIP Code", "type": "text_input", "validation": lambda x: x.isdigit() and len(x) == 5, "error": "ZIP Code must be a 5-digit number."},
    {"text": "Please select your family size:", "key": "Family", "type": "text_input", "validation": lambda x: x.isdigit() and 1 <= int(x) <= 4, "error": "Family size must be between 1 and 4."},
    {"text": "Please enter your CCAvg (Credit Card Average Usage):", "key": "CCAvg", "type": "number_input", "params": {"min_value": 0.0, "max_value": 20.0, "value": 1.6}},
    {"text": "Please select your education level (1/2/3):", "key": "Education", "type": "text_input", "validation": lambda x: x in ["1", "2", "3"], "error": "Education level must be 1, 2, or 3."},
    {"text": "Please enter your mortgage value:", "key": "Mortgage", "type": "number_input", "params": {"min_value": 0, "max_value": 1000, "value": 0}},
    {"text": "Please select your home ownership status (Home Owner/Rent/Home Mortgage):", "key": "Home Ownership", "type": "text_input", "validation": lambda x: x in ["Home Owner", "Rent", "Home Mortgage"], "error": "Home Ownership must be Home Owner, Rent, or Home Mortgage."},
    {"text": "Do you have a securities account? (0 for No, 1 for Yes):", "key": "Securities Account", "type": "text_input", "validation": lambda x: x in ["0", "1"], "error": "Securities Account must be 0 or 1."},
    {"text": "Do you have a CD account? (0 for No, 1 for Yes):", "key": "CD Account", "type": "text_input", "validation": lambda x: x in ["0", "1"], "error": "CD Account must be 0 or 1."},
    {"text": "Do you use online banking? (0 for No, 1 for Yes):", "key": "Online", "type": "text_input", "validation": lambda x: x in ["0", "1"], "error": "Online Banking must be 0 or 1."},
    {"text": "Do you have a credit card? (0 for No, 1 for Yes):", "key": "CreditCard", "type": "text_input", "validation": lambda x: x in ["0", "1"], "error": "Credit Card must be 0 or 1."},
]

# Mapping dictionaries for categorical variables
gender_mapping = {'M': 1, 'F': 2, 'O': 3}
home_ownership_mapping = {'Home Owner': 1, 'Rent': 2, 'Home Mortgage': 3}

st.title("Loan Acceptance Prediction Chatbot")
st.markdown("""
**Welcome to the Loan Acceptance Prediction Chatbot.**

This chatbot will guide you through the process of providing the necessary information to predict the likelihood of your loan application being accepted. Please fill in the following details accurately to get the most reliable prediction.

Here's what you'll need to provide:
- Your personal details (e.g., Age, Gender, Experience)
- Financial information (e.g., Income, CCAvg, Mortgage)
- Additional details (e.g., Home Ownership, Securities Account, CD Account, etc.)

Please note that '0' stands for 'No' and '1' stands for 'Yes'.

Once you have entered all the required information, click the 'Predict' button to see the prediction result.
""")
# Conversational flow
if st.session_state['step'] < len(steps):
    current_step = steps[st.session_state['step']]
    
    # Display previous messages
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.write(message['text'])
    
    # Display current question
    with st.chat_message("bot"):
        st.write(current_step["text"])
    
    user_input = None
    if current_step["type"] == "text_input":
        user_input = st.text_input("Your response:", key=current_step["key"])
    elif current_step["type"] == "number_input":
        user_input = st.number_input("Your response:", **current_step["params"], key=current_step["key"])
    
    if st.button("Next"):
        if user_input is not None and str(user_input) != "":
            # Validate user input
            if "validation" in current_step and not current_step["validation"](user_input):
                st.session_state['messages'].append({"role": "bot", "text": current_step["error"]})
            else:
                st.session_state['user_data'][current_step["key"]] = user_input
                st.session_state['messages'].append({"role": "user", "text": str(user_input)})
                next_step()
                st.experimental_rerun()
        else:
            st.session_state['messages'].append({"role": "bot", "text": "Please provide a valid response."})

else:
    # Display previous messages
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.write(message['text'])

    with st.chat_message("bot"):
        st.write("Here is the summary of your inputs:")
    user_data = st.session_state['user_data']
    st.write(user_data)
    
    user_data['Gender'] = gender_mapping.get(user_data['Gender'], 3)
    user_data['Home Ownership'] = home_ownership_mapping.get(user_data['Home Ownership'], 3)

    # Convert user_data to DataFrame
    user_input_df = pd.DataFrame(user_data, index=[0])
    
    # Make predictions
    if st.button("Predict"):
        prediction = model.predict(user_input_df.drop(columns=['ID', 'ZIP Code']))[0]
        st.subheader('Prediction Result')
        with st.chat_message("bot"):
            if prediction == 0:
                st.write("The loan is likely to be rejected.")
            elif prediction == 1:
                st.write("The loan is likely to be accepted.")
            else:
                st.write("Error")
    
    # Optionally display the DataFrame for debugging
    st.write(user_input_df)

# Reset the chatbot
if st.button("Start Over"):
    reset()
    st.experimental_rerun()
