import streamlit as st
session = st.session_state 

st.set_page_config(
    page_title="Kai Sen App",
    page_icon="🚄",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Insomniagung',
        'About': "Aplikasi Sentimen Analisis oleh Agung Gunawan."
    }
)

import time
import streamlit_authenticator as stauth
import base64

# --- import pages ---
import predictor as pr
from dash_admin.admin_home import admin_home_page
from dash_admin.admin_analysis import admin_analysis_page
from dash_admin.admin_report import admin_report_page
from dash_admin.admin_database import admin_database_page

from dash_user.user_home import user_home_page
from dash_user.user_database import user_database_page

from deta import Deta
#database
deta = Deta(st.secrets["DETA_PROJECT_KEY"])
db = deta.Base("users")

# def get_user(key):
#     return db.get(key)
def get_user(key):
    user_data = db.get(str(key))
    if user_data is not None:
        return user_data
    else:
        return {}

def put_user(username, name, password, role):
    return db.put({"key": username, "name": name, "password": password, "role": role})

def get_all_keys():
    users = db.fetch().items
    keys = [user["key"] for user in users]
    return keys

# alert
def red_alert(str):
    st.write("")
    with st.error(str):
        time.sleep(2)
        st.empty()

def green_alert(str): 
    st.write("")
    with st.success(str):
        time.sleep(2)
        st.empty()           

# --- css ---

def add_background():
    image_file = 'img/bg_images.png'
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
    
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #01080d;        
        border-radius:25px;
        padding:7px 18px 7px 18px;        
    }
    div.stButton > button:hover {
        background-color: #05263d;
        
    }
    </style>""", 
unsafe_allow_html=True)

def main():
    add_background()

    # --- authenticator ---    
    users = db.fetch().items
    usernames = [user["key"] for user in users]
    names = [users["name"] for users in users]
    passwords = [user["password"] for user in users]

    credentials = {"usernames":{}}

    for un, name, pw in zip(usernames, names, passwords):
        user_dict = {"name": name, "password": pw}
        credentials["usernames"].update({un: user_dict})

    authenticator = stauth.Authenticate(credentials, "login_cookie", "key_login", cookie_expiry_days=7)
    
    authenticator._check_cookie()
    if not session['authentication_status']:    
        menu = ["Login", "Register"]
        selected = st.radio(label = "&nbsp;&nbsp; **FORM** ", 
                                options = menu, 
                                index=0, 
                                horizontal=True,
                                label_visibility = "collapsed")
        
        if selected == "Login":
            name_auth, authentication_status, username_auth = authenticator.login("Login", "main")
            session['name'] = name_auth
            session['authentication_status'] = authentication_status
            session['username'] = username_auth
            
            if authentication_status == False:
                st.error("Username atau password salah. Mohon isi kolom dengan benar.")
            if authentication_status == None:
                st.warning("Silakan login terlebih dahulu.")
                
        if selected == "Register":
            all_keys = get_all_keys()

            st.info("Silakan isi form untuk mendaftar user baru.", icon="ℹ️")
            form_register = st.form('form_register', clear_on_submit=True)
            form_register.subheader("Register")

            # form_register.write("")
            txt_username = form_register.text_input("Input Username :")
            txt_name = form_register.text_input("Input Nama :")
            txt_role = "user"
            txt_password = form_register.text_input("Input Password :", type="password")

            # HashSha256
            hashed_password = stauth.Hasher([txt_password]).generate()
            hashed_password = hashed_password[0]

            if form_register.form_submit_button("Register"):
                if txt_username and txt_name and txt_password and txt_role:

                    if txt_username in all_keys:
                        red_alert("Username sudah ada, coba yang lain.")

                    else:
                        result_put = put_user(txt_username, txt_name, hashed_password, txt_role)
                        if result_put != None:
                            green_alert("Registrasi berhasil. Data tersimpan.")
                            # st.experimental_rerun()
                else:
                    red_alert("Mohon isi kolom yang masih kosong.")       
        
    # --- main pages ---
    else:
        authenticator.logout("Logout", "sidebar")
        username = session['username']
        
        role = get_user(username).get('role')
        session['role'] = role
        
        # ROLE ADMIN
        if role == "admin":
            #sidebar
            with st.sidebar:
                st.divider()
                
                menu = ["🏡 Home", "📋 Sentiment Analysis", "💬 Sentiment Predictor", "📚 Report", "⚙️ Account Management"]
                selected = st.selectbox(label = "&nbsp;&nbsp; **DASHBOARD** ", 
                                        options = menu, 
                                        index=0, 
                                        label_visibility = "visible")
                
                # st.info(f"Username: **{username.capitalize()}**", icon="ℹ️")
                # st.info(f"Role: **{role.capitalize()}**", icon="ℹ️")
                
            #contents
            if selected == "🏡 Home":
                st.sidebar.info(f"Username: **{username.capitalize()}**", icon="ℹ️")
                st.sidebar.info(f"Role: **{role.capitalize()}**", icon="ℹ️")
                admin_home_page()
            elif selected == "📋 Sentiment Analysis":
                admin_analysis_page()
            elif selected == "💬 Sentiment Predictor":
                pr.predictor_page()    
            elif selected == "📚 Report":
                admin_report_page()
            elif selected == "⚙️ Account Management":
                admin_database_page()
        
        # ROLE USER
        elif role == "user":
            #sidebar
            with st.sidebar:
                st.divider()
                menu = ["🏡 Home", "💬 Sentiment Predictor", "⚙️ Account Management"]
                selected = st.selectbox(label = "&nbsp;&nbsp; **DASHBOARD** ", 
                                        options = menu, 
                                        index=0, 
                                        label_visibility = "visible")

                # st.info(f"Username: **{username.capitalize()}**", icon="ℹ️")
                # st.info(f"Role: **{role.capitalize()}**", icon="ℹ️")

            #contents
            if selected == "🏡 Home":
                st.sidebar.info(f"Username: **{username.capitalize()}**", icon="ℹ️")
                st.sidebar.info(f"Role: **{role.capitalize()}**", icon="ℹ️")
                user_home_page()
            elif selected == "💬 Sentiment Predictor":
                pr.predictor_page()      
            elif selected == "⚙️ Account Management":
                user_database_page()
    
    st.write("")
    st.divider()
    st.write('''
         <p style="text-decoration: none;
         font-family: 'Open Sans'; font-size: 13px; font-weight: bold;
         color: white; background-color: #00386b; 
         border-radius: 5px; padding: 9px 22px;">
         © Agung Gunawan (2019230012) | Sentiment Analysis Random Forest Classifier</p>
     ''', unsafe_allow_html = True)
    
    # with st.expander("Session"):
    #         session
            
if __name__ == "__main__":
    main()

hide_streamlit = """ <style> footer {visibility: hidden;} </style> """    
st.markdown(hide_streamlit, unsafe_allow_html=True)