import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import time

session = st.session_state

from deta import Deta
#database
deta = Deta(st.secrets["DETA_PROJECT_KEY"])
db = deta.Base("users")

def get_user(key):
    return db.get(key)

def update_user(updates, key):
    return db.update(updates, key)

def get_user_data(key):
    users = db.fetch().items
    if key is not None:
        users = [user for user in users if user["key"] == key]
    return users

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

def read_data():
    key = session['username']
    df = get_user_data(key)
    df = pd.DataFrame(df, columns=["key", "name", "password", "role"])
    df = df.rename(columns={"key": "username"})
    
    read_data = st.dataframe(data=df, use_container_width=True, 
                             hide_index=True, column_order=("username", "name", "role", "password"))

def user_database_page():
    #get all keys
    key = session['username']
    all_keys = key
    
    st.title("Account Management", help="Halaman untuk update Data User dari Cloud Deta Base.")
    st.divider()
       
    st.subheader("Data User")
    read_data()

    st.info("Silakan isi form update untuk mengubah data user.", icon="ℹ️")
    st.subheader("Update")

    # --- FORM ---
    with st.expander("", expanded=True):
        option = st.text_input("Username :", all_keys, disabled=True)

        #get_name
        get_name = get_user(option).get("name")
        get_role = get_user(option).get("role")

        txt_name_update = st.text_input("Update Nama :", get_name)
        if get_role == 'admin':
            txt_role_update = st.radio("Pilih Role :", ("admin", "user"), index=0, disabled=True, horizontal=True)
        else:
            txt_role_update = st.radio("Pilih Role :", ("admin", "user"), index=1, disabled=True, horizontal=True)
        txt_password_update = st.text_input("Update Password :", type="password")

        # HashSha256
        hashed_password_update = stauth.Hasher([txt_password_update]).generate()
        hashed_password_update = hashed_password_update[0]

        key_update = option
        name = txt_name_update
        password = hashed_password_update
        role = txt_role_update
        updates = {"name": name, "password": password, "role": role}

        if st.button("Update"):
            if option and txt_name_update and txt_password_update and txt_role_update:
                if key_update == None:
                    red_alert("Username tidak tersedia.")

                else:
                    result_update = update_user(updates, key_update)    
                    if result_update == None:
                        green_alert("Update berhasil. Data tersimpan.")
                        st.experimental_rerun()
            else:
                red_alert("Mohon isi kolom yang masih kosong.")

    