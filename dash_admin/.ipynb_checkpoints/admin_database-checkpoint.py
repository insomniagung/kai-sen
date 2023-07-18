import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import time

from deta import Deta
#database
deta = Deta(st.secrets["DETA_PROJECT_KEY"])
db = deta.Base("users")

def put_user(username, name, password, role):
    return db.put({"key": username, "name": name, "password": password, "role": role})

def get_user(key):
    return db.get(key)

def update_user(updates, key):
    return db.update(updates, key)

def delete_user(key):
    return db.delete(key)

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

def yellow_alert(str):
    st.write("")
    with st.warning(str):
        time.sleep(2)
        st.empty()
        
def green_alert(str): 
    st.write("")
    with st.success(str):
        time.sleep(2)
        st.empty()   


session = st.session_state  

def read_data():
    df = db.fetch().items
    df = pd.DataFrame(df, columns=["key", "name", "password", "role"])
    df = df.rename(columns={"key": "username"})
    #df = df.drop({"password","role"}, axis=1)
    
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.index.name = "No."
    
    # format_text = lambda text: text[:15] + "..." if len(text) > 15 else text
    # df = df.applymap(format_text)
    
    read_data = st.dataframe(data=df, use_container_width=True, column_order=("username", "name", "role", "password"))
    #read_data = st.dataframe(data=df, use_container_width=True, hide_index=True)
    
def admin_database_page():
    #get all keys
    all_keys = get_all_keys()
    
    st.title("Account Management", help="Halaman untuk create, read, update, dan delete pada Data User dari Cloud Deta Base.")
    st.divider()
    
    tab1, tab2, tab3 = st.tabs([f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Register &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
                                "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
                                "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Delete &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"])
    
    with tab1: #Read & Register
        
        st.subheader("Data User")
        read_data()
        
        st.info("Silakan isi form register untuk menambahkan user baru.", icon="ℹ️")
        st.subheader("Register")
        
        form_register = st.form('form_register', clear_on_submit=True)
        
        txt_username = form_register.text_input("Input Username :")
        txt_name = form_register.text_input("Input Nama :")
        txt_role = form_register.radio("Pilih Role :", ("admin", "user"), index=1, horizontal=True)
        txt_password = form_register.text_input("Input Password :", type="password")
        
        
        # HashSha256
        hashed_password = stauth.Hasher([txt_password]).generate()
        hashed_password = hashed_password[0]
        
        #form_register.write("")
        if form_register.form_submit_button("Register"):
            if txt_username and txt_name and txt_password and txt_role:
                
                if txt_username in all_keys:
                    red_alert("Username sudah ada, coba yang lain.")
                    
                else:
                    result_put = put_user(txt_username, txt_name, hashed_password, txt_role)
                    if result_put != None:
                        green_alert("Registrasi berhasil. Data tersimpan.")
                        st.experimental_rerun()
            else:
                red_alert("Mohon isi kolom yang masih kosong.")
                
    with tab2: #Read & Update
        st.subheader("Data User")
        read_data()
        
        st.info("Silakan isi form update untuk mengubah data user.", icon="ℹ️")
        st.subheader("Update")
        
        # --- FORM ---
        with st.expander("", expanded=True):
            option = st.selectbox('Pilih Username :', all_keys)
            
            #get_name
            get_name = get_user(option).get("name")
            get_role = get_user(option).get("role")

            txt_name_update = st.text_input("Update Nama :", get_name)
            if get_role == 'admin':
                txt_role_update = st.radio("Pilih Role :", ("admin", "user"), index=0, horizontal=True)
            else:
                txt_role_update = st.radio("Pilih Role :", ("admin", "user"), index=1, horizontal=True)
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

    with tab3: #Read & Delete
        st.subheader("Data User")
        read_data()
        
        st.info("Silakan pilih user yang ingin dihapus.", icon="ℹ️")
        st.subheader("Delete")

        form_delete = st.form('form_delete')        
        option_delete = form_delete.selectbox('Pilih Username :', all_keys)
        key_delete = get_user(option_delete)
        
        if form_delete.form_submit_button("Delete"):
            if option_delete:                
                if key_delete == None:
                    red_alert("Username tidak tersedia.")                    
                else: #user tersedia
                    
                    if option_delete == session['username']:
                        red_alert("Data tidak dapat dihapus karena sedang digunakan.")
                    elif option_delete == 'admin' or option_delete == 'user':
                        yellow_alert("Data tidak dapat dihapus karena akun utama.")
                    else:
                        option_delete_info = option_delete
                        result_del = delete_user(option_delete_info)
                        if result_del == None:
                            green_alert(f"User '{option_delete_info}' berhasil dihapus.")
                            st.experimental_rerun()
                            
hide_streamlit = """ <style> footer {visibility: hidden;} </style> """                            
st.markdown(hide_streamlit, unsafe_allow_html=True)