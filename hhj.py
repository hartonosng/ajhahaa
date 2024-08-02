import streamlit as st
import pandas as pd

# Contoh data
data = pd.DataFrame({
    'customerid': [1, 2, 3],
    'customername': ['Customer A', 'Customer B', 'Customer C']
})

def show_company_level_summary(customer_id):
    st.title("Company Level Summary")
    customer_info = data[data['customerid'] == customer_id]
    st.write(f"Detail untuk Customer ID: {customer_id}")
    st.dataframe(customer_info)

def general_summary():
    st.title("General Summary")
    
    # Mengkonversi dataframe menjadi HTML dengan tombol
    html = data.to_html(classes='table table-striped', escape=False)
    
    # Menambahkan tombol ke HTML
    html = html.replace('<table', '<table id="data-table"')
    html = html.replace('</tbody>', '''
    </tbody>
    <script>
    const buttons = document.querySelectorAll('.view-details');
    buttons.forEach(button => {
        button.addEventListener('click', () => {
            const customerId = button.getAttribute('data-customerid');
            window.location.href = `?action=select&id=${customerId}`;
        });
    });
    </script>
    ''')
    
    # Menambahkan tombol di setiap baris
    for index, row in data.iterrows():
        html = html.replace(f'<td>{row["customerid"]}</td>', f'<td>{row["customerid"]} <button class="view-details" data-customerid="{row["customerid"]}">Lihat Detail</button></td>')
    
    # Menampilkan HTML
    st.markdown(html, unsafe_allow_html=True)
    
    # Mengambil customer_id dari query parameter
    if 'action' in st.experimental_get_query_params() and st.experimental_get_query_params()['action'][0] == 'select':
        customer_id = st.experimental_get_query_params()['id'][0]
        st.session_state['selected_customerid'] = customer_id
        st.experimental_rerun()

def main():
    st.sidebar.title("Menu")
    selection = st.sidebar.radio("Pilih Halaman", ["General Summary", "Company Level Summary"])

    if selection == "General Summary":
        general_summary()
    elif selection == "Company Level Summary":
        if 'selected_customerid' in st.session_state:
            show_company_level_summary(st.session_state['selected_customerid'])
        else:
            st.write("Pilih customer ID dari General Summary.")

if __name__ == "__main__":
    main()


