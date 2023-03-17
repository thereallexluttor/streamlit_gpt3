import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import dataframe_cleaner as dc
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import dataframe_info as di
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import matplotlib.pyplot as plt



# function to generate profile_reports
def automl2():
    st.title('Machine Learning Automatizado')
    st.markdown('La magia de la IA a velocidades disruptivas.')

    # load the file to do the real f*cking magic!
    data_file = st.file_uploader("Subir archivo .csv !")

    # Catch the file and load it!
    if data_file is not None:
        data_df = pd.read_csv(data_file)

        # instance the cleanes
        cleaner = dc.CleanDataframe(data_df)

        # fix the names
        cleaner.name_formater()
    
        # clean the data
        data_df_final = cleaner.clean_dataframe()

        # instante the categorical cleanes
        df_no_categorical = di.InfoDataframe(data_df_final)

        # get the next df
        final_df = df_no_categorical.remove_categorical_columns()

        final_df

        # Crear una lista con el nombre de las variables
        columnas = final_df.columns.tolist()
        
        # Agregar una opción para seleccionar una variable
        variable = st.selectbox("Selecciona una variable objetivo", columnas)



        data = final_df
        X = data.drop(variable, axis=1)
        y = data[variable]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

         # Instanciar el modelo LazyClassifier
        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

        # Entrenar el modelo y obtener los resultados
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        # Mostrar los resultados
        models
         
         
        # crear un diccionario de modelos y predicciones
        model_dict = dict(zip(models, predictions))
        # plotear los resultados
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(model_dict)), list(model_dict.values()), align='center')
        ax.set_yticks(range(len(model_dict)))
        ax.set_yticklabels(list(model_dict.keys()))
        ax.set_xlabel('Accuracy')
        ax.set_title('LazyClassifier Model Comparison')
        st.pyplot(fig)


        

def Profile_report_section():
    st.title('Exploración de datos automatizada X.Insights')
    st.markdown('Explorar y analizar datos de manera automatizada y visualizar los resultados de forma interactiva.')

    # load the file to do the real f*cking magic!
    data_file = st.file_uploader("Subir archivo .csv !")
    
    # Catch the file and load it!
    if data_file is not None:
        data_df = pd.read_csv(data_file)

        # instance the cleanes
        cleaner = dc.CleanDataframe(data_df)

        # fix the names
        cleaner.name_formater()
    
        # clean the data
        data_df_final = cleaner.clean_dataframe()
        
        #plot the dataframe
        pr = ProfileReport(data_df_final, explorative=True)

        st.title("Pandas Profiling in Streamlit")
        st.write(data_df_final)
        st_profile_report(pr)

        # instante the categorical cleanes
        df_no_categorical = di.InfoDataframe(data_df_final)

        # get the next df
        final_df = df_no_categorical.remove_categorical_columns()

        final_df

        




    else:
        st.markdown('*')
    
   


# a sidebar menu
with st.sidebar:
    
    #creates the menu's format with their names
    selected = option_menu("XactAI", ["X.Insights", 'X.AutoML', 'X.Query'],   
        icons=['bar-chart-line', 'graph-up-arrow'], menu_icon="cast", default_index=1)

# X.Insights allows us to generate exploratory analysis 
if selected == "X.Insights":
    Profile_report_section()




if selected == "X.AutoML":
    automl2()
if selected == "X.Query":
    print('c')
