############################## THE AMIN APP SECTION ##############################

from app_plot import *
from app_model import *
from app_about_me import *
from app_data import *
from app_about_pro import *

matplotlib.use('Agg')


def main():
    st.title("Project Neural.ME")
    st.text("Using version: 1.0.0")
    activities = ["About the project", "Data", "Visualisations", "Neural net model", "About the developer"]
    choice = st.sidebar.selectbox("CHOOSE WHERE YOU WANT TO GO: ", activities)

    ####################################### FETCHING THE DATA  #######################################
    if choice == 'Data':
        fetch_data(choice)

    ####################################### MAKING THE PLOTS #######################################
    elif choice == 'Visualisations':
        visualise(choice)

    ####################################### THE MODEL BUILD #######################################
    elif choice == 'Neural net model':
        st.subheader("The Model build from scratch")
        model_build()

    ####################################### ABOUT THE DEVELOPER #######################################
    elif choice == 'About the developer':
        st.subheader("About the developer")
        about_me()

    ####################################### ABOUT THE DEVELOPER #######################################
    elif choice == 'About the project':
        about_project()


if __name__ == '__main__':
    main()
