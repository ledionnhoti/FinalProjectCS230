'''Name: Ledi Hoti
CS230: Section 002
Data: California Rest Areas

Description:
This program analyzes rest areas in California and rates them based on amenities offered, looks at correlations between
different amenities, and visualizes the distribution of them across the state
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np
from collections import Counter
import re


def load_data():
    '''
    Loads data from the "Rest_Areas.csv" file and preprocesses it by converting amenities from Yes/No to binary (1/0), 
    and calculates a score based on the presence of amenities.

    Returns:
        df (DataFrame): A DataFrame with processed data and additional 'SCORE' column representing the total amenities.
    '''

    df = pd.read_csv('Rest_Areas.csv')
    amenity_columns = ['RESTROOM', 'WATER', 'PICNICTAB', 'PHONE', 'HANDICAP', 'RV_STATION', 'VENDING', 'PET_AREA']
    
    # Convert all Yes/No-s to 1/0-s for calculation purposes
    for column in amenity_columns:
        df[column] = [1 if x == 'Yes' else 0 for x in df[column]]
    
    df['SCORE'] = df[amenity_columns].sum(axis=1)
    return df


def display_home(df):
    '''
    Displays the home page of the Streamlit app, including various sections of text, data analysis, and visualizations.

    Parameters:
        df (DataFrame): The DataFrame containing rest areas data used for display and analysis.
    '''
    
    st.header("Statistical Analysis of CALIFORNIA Rest Areas", divider="red")
    
    st.write("""
    This CS 230 project encapsulates an analysis of rest areas in the state of California 
    It includes various levels of statistical and visual statements which include the following:
    
    - A **frequency distribution histogram** of postmile ranges across the state
    - A **map** that showcases the locations of the rest areas across the country with multiple filtering options
    - A **pie chart** that showcases the distribution of rest area scores _(a self-calculated variable that evaluates
    rest areas based on number of present amenities)_
    - A **bonus correlation heatmap** of different amenities available for respective rest areas""")

    st.image('California.png', caption="A Chat GPT generated image of a peaceful rest area in the California woods...",
             use_column_width=True)
    
    st.subheader("Brief Data Set Description", divider="red")
    st.write("""
        The dataset provides detailed information on 87 rest areas across California, capturing their geographical
        locations, available amenities, and other relevant details. Each entry includes specific location data such
        as latitude, longitude, city, and county, alongside the route and traffic direction. The dataset notably tracks
        the presence of key amenities such as restrooms, water facilities, picnic tables, phone access, handicapped
        accessibility, RV stations, vending machines, and pet areas. All rest areas offer restroom and water facilities,
        showcasing their universal availability. 
        
        Additional details include the address, zipcode, postmiles, etc. Lastly, the data set also includes a self-calculated 
        field called \"Score\" which evaluates each rest area based on the number of amenities present. This comprehensive 
        dataset serves as an invaluable resource for analyzing the distribution of amenities across
        California's rest areas, helping to enhance traveler experience and planning.
        
        Scroll to the bottom of this page to get a quick peak of the data I worked with!""")
    
    st.subheader("QUICK DEFINITION - \"Postmiles\"", divider="red")
    st.write("""
        **Postmiles** in California are markers used to track specific locations along state highways. These are measured in miles 
        from a county's southern or western boundary to the specific point along the highway. **Postmiles** are crucial for navigation, 
        highway maintenance, and emergency response, serving as precise references that help in locating specific sites within the 
        extensive road network of the state. They are visible on small, rectangular metal signs along the highway, typically seen on
        the right side of the road.
        
        Following, you can see **a simple descriptive data analysis** that focuses on aspects of POSTMILE for each rest area: """)

    stats, avg_val, max_val, stdev = summary_statistics(df)
        
    for stat in stats:
        if stat == 'Average Postmiles':
            st.write(f"- {stat} = {avg_val:.1f} miles")
        if stat == 'Maximum Postmiles':
            st.write(f"- {stat} = {max_val:.1f} miles")
        if stat == 'Standard Deviation of Postmiles':
            st.write(f"- {stat} = {stdev:.1f} miles")
    
    st.subheader("Top 10 Most Common Words in Rest Area Names", divider="red")
    st.write("After a simple natural language processing performance, below you can see the most popular words used in rest area names:")
    top_words_df = text_analysis(df)
    st.dataframe(top_words_df.style.set_properties(**{'text-align': 'left'}), width=300, height=300)  # Adjust width and height as needed
    
    st.subheader('Data Set Table (abbreviated)', divider="red")
    st.write("""
        Below you can see a snippet of some of my assigned data set - CALIFORNIA REST AREAS _(sorted in 
        ascending order on name)_""")
    
    columns_to_display = ['NAME', 'CITY', 'COUNTY', 'POSTMILE', 'SCORE']
    sorted_df = df[columns_to_display].sort_values(by='NAME', ascending=True)
    st.dataframe(sorted_df)


def summary_statistics(df, stats=['mean', 'max', 'std']):
    '''
    Computes summary statistics for the 'POSTMILE' column in the given DataFrame based on the specified statistics list.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        stats (list of str): A list of strings indicating which statistics to compute.

    Returns:
        tuple: A tuple containing:
               - results (list of str): Descriptions of the computed statistics.
               - mean_value (float): The mean value of the 'POSTMILE' column.
               - max_value (float): The maximum value of the 'POSTMILE' column.
               - std_dev (float): The standard deviation of the 'POSTMILE' column.
    '''
    
    results = []
    mean_value = df['POSTMILE'].mean() if 'mean' in stats else None
    max_value = df['POSTMILE'].max() if 'max' in stats else None
    std_dev = df['POSTMILE'].std() if 'std' in stats else None

    if 'mean' in stats:
        results.append('Average Postmiles')
    if 'max' in stats:
        results.append('Maximum Postmiles')
    if 'std' in stats:
        results.append('Standard Deviation of Postmiles')

    # Return multiple values
    return results, mean_value, max_value, std_dev


def text_analysis(data):
    '''
    Analyzes the 'NAME' column in the DataFrame to find the most common words used.

    Parameters:
        data (DataFrame): The DataFrame containing rest areas names to analyze.

    Returns:
        top_words_df (DataFrame): A DataFrame containing the top 10 most frequent words and their counts.
    '''
    
    # Combine all names into a single string
    text = ' '.join(data['NAME'])
    # Normalize the text by converting to lower case
    text = text.lower()
    # Remove non-alphabetic characters and split into words
    words = re.findall(r'\b[a-z]+\b', text)
    # Count the frequency of each word using Counter
    word_counts = Counter(words)
    # Get the top 10 most common words
    top_words = word_counts.most_common(10)
    # Convert the list of tuples to a DataFrame for nice table format
    top_words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    
    return top_words_df    


def plot_postmile_histogram(df):
    '''
    Plots a histogram of the 'POSTMILE' column from the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the data.

    Note: This function also interacts with Streamlit to get user input for filtering histogram data.
    '''
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = ax.hist(df['POSTMILE'], bins=20, color='orange', alpha=0.75, label='Postmiles')

    # Calculate the mean of POSTMILE and add a vertical line
    mean_postmile = df['POSTMILE'].mean()
    ax.axvline(mean_postmile, color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_postmile:.2f}')

    # Calculate a curved trend line for the histogram
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    z = np.polyfit(bin_centers, n, 2)  # Fitting a quadratic polynomial (degree 2)
    p = np.poly1d(z)
    ax.plot(bin_centers, p(bin_centers), "r--", label='Trendline')
    
    # Adding labels and title
    ax.set_xlabel('Postmiles')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Postmiles')
    ax.legend()
    ax.grid(True)

    # Streamlit number input for user-defined distances
    floor_distance = st.number_input("Enter starting distance", min_value=0.0, max_value=160.0, step=5.0)
    ceiling_distance = st.number_input("Enter ending distance", min_value=1.0, max_value=161.15, step=5.0)

    # Calculate the indices for the bins
    floor_idx = np.searchsorted(bins, floor_distance) - 1
    ceiling_idx = np.searchsorted(bins, ceiling_distance) - 1

    # Sum the counts in the specified range
    postmiles_inbetween = sum(n[floor_idx:ceiling_idx + 1])
    total_postmiles = len(df)
    probability = (postmiles_inbetween / total_postmiles) * 100

    st.write(f"The probability of a postmile being between {floor_distance} miles and {ceiling_distance} miles is {probability:.2f}%.")

    # Highlight the range on the histogram
    plt.axvspan(bins[floor_idx], bins[ceiling_idx], color='red', alpha=0.3)

    st.pyplot(fig)

    
def histogram(df): 
    '''
    Facilitates the visualization of the postmile histogram and relevant descriptions on Streamlit.

    Parameters:
        df (DataFrame): The DataFrame containing rest areas data.
    '''
    
    st.header("Postmile Distance Distribution", divider="red")

    st.write("""
    This histogram represents the distribution of postmiles for rest areas across California, 
    where postmiles are used to measure the distance along a state highway from a county's 
    southern or western boundary to the specific location of each rest area.""")

    st.write("Below is the histogram visualizing the postmile distribution across California rest areas:")
    plot_postmile_histogram(df)
    
    st.write("""
    **Key Findings:**
    - The **mean postmile**, indicated by the blue dashed line, provides a central point in the distribution, 
      helping to contextualize where the majority of rest areas are concentrated along the highways.
    - The distribution of postmiles is not uniform, suggesting that rest areas are clustered more densely 
      in certain segments of the highways. This could reflect areas with higher traffic volumes or strategic 
      locations important for traveler support.
    - The trendline, shown in red, illustrates the general trend of how rest area frequencies change 
      across the range of postmiles. This trendline indicates fluctuations in rest area density, which might 
      be due to geographical, administrative, or historical factors influencing where rest areas have been established.

    **Conclusions:**
    It is clear that rest areas are strategically placed but vary significantly in frequency across different 
    highway stretches. Understanding these patterns can help in planning better highway services and improving 
    travel experiences. This distribution also serves as a foundational analysis for further detailed studies 
    on access, usage, and needs assessment for highway rest areas in California.
    
    Here is a calm video of nature to relax you as you grade this project :)
    """)
    
    st.video('Relaxing_Video.mp4')


def plot_amenity_correlation(df):
    '''
    Generates a correlation heatmap for amenities in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing binary indicators for amenities.

    Returns:
        fig (matplotlib.figure.Figure): A matplotlib figure containing the correlation heatmap.
    '''
    
    correlation = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Correlation Between Respective Amenities')
    
    return fig


def count_amenities(df):
    '''
    Counts the presence of each amenity across all rest areas in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing binary indicators for amenities.

    Returns:
        amenity_counts (dict): A dictionary where keys are amenities and values are their respective counts.
    '''
    
    amenity_columns = ['RESTROOM', 'WATER', 'PICNICTAB', 'PHONE', 'HANDICAP', 'RV_STATION', 'VENDING', 'PET_AREA']

    # Initialize a dictionary to count occurrences of each amenity
    amenity_counts = {}

    # Loop through each row and increment counts for each amenity present
    for index, row in df.iterrows():
        for amenity in amenity_columns:
            if row[amenity] == 1: 
                if amenity in amenity_counts:
                    amenity_counts[amenity] += 1
                else:
                    amenity_counts[amenity] = 1

    return amenity_counts


def display_heatmap(df):
    '''
    Displays the amenities correlation heatmap and related text on Streamlit.

    Parameters:
        df (DataFrame): The DataFrame containing rest areas data with amenities.
    '''
    
    st.header('Amenities Correlation Analysis', divider="red")
    
    st.write("""
        This page includes a correlation heatmap that showcases the relationship between different amenities. The higher the 
        correlation coefficient, the higher the likelihood that one amenity is present when the other is too. It should be noted that 
        some of the perfect correlation scores are due to there being a combination of some of the amenities with themselves, which 
        logically result in a perfect positive correlation.""")
    
    amenity_columns = ['RESTROOM', 'WATER', 'PICNICTAB', 'PHONE', 'HANDICAP', 'RV_STATION', 'VENDING', 'PET_AREA']
    
    amenity_data = df[amenity_columns]
    fig = plot_amenity_correlation(amenity_data)
    st.pyplot(fig)
    
    # Display the sorted counts of amenities in the first column
    amenities = count_amenities(df)
    if amenities:
        st.write("""
            Below is a table of amenities and the number of times you can find them across the state, sorted in descending order.
            Since there is a total of 87 rest areas, we can see that restrooms, water stations, and handicap areas top the 
            charts - they are present in 100% of the rest areas:""")
        st.write('')
        # Create columns for layout
        col1, col2 = st.columns([3, 2])  # Adjust the ratio as needed
        
        # Creating DataFrame from dictionary
        amenity_df = pd.DataFrame(list(amenities.items()), columns=['Amenity Type', 'Count of Presence'])
        amenity_df['Amenity Type'] = amenity_df['Amenity Type'].apply(lambda x: x.replace('_', ' ').capitalize())
        amenity_df.sort_values(by='Count of Presence', ascending=False, inplace=True)
        col1.dataframe(amenity_df)
        
    # Load and display the image in the second column
    col2.image('Amenities.png', caption="A Chat GPT generated image of different amenity symbols...", width=300)  # Set width to control size


def display_map(df):
    '''
    Displays an interactive map on Streamlit showing the locations of rest areas filtered by amenities and cities.

    Parameters:
        df (DataFrame): The DataFrame containing rest areas data.
    '''
    
    st.header('Locations Analysis', divider="red")
    
    st.write("""
        Below you can see a map of California - combined with multiple filters you can play around with. Keep in mind that
        once you select a filter (e.g. City), other filters (e.g. amenity type) are overruled.""")
    
    amenity_options = ['RESTROOM', 'WATER', 'PHONE', 'RV_STATION', 'VENDING', 'PET_AREA', 'PICNICTAB']
    selected_amenity = st.selectbox('Select an amenity to display:', amenity_options)
    
    df = df[df[selected_amenity] == 1]  # Apply amenity filter directly

    color_option = st.selectbox('Choose a color for the markers:', ['Red', 'Blue', 'Green'])
    color_map = {'Red': [255, 30, 0, 160], 'Blue': [0, 30, 255, 160], 'Green': [30, 255, 0, 160]}
    zoom_level = st.slider('Zoom level:', 1, 20, 6)

    # Clean and prepare city data
    df = df.dropna(subset=['CITY', 'LATITUDE', 'LONGITUDE'])  
    df['CITY'] = df['CITY'].str.strip()
    df = df[df['CITY'] != '']

    # Prepare city options for multi-select
    unique_sorted_cities = sorted(df['CITY'].unique())
    selected_cities = st.multiselect('Select cities to filter:', unique_sorted_cities)

    # Filter DataFrame for selected cities, if any
    if selected_cities:
        df_filtered = df[df['CITY'].isin(selected_cities)]
    else:
        df_filtered = df

    # Generate tooltip content
    tool_tip = {
        "html": 
        "<b>NAME:</b> {NAME}<br>"
        "<b>CITY:</b> {CITY}<br>"
        "<b>COUNTY:</b> {COUNTY}<br>"
        "<b>SCORE:</b> {SCORE}<br>",
        "style": {"backgroundColor": "wine", "color": "white"}
    }

    # Configure the map layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_filtered,
        get_position='[LONGITUDE, LATITUDE]',
        get_color=color_map[color_option],
        get_radius=10000,
        pickable=True,
        tooltip=tool_tip
    )

    # Set the view state for the map
    view_state = pdk.ViewState(
        latitude=df_filtered['LATITUDE'].mean() if not df_filtered.empty else 37.5,
        longitude=df_filtered['LONGITUDE'].mean() if not df_filtered.empty else -119.5,
        zoom=zoom_level,
        pitch=0
    )

    # Create the map
    mapp = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/satellite-streets-v11',
        tooltip=tool_tip
    )
    
    st.pydeck_chart(mapp)


def display_pie_chart(df):
    '''
    Displays a pie chart analysis of rest areas' scores on Streamlit.

    Parameters:
        df (DataFrame): The DataFrame containing rest areas data.
    '''
    
    st.header('Score Distribution Analysis', divider="red")
    
    st.write("""
        This analysis evaluates each rest area based on its postmile and compares it to the average postmile
        across all rest areas. Users can choose to view the distribution of rest areas with above average, below average,
        or all postmiles.
    """)

    # Calculate the average postmiles
    average_postmile = df['POSTMILE'].mean()
    st.write(f"The average postmile across all rest areas is {average_postmile:.2f} miles.")

    # Option to select category to view
    option = st.selectbox(
        'Choose which rest areas to display:',
        ['All Rest Areas', 'Above Average Postmiles', 'Below Average Postmiles']
    )

    # Filter data based on selection
    if option == 'Above Average Postmiles':
        filtered_df = df[df['POSTMILE'] > average_postmile]
    elif option == 'Below Average Postmiles':
        filtered_df = df[df['POSTMILE'] <= average_postmile]
    else:
        filtered_df = df  # For 'All Rest Areas'

    # Plotting the pie chart
    if not filtered_df.empty:
        fig = plot_score_distribution(filtered_df)
        st.pyplot(fig)
    else:
        st.error('No data available for this category.')

    st.write('')
    st.write('How about some more AI-generated pictures?!')
    st.write('Below you will see different levels of rest areas based on their score - you can have lots of fun or you can go '
             'to rest to a place that looks like something that came out of the Walking Dead :)!')
    st.write('')  

    st.image('AmenitiesLevels.png', caption="A Chat GPT generated image of different kinds of rest areas based on their score...",
             width=500)

    
def plot_score_distribution(df):
    '''
    Plots a pie chart of rest areas scores based on the 'SCORE' column in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the scores data.

    Returns:
        fig (matplotlib.figure.Figure): A matplotlib figure containing the pie chart.
    '''
    
    score_counts = df['SCORE'].value_counts().sort_index()
            
    fig, ax = plt.subplots()
    labels = ['Score is ' + str(i) for i in score_counts.index]
    wedges, texts, autotexts = ax.pie(score_counts, labels=labels, startangle=90, autopct=lambda pct: "{:.1f}%".format(pct) if pct > 5 else '')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Distribution of Rest Area Scores')

    # Make the percentage labels easier to read and not overlapping
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.tight_layout()

    return fig


def main():
    '''
    The main function that loads data, sets up Streamlit sidebar and pages, and handles navigation between different pages.
    '''
    
    df = load_data()
    
    st.sidebar.image('Futuristic.png')
    st.sidebar.title("PAGES")
    options = ["**Home**", "**Postmile Frequency Analysis**", "**Location Analysis**", "**Score Distribution Analysis**", "**Correlation Analysis**"]
    selection = st.sidebar.radio("Choose a chapter you want to explore", options)
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="**Download Full Data Set**",
        data=csv_data,
        file_name='Rest_Areas.csv',
        mime='text/csv')
    
    if selection == "**Home**":
        display_home(df)
    elif selection == "**Postmile Frequency Analysis**":
        histogram(df)
    elif selection == "**Location Analysis**":
        display_map(df)
    elif selection == "**Score Distribution Analysis**":
        display_pie_chart(df)
    elif selection == "**Correlation Analysis**":
        display_heatmap(df)


if __name__ == "__main__":
    main()
