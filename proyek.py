import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set the title of the dashboard
st.title("Dashboard Proyek Analisis Data Bike Hirings")

# Display logo
logo_url = "https://www.thecyclingexperts.co.uk/uploaded_images/new-tce-post-oct-2010/dsc03227_website.jpg"
st.markdown(f'<a href="{logo_url}" target="_blank"><img src="{logo_url}" width="100"></a>', unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>by Ahmad Hanafi</h3>", unsafe_allow_html=True)

# Load data
merged_data = pd.read_csv('merged_data_output.csv')

# Ensure 'dteday' is in datetime format
merged_data['dteday'] = pd.to_datetime(merged_data['dteday'], errors='coerce')

# Create a list of all hours for each day (hourly data)
merged_data['dteday'] = merged_data['dteday'].dt.floor('H')  # Adjusting the datetime to hourly precision (e.g., '2011-01-01 00:00:00')

# Expand data to include hourly intervals (for each day)
expanded_data = pd.DataFrame()
for date in merged_data['dteday'].dt.date.unique():
    # Generate hours for each date
    hours = pd.date_range(start=f'{date} 00:00:00', end=f'{date} 23:00:00', freq='H')
    # Get the data for this particular date
    date_data = merged_data[merged_data['dteday'].dt.date == date]
    # Merge the date data with the hours for that date 
    hourly_data = pd.DataFrame({'dteday': hours})
    hourly_data = hourly_data.merge(date_data, on='dteday', how='left')  # Merge data based on the exact hour

    # Append the hourly data for the particular date to the expanded dataset
    expanded_data = pd.concat([expanded_data, hourly_data], ignore_index=True)

# Select start and end dates
start_date = st.date_input('Select Start Date', value=expanded_data['dteday'].min().date())
end_date = st.date_input('Select End Date', value=expanded_data['dteday'].max().date())
user_type = st.selectbox('Select User Type', ['All', 'Casual', 'Registered'])

# Convert selected dates to datetime for comparison
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter Data by date range
filtered_data = expanded_data[(expanded_data['dteday'] >= start_date) & (expanded_data['dteday'] <= end_date)]

# Filter by user type based on the selected columns
if user_type == 'Casual':
    filtered_data = filtered_data[['dteday', 'casual_hour']]  # Select only casual columns
    filtered_data = filtered_data[filtered_data['casual_hour'] > 0]  # Filter for hours with casual rentals
elif user_type == 'Registered':
    filtered_data = filtered_data[['dteday', 'registered_hour']]  # Select only registered columns
    filtered_data = filtered_data[filtered_data['registered_hour'] > 0]  # Filter for hours with registered rentals
elif user_type == 'All':
    # Keep all columns if 'All' is selected
    filtered_data = filtered_data[['dteday', 'casual_hour', 'registered_hour', 'cnt_hour']]

# Display filtered data
st.write(f"Showing data from {start_date.date()} to {end_date.date()} for {user_type} users.")
st.write(filtered_data)

# Data Summary
st.write(filtered_data.describe())

# Plot Rental Trends
fig, ax = plt.subplots(figsize=(10, 6))

# Adjusting the plot based on the user type
if user_type == 'Casual':
    sns.lineplot(data=filtered_data, x='dteday', y='casual_hour', label='Casual Hour', ax=ax)
elif user_type == 'Registered':
    sns.lineplot(data=filtered_data, x='dteday', y='registered_hour', label='Registered Hour', ax=ax)
else:  # 'All' selected
    sns.lineplot(data=filtered_data, x='dteday', y='cnt_hour', label='Total Hour', ax=ax)

# Customize plot
ax.set_title(f"Rental Trends: {user_type} Users from {start_date.date()} to {end_date.date()}")
ax.set_xlabel('Date and Hour')
ax.set_ylabel('Rental Count')
st.pyplot(fig)

# Add a 'month_year' column for grouping by year and month
merged_data['month_year'] = merged_data['dteday'].dt.to_period('M')

# Filter the data for the years 2011 and 2012
filtered_data = merged_data[(merged_data['dteday'].dt.year == 2011) | (merged_data['dteday'].dt.year == 2012)]

# Group by 'month_year' and sum the rentals for casual customers
monthly_rentals_casual = filtered_data.groupby('month_year')[['casual_hour']].sum()
monthly_rentals_casual.index = monthly_rentals_casual.index.to_timestamp()

# Group by 'month_year' and sum the rentals for registered customers
monthly_rentals_registered = filtered_data.groupby('month_year')[['registered_hour']].sum()
monthly_rentals_registered.index = monthly_rentals_registered.index.to_timestamp()

# Create a 1x2 subplot layout for side-by-side plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the monthly trends for casual customers on ax1
sns.lineplot(data=monthly_rentals_casual, x=monthly_rentals_casual.index, y='casual_hour', label='Casual', color='orange', ax=ax1)
ax1.set_title('Tren Peminjaman Sepeda oleh Casual User (2011-2012)')
ax1.set_xlabel('Bulan')
ax1.set_ylabel('Jumlah Peminjaman Sepeda')
ax1.grid(True)
ax1.tick_params(axis='x', rotation=45)

# Plot the monthly trends for registered customers on ax2
sns.lineplot(data=monthly_rentals_registered, x=monthly_rentals_registered.index, y='registered_hour', label='Registered', color='blue', ax=ax2)
ax2.set_title('Tren Peminjaman Sepeda oleh Registered User (2011-2012)')
ax2.set_xlabel('Bulan')
ax2.set_ylabel('Jumlah Peminjaman Sepeda')
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)

# Add a title for the entire figure
fig.suptitle('Trend Peningkatan User Casual & Registered User Di Tahun 2011-2012', fontsize=16)
# Display the plots in Streamlit
st.pyplot(fig)

# Group by 'year' and calculate total rentals for casual and registered users
yearly_rentals = merged_data.groupby('yr_hour')[['casual_hour', 'registered_hour']].sum()

# Rename columns for clarity
yearly_rentals.columns = ['Casual', 'Registered']

# Plot the yearly rentals as a bar chart
yearly_rentals.plot(kind='bar', figsize=(8, 6), color=['orange', 'blue'], width=0.8)

# Add title and labels
plt.title('Peminjaman Sepeda Berdasarkan Tahun (Casual & Registered)')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Peminjaman Sepeda')
plt.legend(title='Jenis Pelanggan')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
st.pyplot()  # Streamlit rendering

# Group by 'season_hour' and 'yr_hour', and calculate total rentals for casual users
seasonal_rentals_by_year = filtered_data.groupby(['season_hour', 'yr_hour'])[['casual_hour']].sum().unstack()

# Rename columns for clarity
seasonal_rentals_by_year.columns = ['Casual (2011)', 'Casual (2012)']

# Create a 1x2 subplot layout for side-by-side plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the yearly rentals as a bar chart on ax1
yearly_rentals.plot(kind='bar', ax=ax1, color=['orange', 'blue'], width=0.8)
ax1.set_title('Peminjaman Sepeda Berdasarkan Tahun (Casual & Registered)')
ax1.set_xlabel('Tahun')
ax1.set_ylabel('Jumlah Peminjaman Sepeda')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot the seasonal rentals by year as a bar chart on ax2
seasonal_rentals_by_year.plot(kind='bar', ax=ax2, color=['orange', 'coral'], width=0.8)
ax2.set_title('Peminjaman Sepeda Berdasarkan Musim dan Tahun (2011-2012)')
ax2.set_xlabel('Musim')
ax2.set_ylabel('Jumlah Peminjaman Sepeda')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add a title for the entire figure
fig.suptitle('Peminjaman Sepeda Berdasarkan Musim dan Tahun (2011-2012)', fontsize=16)

# Display the plots in Streamlit
st.pyplot(fig)

# Group by month and calculate total rentals
monthly_totals = merged_data.groupby('month_name_hour')['cnt_hour'].sum().sort_values(ascending=True)

# Create a colormap with unique colors for each month
colors = plt.cm.tab20c(range(len(monthly_totals)))

# Plotting the monthly rentals as a horizontal bar chart
plt.figure(figsize=(10, 6))
monthly_totals.plot(kind='barh', color=colors)
plt.title('Total Peminjaman Sepeda Berdasarkan Bulan')
plt.xlabel('Jumlah Peminjaman')
plt.ylabel('Bulan')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot()  # Render the plot with Streamlit

# Display the peak month
peak_month = monthly_totals.idxmax()
st.write(f"Peminjaman tertinggi terjadi pada bulan '{peak_month}'.")

# Streamlit App Header
st.title("Interactive Bike Rentals Dashboard")
st.subheader("View total bike rentals for Casual, Registered, or Both users by Month")

# Display the full data in a table
st.write("Total bike rentals for Casual, Registered, and All users by month:")
st.dataframe(merged_data.set_index('month_name_hour'))

# Interactive selection to show casual, registered, or both in the chart
rental_type = st.radio("Select Rental Type to View:", ('Casual', 'Registered', 'All'))

# Filter data based on the selection
if rental_type == 'Casual':
    rental_data = merged_data[['month_name_hour', 'casual_hour']]
elif rental_type == 'Registered':
    rental_data = merged_data[['month_name_hour', 'registered_hour']]
else:  # Show both Casual and Registered
    rental_data = merged_data[['month_name_hour', 'cnt_hour']]

# Display the selected data as a table
st.write(f"Total bike rentals for {rental_type} users by month:")
st.dataframe(rental_data.set_index('month_name_hour'))

# Create an interactive bar chart for the selected rental data
st.bar_chart(rental_data.set_index('month_name_hour').sort_index())

# Group by hour and calculate total rentals
hourly_totals = merged_data.groupby('hour_description')['cnt_hour'].sum()

# Get the peak hour
peak_hour = hourly_totals.idxmax()
peak_hour_value = hourly_totals.max()

# Plotting the hourly rentals as a line chart
plt.figure(figsize=(15, 7))
plt.plot(hourly_totals.index, hourly_totals.values, marker='o', color='red', linestyle='-', linewidth=2, markersize=6)
plt.title('Total Peminjaman Sepeda Berdasarkan Jam')
plt.xlabel('Jam')
plt.ylabel('Jumlah Peminjaman')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.xticks(range(0, 24))  # Show all 24 hours on x-axis
st.pyplot()  # Render the plot with Streamlit

# Display the peak hour
st.write(f"Peminjaman tertinggi terjadi pada jam '{peak_hour}:00' dengan {peak_hour_value} peminjaman.")


# Group by 'hour_description' and calculate the total 'cnt_hour'
hourly_totals = merged_data.groupby('hour_description')['cnt_hour'].sum()

# Streamlit App Header
st.title("Interactive Hourly Bike Rentals Dashboard")
st.subheader("View total bike rentals by Hour of the Day (12 AM to 11 PM)")

# Display the full grouped data in a table
st.write("Total bike rentals by hour of the day:")
st.dataframe(hourly_totals)

# Interactive selection for chart type (bar or line)
chart_type = st.radio("Select the chart type:", ('Line Chart', 'Bar Chart'))

# Display the selected chart type
if chart_type == 'Line Chart':
    st.line_chart(hourly_totals)
else:
    st.bar_chart(hourly_totals)

# Allow users to select a specific hour (from 12 AM to 11 PM)
selected_hour = st.selectbox("Select an hour to view total rentals:", hourly_totals.index.tolist())

# Filter data based on the selected hour
selected_hour_data = hourly_totals[selected_hour]
st.write(f"Total bike rentals at {selected_hour}: {selected_hour_data}")


st.subheader("Faktor-faktor yang mempengaruhi tingkat peminjaman sepeda")

# Line plot: Temperature vs Bike Rentals (Hourly)
plt.figure(figsize=(8, 5))
sns.lineplot(x=merged_data['temp_hour'], y=merged_data['cnt_hour'], color='orange')
plt.title('Temperature vs. Bike Rentals (Hourly)')
plt.xlabel('Temperature (Hourly)')
plt.ylabel('Total Bike Rentals (Hourly)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Weather Condition vs Average Bike Rentals (Hourly)
plt.figure(figsize=(12, 6))
# Group data by weather condition and calculate the mean rentals
weather_avg_rentals = merged_data.groupby('weather_condition_hour')['cnt_hour'].mean().reset_index()
sns.barplot(x='cnt_hour', y='weather_condition_hour', data=weather_avg_rentals, palette='coolwarm', orient='h')
plt.title('Weather Condition vs. Average Bike Rentals (Hourly)')
plt.xlabel('Average Bike Rentals (Hourly)')
plt.ylabel('Weather Condition')
st.pyplot(plt)

# Pie chart: Season vs Total Bike Rentals (Hourly)
season_rentals = merged_data.groupby('season_hour')['cnt_hour'].sum()

# Plotting the pie chart
plt.figure(figsize=(10, 8))
plt.pie(season_rentals, labels=season_rentals.index, autopct='%1.1f%%', colors=sns.color_palette('coolwarm', len(season_rentals)), startangle=140)
plt.title('Proportion of Bike Rentals by Season (Hourly)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(plt)

# Tab layout for plotting
plot_choice = st.selectbox("Select Plot", ("Temperature vs Bike Rentals", "Weather Condition vs Average Bike Rentals", "Season vs Total Bike Rentals"))

if plot_choice == "Temperature vs Bike Rentals":
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=merged_data['temp_hour'], y=merged_data['cnt_hour'], color='orange')
    plt.title('Temperature vs. Bike Rentals (Hourly)')
    plt.xlabel('Temperature (Hourly)')
    plt.ylabel('Total Bike Rentals (Hourly)')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

elif plot_choice == "Weather Condition vs Average Bike Rentals":
    plt.figure(figsize=(12, 6))
    # Group data by weather condition and calculate the mean rentals
    weather_avg_rentals = merged_data.groupby('weather_condition_hour')['cnt_hour'].mean().reset_index()
    sns.barplot(x='cnt_hour', y='weather_condition_hour', data=weather_avg_rentals, palette='coolwarm', orient='h')
    plt.title('Weather Condition vs. Average Bike Rentals (Hourly)')
    plt.xlabel('Average Bike Rentals (Hourly)')
    plt.ylabel('Weather Condition')
    st.pyplot(plt)

elif plot_choice == "Season vs Total Bike Rentals":
    season_rentals = merged_data.groupby('season_hour')['cnt_hour'].sum()
    plt.figure(figsize=(10, 8))
    plt.pie(season_rentals, labels=season_rentals.index, autopct='%1.1f%%', colors=sns.color_palette('coolwarm', len(season_rentals)), startangle=140)
    plt.title('Proportion of Bike Rentals by Season (Hourly)')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(plt)

# Binning and Visualization functions
def binning_and_plot(df, column, labels, color, title):
    df[f'{column}_bins'] = pd.cut(df[column], bins=3, labels=labels)
    # Visualizing the binning
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df[f'{column}_bins'], kde=False, color=color, discrete=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(f'{column} Bins')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    return fig

# Create interactive tabs using Streamlit's selectbox for binning
binning_choice = st.selectbox("Select Binning Type", ('Registered Binning', 'Casual Binning', 'Count Hour Binning'))

if binning_choice == 'Registered Binning':
    registered_plot = binning_and_plot(merged_data, 'registered_day', ['Low', 'Medium', 'High'], 'skyblue', 'Binning of Registered into Low, Medium, High')
    st.pyplot(registered_plot)

elif binning_choice == 'Casual Binning':
    casual_plot = binning_and_plot(merged_data, 'casual_day', ['Low', 'Medium', 'High'], 'pink', 'Binning of Casual into Low, Medium, High')
    st.pyplot(casual_plot)

elif binning_choice == 'Count Hour Binning':
    cnt_hour_plot = binning_and_plot(merged_data, 'cnt_hour', ['Low', 'Medium', 'High'], 'orange', 'Binning of Count Hour into Low, Medium, High')
    st.pyplot(cnt_hour_plot)