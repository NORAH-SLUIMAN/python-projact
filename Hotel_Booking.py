# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/nona/Desktop/hotel_bookings.csv'
df = pd.read_csv(file_path)

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Set display option to show all columns
pd.set_option('display.max_columns', 32)

# Display the available columns in the dataset
print("Available columns in the dataset:")
print(df.columns)

# Calculate unique values for 'hotel' column
hotel_unique_count = df['hotel'].nunique()
print(f"Number of unique values in the 'hotel' column: {hotel_unique_count}")

# Calculate unique values for 'is_canceled' column
is_canceled_unique_count = df['is_canceled'].nunique()
print(f"Number of unique values in the 'is_canceled' column: {is_canceled_unique_count}")

# Plot: Number of bookings by hotel type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='hotel', hue='is_canceled')
plt.title('Number of Bookings by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('Number of Bookings')
plt.legend(title='Was the booking canceled?', labels=['No', 'Yes'])
plt.show()

# Check for missing values in the dataset
print("Checking for missing values in the dataset:")
print(df.isnull().sum())

# Replacing missing values with 0 in columns 'company' and 'agent'
df[['company', 'agent']] = df[['company', 'agent']].fillna(0)

# Replace missing values in 'children' column with the mean value
df['children'].fillna(df['children'].mean(), inplace=True)

# Replace missing values in 'country' column with the mode
df['country'].fillna(df['country'].mode()[0], inplace=True)

# Checking for missing values after replacement
print("Checking for missing values after replacement:")
print(df.isnull().sum())

# Filter subset where all guest counts (adults, children, babies) are zero
subset = df[(df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0)]
print("Subset of data with no guests (adults, children, babies):")
print(subset[['adults', 'children', 'babies']])

# Deleting rows where no guests (adults, children, babies)
df_cleaned = df[~((df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0))]
print("Data after removing rows with no guests:")
print(df_cleaned.head())

# Dropping duplicate values
df_cleaned.drop_duplicates(inplace=True)
print(f"Shape of the data after dropping duplicates: {df_cleaned.shape}")

# Convert 'children', 'company', and 'agent' to integer type
df_cleaned[['children', 'company', 'agent']] = df_cleaned[['children', 'company', 'agent']].astype('int64')

# Convert 'reservation_status_date' to datetime format
df_cleaned['reservation_status_date'] = pd.to_datetime(df_cleaned['reservation_status_date'], format='%Y-%m-%d')

# Adding a new column for total stay days
df_cleaned['total_stay'] = df_cleaned['stays_in_weekend_nights'] + df_cleaned['stays_in_week_nights']

# Adding a new column for total number of people
df_cleaned['total_people'] = df_cleaned['adults'] + df_cleaned['children'] + df_cleaned['babies']

# Remove outliers: Drop rows where 'adr' > 5000
df_cleaned.drop(df_cleaned[df_cleaned['adr'] > 5000].index, inplace=True)

# Select numerical columns to analyze correlations
num_df1 = df_cleaned[['lead_time', 'previous_cancellations', 'previous_bookings_not_canceled', 
                      'booking_changes', 'days_in_waiting_list', 'adr', 
                      'required_car_parking_spaces', 'total_of_special_requests', 
                      'total_stay', 'total_people']]

# Calculate correlation matrix
corrmat = num_df1.corr()

# Plot heatmap of correlations
plt.figure(figsize=(12, 7))
sns.heatmap(corrmat, annot=True, fmt='.2f', annot_kws={'size': 10}, vmax=.8, square=True)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Scatter plot: Relationship between 'adr' and 'total_stay' after removing outliers
plt.figure(figsize=(12, 6))
sns.scatterplot(y='adr', x='total_stay', data=df_cleaned)
plt.title('Scatter Plot of ADR vs Total Stay (Outliers Removed)')
plt.xlabel('Total Stay (Days)')
plt.ylabel('Average Daily Rate (ADR)')
plt.show()

# Analyzing bookings by agent
d1 = pd.DataFrame(df_cleaned['agent'].value_counts()).reset_index().rename(columns={'index': 'agent', 'agent': 'num_of_bookings'}).sort_values(by='num_of_bookings', ascending=False)
d1.drop(d1[d1['agent'] == 0].index, inplace=True)  # 0 represents that booking is not made by an agent
d1 = d1[:10]  # Selecting top 10 performing agents

# Plotting top 10 agents
plt.figure(figsize=(10, 5))
sns.barplot(x='agent', y='num_of_bookings', data=d1, order=d1.sort_values('num_of_bookings', ascending=False).agent)
plt.title('Top 10 Agents by Number of Bookings')
plt.xlabel('Agent')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45)
plt.show()

# Analyzing assigned room types and ADR
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Grouping by assigned room type and counting bookings
grp_by_room = df_cleaned.groupby('assigned_room_type')
d1['Num_of_bookings'] = grp_by_room.size()

# Count plot for assigned room types
sns.countplot(ax=axes[0], x=df_cleaned['assigned_room_type'])
axes[0].set_title('Number of Bookings by Assigned Room Type')
axes[0].set_xlabel('Assigned Room Type')
axes[0].set_ylabel('Number of Bookings')

# Box plot for ADR by assigned room types
sns.boxplot(ax=axes[1], x=df_cleaned['assigned_room_type'], y=df_cleaned['adr'])
axes[1].set_title('ADR by Assigned Room Type')
axes[1].set_xlabel('Assigned Room Type')
axes[1].set_ylabel('Average Daily Rate (ADR)')

plt.tight_layout()
plt.show()

# Count plot for meal types
plt.figure(figsize=(10, 8))
sns.countplot(x=df_cleaned['meal'])
plt.title('Number of Bookings by Meal Type')
plt.xlabel('Meal Type')
plt.ylabel('Number of Bookings')
plt.show()
# Univariate Analysis

# Q1) Which agent makes most number of bookings?
d1 = pd.DataFrame(df_cleaned['agent'].value_counts()).reset_index().rename(columns={'index': 'agent', 'agent': 'num_of_bookings'}).sort_values(by='num_of_bookings', ascending=False)
d1.drop(d1[d1['agent'] == 0].index, inplace=True)  # 0 represents that booking is not made by an agent
d1 = d1[:10]  # Selecting top 10 performing agents

# Plotting the top 10 agents by number of bookings
plt.figure(figsize=(10, 5))
sns.barplot(x='agent', y='num_of_bookings', data=d1, order=d1.sort_values('num_of_bookings', ascending=False).agent)
plt.title('Top 10 Agents by Number of Bookings')
plt.xlabel('Agent')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45)
plt.show()

# Analyzing meal types
plt.figure(figsize=(10, 8))
sns.countplot(x=df_cleaned['meal'])
plt.title('Number of Bookings by Meal Type')
plt.xlabel('Meal Type')
plt.ylabel('Number of Bookings')
plt.show()

# Grouping by assigned room type
grp_by_room = df_cleaned.groupby('assigned_room_type')
d1['Num_of_bookings'] = grp_by_room.size()

# Plotting assigned room types and ADR
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Count plot for assigned room types
sns.countplot(ax=axes[0], x=df_cleaned['assigned_room_type'])
axes[0].set_title('Number of Bookings by Assigned Room Type')
axes[0].set_xlabel('Assigned Room Type')
axes[0].set_ylabel('Number of Bookings')

# Box plot for ADR by assigned room types
sns.boxplot(ax=axes[1], x=df_cleaned['assigned_room_type'], y=df_cleaned['adr'])
axes[1].set_title('ADR by Assigned Room Type')
axes[1].set_xlabel('Assigned Room Type')
axes[1].set_ylabel('Average Daily Rate (ADR)')

plt.tight_layout()
plt.show()
# Selecting and counting number of cancelled bookings for each hotel
cancelled_data = df_cleaned[df_cleaned['is_canceled'] == 1]
cancel_grp = cancelled_data.groupby('hotel')
D1 = pd.DataFrame(cancel_grp.size()).rename(columns={0: 'total_cancelled_bookings'})

# Counting total number of bookings for each type of hotel
grouped_by_hotel = df_cleaned.groupby('hotel')
total_booking = grouped_by_hotel.size()
D2 = pd.DataFrame(total_booking).rename(columns={0: 'total_bookings'})

# Concatenating total cancelled bookings and total bookings dataframes
D3 = pd.concat([D1, D2], axis=1)

# Calculating cancel percentage 
D3['cancel_%'] = round((D3['total_cancelled_bookings'] / D3['total_bookings']) * 100, 2)

# Displaying the final DataFrame with cancellation percentages
print(D3)

# Plotting the cancellation percentage for each hotel
plt.figure(figsize=(10, 5))
sns.barplot(x=D3.index, y=D3['cancel_%'])
plt.title('Cancellation Percentage by Hotel')
plt.xlabel('Hotel')
plt.ylabel('Cancellation Percentage (%)')
plt.xticks(rotation=45)
plt.show()
# Selecting and counting repeated customers bookings
repeated_data = df_cleaned[df_cleaned['is_repeated_guest'] == 1]
repeat_grp = repeated_data.groupby('hotel')
D1 = pd.DataFrame(repeat_grp.size()).rename(columns={0: 'total_repeated_guests'})

# Counting total bookings for each hotel
total_booking = grouped_by_hotel.size()
D2 = pd.DataFrame(total_booking).rename(columns={0: 'total_bookings'})

# Concatenating total repeated guests and total bookings dataframes
D3 = pd.concat([D1, D2], axis=1)

# Calculating repeat percentage
D3['repeat_%'] = round((D3['total_repeated_guests'] / D3['total_bookings']) * 100, 2)

# Displaying the final DataFrame with repeat percentages
print(D3)

# Plotting the repeat percentage for each hotel
plt.figure(figsize=(10, 5))
sns.barplot(x=D3.index, y=D3['repeat_%'])
plt.title('Repeat Customers Percentage by Hotel')
plt.xlabel('Hotel')
plt.ylabel('Repeat Customers Percentage (%)')
plt.xticks(rotation=45)
plt.show()
# Grouping by distribution channel and calculating booking percentage
group_by_dc = df1.groupby('distribution_channel')
d1 = pd.DataFrame(round((group_by_dc.size() / df1.shape[0]) * 100, 2)).reset_index().rename(columns={0: 'Booking_%'})

# Plotting pie chart for booking percentage by distribution channel
plt.figure(figsize=(8, 8))
data = d1['Booking_%']
labels = d1['distribution_channel']
plt.pie(x=data, autopct="%.2f%%", explode=[0.05] * len(labels), labels=labels, pctdistance=0.5)
plt.title("Booking % by Distribution Channels", fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
plt.show()

# Grouping by distribution channel to calculate median lead time
d2 = pd.DataFrame(round(group_by_dc['lead_time'].median(), 2)).reset_index().rename(columns={'lead_time': 'median_lead_time'})

# Plotting bar chart for median lead time by distribution channel
plt.figure(figsize=(7, 5))
sns.barplot(x=d2['distribution_channel'], y=d2['median_lead_time'])
plt.title('Median Lead Time by Distribution Channel')
plt.ylabel('Median Lead Time')
plt.xlabel('Distribution Channel')
plt.show()

# Grouping by both distribution channel and hotel to calculate average adr
group_by_dc_hotel = df1.groupby(['distribution_channel', 'hotel'])
d5 = pd.DataFrame(round(group_by_dc_hotel['adr'].agg(np.mean), 2)).reset_index().rename(columns={'adr': 'avg_adr'})

# Plotting bar chart for average adr by distribution channel and hotel
plt.figure(figsize=(7, 5))
sns.barplot(x=d5['distribution_channel'], y=d5['avg_adr'], hue=d5['hotel'])
plt.ylim(40, 140)  # Set y-axis limit for better visibility
plt.title('Average ADR by Distribution Channel and Hotel')
plt.ylabel('Average ADR')
plt.xlabel('Distribution Channel')
plt.legend(title='Hotel')
plt.show()
# Grouping by distribution channel and calculating booking percentage
group_by_dc = df1.groupby('distribution_channel')
d1 = pd.DataFrame(round((group_by_dc.size() / df1.shape[0]) * 100, 2)).reset_index().rename(columns={0: 'Booking_%'})

# Plotting pie chart for booking percentage by distribution channel
plt.figure(figsize=(8, 8))
data = d1['Booking_%']
labels = d1['distribution_channel']
plt.pie(x=data, autopct="%.2f%%", explode=[0.05] * len(labels), labels=labels, pctdistance=0.5)
plt.title("Booking % by Distribution Channels", fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
plt.show()

# Grouping by distribution channel to calculate median lead time
d2 = pd.DataFrame(round(group_by_dc['lead_time'].median(), 2)).reset_index().rename(columns={'lead_time': 'median_lead_time'})

# Plotting bar chart for median lead time by distribution channel
plt.figure(figsize=(7, 5))
sns.barplot(x=d2['distribution_channel'], y=d2['median_lead_time'])
plt.title('Median Lead Time by Distribution Channel')
plt.ylabel('Median Lead Time')
plt.xlabel('Distribution Channel')
plt.show()

# Grouping by both distribution channel and hotel to calculate average adr
group_by_dc_hotel = df1.groupby(['distribution_channel', 'hotel'])
d5 = pd.DataFrame(round(group_by_dc_hotel['adr'].agg(np.mean), 2)).reset_index().rename(columns={'adr': 'avg_adr'})

# Plotting bar chart for average adr by distribution channel and hotel
plt.figure(figsize=(7, 5))
sns.barplot(x=d5['distribution_channel'], y=d5['avg_adr'], hue=d5['hotel'])
plt.ylim(40, 140)  # Set y-axis limit for better visibility
plt.title('Average ADR by Distribution Channel and Hotel')
plt.ylabel('Average ADR')
plt.xlabel('Distribution Channel')
plt.legend(title='Hotel')
plt.show()
d1 = pd.DataFrame((group_by_dc['is_canceled'].sum()/group_by_dc.size())*100).drop(index = 'Undefined').rename(columns = {0: 'Cancel_%'})
plt.figure(figsize = (10,5))
sns.barplot(x = d1.index, y = d1['Cancel_%'])
plt.show()# Plotting bar chart for average adr by distribution channel and hotel
plt.figure(figsize=(7, 5))
sns.barplot(x=d5['distribution_channel'], y=d5['avg_adr'], hue=d5['hotel'])
plt.ylim(40, 140)  # Set y-axis limit for better visibility
plt.title('Average ADR by Distribution Channel and Hotel')
plt.ylabel('Average ADR')
plt.xlabel('Distribution Channel')
plt.legend(title='Hotel')
plt.show()

# Calculating cancellation percentage for each distribution channel (excluding 'Undefined')
d1 = pd.DataFrame((group_by_dc['is_canceled'].sum() / group_by_dc.size()) * 100).drop(index='Undefined').rename(columns={0: 'Cancel_%'})

# Plotting bar chart for cancellation percentage by distribution channel
plt.figure(figsize=(10, 5))
sns.barplot(x=d1.index, y=d1['Cancel_%'])
plt.title('Cancellation % by Distribution Channel')
plt.ylabel('Cancellation %')
plt.xlabel('Distribution Channel')
plt.show()

# Plotting boxplot for ADR based on whether the same room was not allotted
plt.figure(figsize=(12, 6))
sns.boxplot(x='same_room_not_alloted', y='adr', data=df1)
plt.title('ADR vs Same Room Not Allotted')
plt.ylabel('ADR')
plt.xlabel('Same Room Not Allotted (1 = Yes, 0 = No)')
plt.show()
# Counting the number of guests arriving in each month
d_month = df1['arrival_date_month'].value_counts().reset_index()
d_month.columns = ['months', 'Number of guests']

# Ordering the months for proper chronological sorting
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
d_month['months'] = pd.Categorical(d_month['months'], categories=months, ordered=True)
d_month = d_month.sort_values('months').reset_index(drop=True)

# Calculating the average adr (price) for each month for both Resort Hotel and City Hotel, excluding canceled bookings
data_resort = df1[(df1['hotel'] == 'Resort Hotel') & (df1['is_canceled'] == 0)]
data_city = df1[(df1['hotel'] == 'City Hotel') & (df1['is_canceled'] == 0)]

resort_hotel = data_resort.groupby('arrival_date_month')['adr'].mean().reset_index()
city_hotel = data_city.groupby('arrival_date_month')['adr'].mean().reset_index()

# Merging the data for resort and city hotels
final_hotel = resort_hotel.merge(city_hotel, on='arrival_date_month')
final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']

# Counting the number of guests in each hotel type for each month
resort_guest = data_resort['arrival_date_month'].value_counts().reset_index()
resort_guest.columns = ['month', 'no_of_guests_in_resort']

city_guest = data_city['arrival_date_month'].value_counts().reset_index()
city_guest.columns = ['month', 'no_of_guests_in_city_hotel']

# Merging the guest counts for both hotel types
final_guest = resort_guest.merge(city_guest, on='month')
final_guest.columns = ['month', 'no_of_guests_in_resort', 'no_of_guests_in_city_hotel']

# Ordering the months in the final_guest DataFrame
final_guest['month'] = pd.Categorical(final_guest['month'], categories=months, ordered=True)
final_guest = final_guest.sort_values('month').reset_index(drop=True)

# Plotting the number of guests in resort and city hotels for each month
plt.figure(figsize=(15, 10))
sns.lineplot(data=final_guest, x='month', y='no_of_guests_in_resort', label='Resort Hotel')
sns.lineplot(data=final_guest, x='month', y='no_of_guests_in_city_hotel', label='City Hotel')
plt.legend(title='Hotel Type')
plt.ylabel('Number of Guests')
plt.title('Number of Guests by Month in Resort and City Hotels')
plt.xticks(rotation=45)  # Rotate month labels for better readability
plt.show()

data_resort = df1[(df1['hotel'] == 'Resort Hotel') & (df1['is_canceled'] == 0)]
data_city = df1[(df1['hotel'] == 'City Hotel') & (df1['is_canceled'] == 0)]

resort_hotel = data_resort.groupby('arrival_date_month')['adr'].mean().reset_index()
city_hotel = data_city.groupby('arrival_date_month')['adr'].mean().reset_index()


final_hotel = resort_hotel.merge(city_hotel, on='arrival_date_month')
final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']


months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
final_hotel['month'] = pd.Categorical(final_hotel['month'], categories=months, ordered=True)
final_hotel = final_hotel.sort_values('month').reset_index(drop=True)


plt.figure(figsize=(15, 10))
sns.lineplot(data=final_hotel, x='month', y='price_for_resort', label='Resort Hotel')
sns.lineplot(data=final_hotel, x='month', y='price_for_city_hotel', label='City Hotel')
plt.legend(title='Hotel Type')
plt.ylabel('Average ADR')
plt.xlabel('Month')
plt.title('Average ADR by Month for Resort and City Hotels')
plt.xticks(rotation=45) 
plt.show()

reindex = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df1['arrival_date_month'] = pd.Categorical(df1['arrival_date_month'], categories=reindex, ordered=True)


plt.figure(figsize=(15, 8))
sns.boxplot(x=df1['arrival_date_month'], y=df1['adr'])
plt.title('ADR Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Average Daily Rate (ADR)')
plt.xticks(rotation=45)
plt.show()


d6 = pd.DataFrame(not_canceled.groupby('arrival_date_day_of_month').size()).rename(columns={0: 'Arrival_num'})
d6['avg_adr'] = not_canceled.groupby('arrival_date_day_of_month')['adr'].agg(np.mean)


fig, axes = plt.subplots(1, 2, figsize=(18, 8))


g = sns.lineplot(ax=axes[0], x=d6.index, y=d6['Arrival_num'])
g.grid()
g.set_xticks([1, 7, 14, 21, 28, 31])
g.set_xticklabels([1, 7, 14, 21, 28, 31])
g.set_title('Number of Arrivals by Day of Month')
g.set_xlabel('Day of Month')
g.set_ylabel('Number of Arrivals')

h = sns.lineplot(ax=axes[1], x=d6.index, y=d6['avg_adr'])
h.grid()
h.set_xticks([1, 7, 14, 21, 28, 31])
h.set_xticklabels([1, 7, 14, 21, 28, 31])
h.set_title('Average ADR by Day of Month')
h.set_xlabel('Day of Month')
h.set_ylabel('Average ADR')

plt.tight_layout()
plt.show()
# Special Request by Market Segment
sns.boxplot(x="market_segment", y="total_of_special_requests", hue='market_segment', data=df1)
plt.title('Special Requests by Market Segment')
plt.xlabel('Market Segment')
plt.ylabel('Total Special Requests')
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

# Special Requests by Number of Kids
df1['kids'] = df1['children'] + df1['babies']
sns.barplot(x="kids", y="total_of_special_requests", data=df1)
plt.title('Special Requests by Number of Kids')
plt.xlabel('Number of Kids')
plt.ylabel('Total Special Requests')
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

# Guests by Country
grouped_by_country = df1.groupby('country')
d1 = pd.DataFrame(grouped_by_country.size()).reset_index().rename(columns={0: 'Count'}).sort_values('Count', ascending=False)[:10]
sns.barplot(x=d1['country'], y=d1['Count'])
plt.title('Top 10 Countries by Number of Guests')
plt.xlabel('Country')
plt.ylabel('Number of Guests')
plt.show()

# Stay Duration Analysis
data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data.head()
stay = data.groupby(['total_nights', 'hotel']).agg('count').reset_index()
stay = stay.iloc[:, :3]  
stay = stay.rename(columns={'is_canceled': 'Number of stays'}) 
stay
plt.figure(figsize=(10, 5))
sns.barplot(x='total_nights', y='Number of stays', data=stay, hue='hotel')
plt.title('Number of Stays by Total Nights and Hotel')
plt.xlabel('Total Nights')
plt.ylabel('Number of Stays')
plt.legend(title='Hotel')
plt.show()
