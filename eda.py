# eda.py
import pandas as pd

# Loading the data
df = pd.read_csv('customer_support_tickets.csv')

print(df.head())

print(df.columns)

# checking whether classes are balanced
for i in df.columns:
    print(df[i].value_counts())

print(df.dtypes)

print('Null values per column:')
print(df.isnull().sum())

# Preview of ticket subjects
print(df["Ticket Subject"].head(20))
print(df['Ticket Subject'].nunique())

# Ticket Subject and Ticket type are the only two columns needed to train our classifier.

print(df['Ticket Subject'].value_counts())
print(df['Ticket Type'].value_counts())



df["text_length"] = df["Ticket Subject"].apply(lambda x: len(x.split()))

print(df["text_length"].describe())

import matplotlib.pyplot as plt

df["text_length"].hist(bins=20)
plt.title("Ticket Subject Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# The number of words in almost every ticket subject is 2 words.

df["desc_length"] = df["Ticket Description"].apply(lambda x: len(str(x).split()))
print(df["desc_length"].describe())

df["desc_length"].hist(bins=20)
plt.title("Ticket Description Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()