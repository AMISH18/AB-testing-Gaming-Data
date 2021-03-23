#!/usr/bin/env python
# coding: utf-8

# ## AB testing – Cookie Game (a Mobile Game)

# Importing required libraries

# In[1]:


import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn 
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2


# Changing the working directory

# In[2]:


os.chdir("D:\MS Business Analytics – University of Cincinnati\Spectre\Project 1 - AB Testing")


# Loading the overall dataset into a variable named "game"

# In[3]:


game = pd.read_csv("cookie_cats.csv")


# Let's have a look at the dataset

# In[4]:


game.head()


# In[5]:


game.describe()


# In[6]:


game.columns


# ## Conducting basic EDA of the data to understand what all information has been provided

# In[7]:


gate_30 = game[game["version"] == "gate_30"]
gate_30.head()


# In[8]:


gate_40 = game[game["version"] == "gate_40"]
gate_40.head()


# Let's see the number of game rounds by each user

# In[9]:


gate_30.min()


# Now, Visualizing the histogram of number of gamerounds 

# In[10]:


gate_30_v = gate_30.loc[(gate_30["sum_gamerounds"] >= 800) & (gate_30["sum_gamerounds"] <= 10000)]
gate_30_v.head()


# In[11]:


plt.hist(gate_30["sum_gamerounds"])


# Clearly, we can see that the number of people who did not play the game again are very high

# Let's 2 histograms, divided by number of game rounds people played

# In[12]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15, 5))
ax1.set_title("Gamerounds between 0 and 800")
ax2.set_title("Gamerounds >= 800, but <= 10,000")
ax3.set_title("Gamerounds > 10,000")

ax1.hist(gate_30.loc[(gate_30["sum_gamerounds"] >= 0) & (gate_30["sum_gamerounds"] <= 1000)]["sum_gamerounds"])
ax2.hist(gate_30.loc[(gate_30["sum_gamerounds"] >= 1000) & (gate_30["sum_gamerounds"] <= 10000)]["sum_gamerounds"])
ax3.hist(gate_30.loc[(gate_30["sum_gamerounds"] > 10000)]["sum_gamerounds"])

plt.show()


# Clearly, we can see how number of games played went down drastically post 300

# Plotting the boxplot as well to see any outlier values in gate_30 data

# In[13]:


plt.boxplot(gate_30["sum_gamerounds"])
plt.title("Box plot for Gameorunds where gate was at level 30 of the game")
plt.show()


# Clearly, there is 1 player who played >40,000 games; Now checking the data by removing that ourlier value

# In[14]:


plt.boxplot(gate_30[gate_30["sum_gamerounds"] < 20000]["sum_gamerounds"])
plt.title("Box plot for Gameorunds where gate was at level 30 of the game")
plt.show()


# Now, looking at the other group of data, i.e. for the test group who were shown the gate at  level 40 

# In[15]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
ax1.set_title("Gamerounds")
ax1.hist(gate_40.loc[(gate_40["sum_gamerounds"] >= 0) & (gate_40["sum_gamerounds"] <= 1000)]["sum_gamerounds"])
ax2.hist(gate_40.loc[(gate_40["sum_gamerounds"] > 1000) & (gate_40["sum_gamerounds"] <= 10000)]["sum_gamerounds"])
ax3.hist(gate_40.loc[(gate_40["sum_gamerounds"] > 10000)]["sum_gamerounds"])
plt.show()


# Clearly, the trend is more or less the same here as well, as it was in the case of players who played game in which the gate was at level 30

# Now, plotting a boxplot to see if this data has any outliers

# In[16]:


plt.boxplot(gate_40["sum_gamerounds"])
plt.title("Gamerounds when gate was shown at level 40")
plt.show()


# Now, taking a look at the data for 1_day retention for players which had the gate set at level 30

# In[17]:


sns.countplot("retention_1", data = gate_30)
number_of_retained_players_1 = gate_30["retention_1"].sum()
print("Total Number of Players = ", gate_30["userid"].count())
print("number of Players which returned = ", number_of_retained_players_1)
print("Ratio of Number of Players Retained After 1 Day = ", number_of_retained_players_1/ gate_30["userid"].count())


# Now, visualizing the same thing for 7-day retention

# In[18]:


sns.countplot("retention_7", data = gate_30)
number_of_retained_players_7 = gate_30["retention_7"].sum()
print("Total Number of Players = ", gate_30["userid"].count())
print("number of Players which returned after day 7 = ", number_of_retained_players_7)
print("Ratio of Number of Players Retained After 1 Day = ", number_of_retained_players_7/ gate_30["userid"].count())


# Now, let's take a look at the data for test group

# Let's see this for group with 30 members

# In[19]:


sns.countplot("retention_1", data = gate_40)
number_of_retained_players_t1 = gate_40["retention_1"].sum()
print("Total Number of Players = ", gate_40["userid"].count())
print("number of Players which returned = ", number_of_retained_players_t1)
print("Ratio of Number of Players Retained After 1 Day = ", number_of_retained_players_t1/ gate_40["userid"].count())


# In[20]:


sns.countplot("retention_7", data = gate_40)
number_of_retained_players_t7 = gate_40["retention_7"].sum()
print("Total Number of Players = ", gate_40["userid"].count())
print("number of Players which returned = ", number_of_retained_players_t7)
print("Ratio of Number of Players Retained After 1 Day = ", number_of_retained_players_t7/ gate_40["userid"].count())


# ## Now, we are trying to understand if the distribution we have is statistically significant

# If we compare both retention rates, we see that by shifting the gate to next level actually decreased the average 1-day, as well as 7-day retention of users

# Now, we need to understand if this distribution of both 1-day and 7-day retention is statistically significant or not

# To understand the same, we will use chi-square test

# Let's state our null and alternative hypothesis

# <b> Null Hypothesis (Ho): There is no chnage in the 1-day retention of players playing the cookie cats game, if gate is moved from level 30 to level 40
# 
# Alternative Hypothesis (H1): There is change in the 1-day retention of players playing the cookie cats game, if gate is moved from level 30 to level 40 </b>
# 
# Confidence Interval is 95%

# Now, let's do this for 1-day retention first; creating the contingency matrix for chi square test, we will have 4 values , gate_30 - retained, gate_30 - not retained, gate_40 - retained, gate_40 - not retained

# In[21]:


Contigency_matrix = np.array([[gate_30["retention_1"].sum(), gate_30["userid"].count() - gate_30["retention_1"].sum()], [gate_40["retention_1"].sum(), gate_40["userid"].count() - gate_40["retention_1"].sum()]])
print(Contigency_matrix)


# In[22]:


stats.chi2_contingency(Contigency_matrix)


# Now, we get the p-value for this test ~7.6%; with alpha value = 5%, we cannot reject the null hypothesis, hence, we conclude that for 1-day retention, changing the gate presence from 30 to 40 has no effect on retention

# Now, completing this test for 7-day retention

# In[23]:


Contigency_matrix_7 = np.array([[gate_30["retention_7"].sum(), gate_30["userid"].count() - gate_30["retention_7"].sum()], [gate_40["retention_7"].sum(), gate_40["userid"].count() - gate_40["retention_7"].sum()]])
print(Contigency_matrix_7)


# In[24]:


stats.chi2_contingency(Contigency_matrix_7)


# Now, here we have p-value < 0.05, hence, the we can reject the null hypothesis and say that there is change in 7-day retention of players if gate is moved form level 30 to level 40

# ## Now, computing the same results using chi-square critical value for both 1-day and 7-day retention

# We have degrees of freedom for both the case as df = 1

# In[25]:


critical_value=chi2.ppf(q=1-0.05,df=1)
print(critical_value)


# Now, we can see that for 1-day retention, chi-square observed < chi-square critical, hence we cannot reject null hypothesis;
# However, in case of 7-day retention, chi-square observed > chi-square critical, hence, we can reject the null hypothesis and accept alternative hypothesis

# Hence, as 7-day retention declined in this case, we may state that we should not change the level of gate from 30 to 40 
