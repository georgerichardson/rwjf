## Scraps from the exploratory data analysis


# Trying to use labelled data to identify health innovation projects



impact_vars = ['impact category', 'primary issue']

t60h = extract_shared_variables(fields_above_thres+impact_vars,t60_dfs)

#Impact variables to focus on
impact_vars = ['impact category', 'primary issue','classifications:title']

#t60h means three sixty health
t60h = extract_shared_variables(fields_above_thres+impact_vars,t60_dfs).reset_index(drop=True)

#First some tidying up of variable names.

t60h.columns

t60h.columns = ['value','award_date','impact_1','currency','description','funder_id','funder_name','identifier',
               'impact_2','impact_3','recipient_id','recipient_name','title']

#Tokenise descriptions and remove stopwords

#Levels of this list comprehension: descriptions -> words in the description if words not in stopword list
#and np.nan if the description is a number

t60h['description_tokens'] = [
    [[w.lower() for w in x.split(" ") if w.lower() not in set(stop)] if type(x)==str else np.nan][0] for x in t60h.description]

#How will this work? 

#NOw we have a list of impact variables
impact_cats = set(t60h['impact_1'].dropna()) | set(t60h['impact_2'].dropna()) | set(t60h['impact_3'].dropna())

health_cats = {x for x in impact_cats if 'health' in x.lower()}

#Now we identify the projects in these categories

#We combine the categories in all the impact variables
t60h['impact_categories'] = [' '.join([str(x),
                                       str(y),
                                       str(z)]) for x,y,z in zip(t60h.impact_1,t60h.impact_2,t60h.impact_3)]

#Focus on the projects where we at least have one category of impact (we go down to 5k)
t60_labelled = t60h.loc[[x!='nan nan nan' for x in t60h.impact_categories]]

#Find the healthy ones
t60_labelled['has_health'] = [any(x in val for x in health_cats) for val in t60_labelled.impact_categories]


#Top words in each group
t60_labelled.groupby('has_health')['description_tokens'].apply(lambda x: pd.Series(
    [val for el in x for val in el]).value_counts()[:10])

#I don't think this is going to be very enlightening
