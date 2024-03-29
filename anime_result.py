import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
#import os

#cwd= os.getcwd()
anime_path= "anime_mal_database.csv"
anime_p_path= "anime_processed.csv"


anime_df= pd.read_csv(anime_path, header=0)
anime_df = anime_df[anime_df['title'].notna()]
anime_df = anime_df[anime_df['genres'].notna()]
anime_df= anime_df.fillna("Not Available")
anime_df= anime_df.set_index('title')


anime_test= pd.read_csv(anime_p_path, header=0)
anime_test= anime_test.set_index('title')
columns_to_drop= anime_test.columns
anime_test['features']= anime_test.values.tolist()
anime_test= anime_test.drop(columns_to_drop, axis=1)
anime_test= anime_test.reset_index()
anime_test.head()

#print(anime_df.head())

def get_suggestions(query_anime):
    
    def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
        # Initialize matrix of zeros
        rows = len(s)+1
        cols = len(t)+1
        distance = np.zeros((rows,cols),dtype = int)

        # Populate matrix of zeros with the indeces of each character of both strings
        for i in range(1, rows):
            for k in range(1,cols):
                distance[i][0] = i
                distance[0][k] = k

        # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row-1] == t[col-1]:
                    cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
                else:
                    # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                    # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                    if ratio_calc == True:
                        cost = 2
                    else:
                        cost = 1
                distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                     distance[row][col-1] + 1,          # Cost of insertions
                                     distance[row-1][col-1] + cost)     # Cost of substitutions
        if ratio_calc == True:
            # Computation of the Levenshtein Distance Ratio
            Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
            return Ratio
        else:
            pass


    def find_close_match(query):
   
        close_list=[]
       
        for anime in anime_names:
            ratio = levenshtein_ratio_and_distance(query, anime,ratio_calc = True)
            close_list.append([anime, ratio])
           
        close_df= pd.DataFrame(close_list, columns=['Anime', 'Ratio'])
        close_df = close_df.sort_values('Ratio', ascending=False)
        close_df= close_df.drop(['Ratio'], axis=1)
        return close_df.head(5)


    def get_result(test):
        anime_result_list=[]
        for row in test.iterrows():
            #print('Similar anime like {}:'.format(row[1]['title']))
            anime_result_list.append(row[1]['title'])
            search = anime_test.drop([row[0]]) # drop current anime
            search['result'] = search['features'].apply(lambda x: cosine_similarity([row[1]['features']], [x]))
            search_result = search.sort_values('result', ascending=False)['title'].head(10)
            for res in search_result.values:
                #print('\t{}'.format(res))
                anime_result_list.append(res)
            #print()
        return(anime_result_list)
    anime_names= anime_test['title'].unique()
    test= anime_test[anime_test['title']== query_anime]

    #print(query_anime)
    if len(test) != 0:
        anime_recommendation_list=get_result(test)
        anime_result_final= anime_df.ix[anime_recommendation_list]
        flag= 0
        anime_result_final= anime_result_final.drop(['scored by', 'rank',   'popularity',	'members',	'favorites'], axis=1)	
        anime_result_final= anime_result_final.reset_index()

        anime_result_final= anime_result_final.values.tolist()
        #print(anime_result_final)
        
        return (anime_result_final, flag, query_anime)
    else:
        anime_close_match_list = []
        close_match = find_close_match(query_anime)
        for i in close_match['Anime'].values:
            #print(i)
            anime_close_match_list.append(i)
        flag = 1
        anime_close_match = pd.DataFrame(anime_close_match_list, columns=['title'])
        anime_close_match= anime_close_match.values.tolist()
        #print(anime_close_match_list)
        return(anime_close_match, flag, query_anime)












    


           
    
