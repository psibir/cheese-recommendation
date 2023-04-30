import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from tabulate import tabulate
import os

class CheeseRecommender:
    def __init__(self, cheese_file):
        self.df = pd.read_csv(cheese_file, sep='\t')
        self.vectorizer = TfidfVectorizer()

    def get_recommendations(self, user_input=None, num_recommendations=5, start_index=0, exclude_words=[]):
        self.df.fillna('', inplace=True)
        cheese_desc = self.df.apply(lambda x: ' '.join(x), axis=1)
        cheese_matrix = self.vectorizer.fit_transform(cheese_desc)
        if user_input is None:
            sim_indices = random.sample(range(len(self.df)), num_recommendations)
        else:
            user_vector = self.vectorizer.transform([user_input])
            sim_scores = cosine_similarity(user_vector, cheese_matrix).flatten()
            sim_indices = sim_scores.argsort()[::-1][start_index:num_recommendations+start_index]
        recommendations = self.df.iloc[sim_indices][['cheese', 'milk', 'origin', 'region', 'kind', 'color', 'texture', 'flavor', 'aroma', 'description', 'producer']]
        recommendations['origin'] = recommendations['origin'].str.split(',').str[0] # only select the first item in the "origin" column
        if exclude_words:
            for word in exclude_words:
                recommendations = recommendations[~recommendations['description'].str.contains(word, case=False)]
        return recommendations

if __name__ == '__main__':
    cheese_file = 'cheeses_tab.tsv'
    recommender = CheeseRecommender(cheese_file)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
        user_input = input('What qualities do you like in cheese? (Press Enter to skip) ')
        exclude_words = input('Enter any qualities you do not want in your cheese (press Enter to skip): ')
        if exclude_words:
            exclude_words = exclude_words.strip().split()
        num_recommendations = 5
        start_index = 0
        if not user_input and not exclude_words:
            print("Here are some random cheeses:")
            recommendations = recommender.get_recommendations()
            print(tabulate(recommendations[['cheese', 'origin', 'texture', 'flavor', 'aroma']], headers='keys', tablefmt='psql', showindex=False))
        else:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
                recommendations = recommender.get_recommendations(user_input=user_input, num_recommendations=num_recommendations, start_index=start_index, exclude_words=exclude_words)
                recommendations.fillna('', inplace=True)
                if recommendations.empty:
                    print("No more recommendations.")
                    break
                print(tabulate(recommendations[['cheese', 'origin', 'texture', 'flavor', 'aroma']], headers='keys', tablefmt='psql', showindex=False))
                more_info = input('Would you like more information on any of these cheeses? (y/n) ')
                if more_info.lower() == 'y':
                    for index, row in recommendations.iterrows():
                        cheese = row['cheese']
                        origin = row['origin'].strip()
                        region = row['region'].strip()
                        kind = row['kind'].strip()
                        milk = row['milk'].strip()
                        producer = row['producer'].strip()
                        description = row['description'].strip()
                        print(f"\n{cheese}")
                        if origin:
                            print(f"Origin: {origin}")
                        if region:
                            print(f"Region: {region}")
                        if kind:
                            print(f"Kind: {kind}")
                        if milk:
                            print(f"Milk: {milk}")
                        if producer:
                            print(f"Producer: {producer}")
                        print(f"Description: {description}")
                next_recommendations = input('Enter "m" to see more recommendations, "r" to restart, or press Enter to quit. ')
                if next_recommendations.lower() == 'm':
                    start_index += num_recommendations
                elif next_recommendations.lower() == 'r':
                    break
                else:
                    sys.exit(0)
