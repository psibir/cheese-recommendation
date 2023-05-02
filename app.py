from flask import Flask, request, jsonify
from recommender import CheeseRecommender

app = Flask(__name__)
cheese_file = 'cheeses_tab.tsv'
recommender = CheeseRecommender(cheese_file)

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_input = request.args.get('user_input', default=None, type=str)
    num_recommendations = request.args.get('num_recommendations', default=5, type=int)
    start_index = request.args.get('start_index', default=0, type=int)
    exclude_words = request.args.get('exclude_words', default='', type=str).strip().split()
    recommendations = recommender.get_recommendations(user_input=user_input, num_recommendations=num_recommendations, start_index=start_index, exclude_words=exclude_words)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
