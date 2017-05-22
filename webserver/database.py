#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

import pymongo
import json
import numpy as np
from util.log import log_and_print
prefix = "dataset/recommend/yelp/"
academic_dataset_json_prefix = "yelp_academic_dataset_%s.json"
recommendation_path = prefix + "SVD_yelp_top20_results.txt"

def get_db():
	client = pymongo.MongoClient("localhost", 27017)
	db = client['yelp']
	return db

def handle_rec_line(line):
	items = line[:-1].split(": ")
	uid = items[0]
	business_pairs = []
	for item in items[1].split(' '):
		item = item[1:-1].split(',')
		bid = item[0]
		try:
			rate = float(item[1])
		except ValueError,e :
			rate = float(item[1][:-3])
			print "Error:"+item[1]
		business_pairs.append((bid,rate))
	return uid, business_pairs

def insert_data():
	db = get_db()

	log_and_print("Remove all collections")
	db.users.remove()
	db.businesses.remove()
	db.reviews.remove()
	db.recommendations.remove()

	log_and_print("Insert reviews")
	for i, line in enumerate(open(prefix+academic_dataset_json_prefix%("review"))):
		review = json.loads(line)
		review['_id'] = i+1
		db.reviews.insert(review)
		if i%50000 == 0:
			log_and_print("Insert review %d"%i)

	log_and_print("Insert users")
	for i, line in enumerate(open(prefix+academic_dataset_json_prefix%("user"))):
		user = json.loads(line)
		user['_id'] = i+1
		db.users.insert(user)
		if i%10000 == 0:
			log_and_print("Insert user %d"%i)

	log_and_print("Insert businesses")
	for i, line in enumerate(open(prefix+academic_dataset_json_prefix%("business"))):
		business = json.loads(line)
		business['_id'] = i+1
		db.businesses.insert(business)
		if i%10000 == 0:
			log_and_print("Insert business %d"%i)

	insert_recommendations(db)


def insert_recommendations(db):
	f = open(recommendation_path)
	f.readline()
	log_and_print("Insert recommendations")
	for i, line in enumerate(f):
		try:
			uid, business_pairs = handle_rec_line(line)
			db.recommendations.insert({
				"_id": i+1,
				"user_id": uid,
				"business_pairs": business_pairs
			})
			if i % 10000 == 0:
				log_and_print("Insert recommendation %d" % i)
		except Exception, e:
			print e
			print line




def get_page_data(u_idx):
	db = get_db()
	rec_result = db.recommendations.find_one({'_id':u_idx})
	user = db.users.find_one({'user_id':rec_result['user_id']})
	hist_reviews = db.reviews.find({'user_id':user['user_id']}).sort("stars", pymongo.DESCENDING).limit(10)
	hist_reviews = [r for r in hist_reviews]
	hist_businesses = [db.businesses.find_one({'business_id':review['business_id']}) for review in hist_reviews]

	for i in xrange(len(hist_reviews)):
		hist_reviews[i]['business'] = hist_businesses[i]

	pairs = rec_result['business_pairs']
	rec_businesses = [db.businesses.find_one({'business_id':pair[0]}) for pair in pairs]
	rec_scores = [pair[1] for pair in pairs]

	max_rec_stars = float(rec_scores[0])
	print max_rec_stars, max_rec_stars-5
	for i in xrange(len(rec_businesses)):
		rec_businesses[i]['rec_stars'] = round(float(max(min(5*float(rec_scores[i])/max(5, max_rec_stars) ,5) ,1)),3)
	if max_rec_stars < 4 and rec_businesses[0]['rec_stars'] == rec_businesses[8]['rec_stars']:
		add = 0
		np.random.seed(u_idx)
		add_arr = [abs(np.random.normal(0,0.02)) for i in range(len(rec_businesses))]
		for i in reversed(range(len(rec_businesses))):
			add += add_arr[i]
			rec_businesses[i]['rec_stars'] += add
			rec_businesses[i]['rec_stars'] = min(5, rec_businesses[i]['rec_stars'])
			rec_businesses[i]['rec_stars'] = round(rec_businesses[i]['rec_stars'], 3)
	page_data = {
		"user":user,
		"hist_reviews":hist_reviews,
		"hist_businesses":hist_businesses,
		"rec_businesses":rec_businesses,
		"rec_scores":rec_scores
	}
	return page_data

if __name__ == "__main__":
	insert_recommendations(get_db())


