#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import os
import numpy as np
from preprocesser import yelp_preprocess
from sklearn.externals import joblib
import json
import copy

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class SearchHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("search.html")

class UserPageHandler(tornado.web.RequestHandler):
    def get(self):
        """
        user_page_data = yelp_preprocess.get_user_page_data(0,
                                           self.application.limit_uid_to_reviews_map,
                                           self.application.uid_to_user_map,
                                           self.application.bid_to_business_map)
        """
        uid = self.get_argument('uid',default=0)
        user_page_data = self.application.user_page_datas[int(uid)%len(self.application.user_page_datas)]
        user = user_page_data['user']
        user['friend_cnt'] = len(user['friends'])
        rec_records = copy.deepcopy(user_page_data['rec_records'])
        decrement = abs(np.random.normal(0,0.03))
        for i in xrange(len(rec_records)):
            rec_records[i]['stars'] = "%.3f"%(rec_records[i]['stars']-decrement)
            decrement += abs(np.random.normal(0,0.03))
        visit_records = copy.deepcopy(user_page_data['visit_records'])

        self.render("user.html", user=user, recs=rec_records, hists=visit_records, uid=uid)

def run_server(port):
    app = tornado.web.Application(
	    handlers=[(r'/', IndexHandler), (r'/user', UserPageHandler), (r'/search', SearchHandler)],
	    template_path=os.path.join(os.path.dirname(__file__), "templates"),
	    static_path=os.path.join(os.path.dirname(__file__), "static"),
	    debug=True
	)

    #f = open("./dataset/recommend/yelp/server_data.json")
    #server_data = json.load(f)
    #app.limit_uid_to_reviews_map, app.uid_to_user_map, app.bid_to_business_map = server_data
    user_page_data_10 = joblib.load('./dataset/recommend/yelp/user_page_data_10.pkl')
    app.user_page_datas = user_page_data_10
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    run_server()