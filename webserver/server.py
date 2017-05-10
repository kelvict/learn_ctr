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
import database
import random

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class SearchHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("search.html")

class UserPageHandler(tornado.web.RequestHandler):
    def get(self):
        uid = self.get_argument('uid',default=0)
        if uid == 0:
            uid = random.randint(1,100000)
            self.redirect("/user?uid=%d"%uid)
        d = database.get_page_data(int(uid))
        self.render("user.html", user=d['user'], recs=d['rec_businesses'], hists=d['hist_reviews'], uid=uid)

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
    #user_page_data_10 = joblib.load('./dataset/recommend/yelp/user_page_data_10.pkl')
    #app.user_page_datas = user_page_data_10
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    run_server()