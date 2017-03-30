#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import os

from tornado.options import define, options
define("port", default=8008, help="run on the given port", type=int)

from preprocesser import yelp_preprocess
from sklearn.externals import joblib
import json

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class SearchHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("search.html")

class UserPageHandler(tornado.web.RequestHandler):
    def get(self):
        user_page_data = yelp_preprocess.get_user_page_data(0,
                                           app.limit_uid_to_reviews_map,
                                           app.uid_to_user_map,
                                           app.bid_to_business_map)
        user = user_page_data['user']
        rec_records = user_page_data['rec_records']
        visit_records = user_page_data['visit_records']
        self.render("user.html", user=user, recs=rec_records, histories=visit_records)


def run_server():
    tornado.options.parse_command_line()
    app = tornado.web.Application(
	    handlers=[(r'/', IndexHandler), (r'/user', UserPageHandler), (r'/search', SearchHandler)],
	    template_path=os.path.join(os.path.dirname(__file__), "templates"),
	    static_path=os.path.join(os.path.dirname(__file__), "static"),
	    debug=True
	)

    f = open("./dataset/recommend/yelp/server_data.json")
    server_data = json.load(f)
    app.limit_uid_to_reviews_map, app.uid_to_user_map, app.bid_to_business_map = server_data
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
if __name__ == "__main__":
    run_server()