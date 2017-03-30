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
define("port", default=8009, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class SearchHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("search.html")

class UserPageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("user.html")

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(
	    handlers=[(r'/', IndexHandler), (r'/user', UserPageHandler), (r'/search', SearchHandler)],
	    template_path=os.path.join(os.path.dirname(__file__), "templates"),
	    static_path=os.path.join(os.path.dirname(__file__), "static"),
	    debug=True
	)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()