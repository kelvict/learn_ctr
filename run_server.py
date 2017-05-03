#!/user/bin/env python
#coding=utf-8
# Author: Zhiheng Zhang (405630376@qq.com)
#

from webserver import server
from tornado.options import define, options, parse_command_line
define("port", default=8009, help="run on the given port", type=int)
parse_command_line()
server.run_server(options.port)