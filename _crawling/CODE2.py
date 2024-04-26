#=======================
# CODE2 program
#=======================
import urllib.request
import json
from CODE1 import getRequestUrl

def getNaverSearch(node, srcText, start, display):
    base = "https://openapi.naver.com/v1/search"
    node = "/%s.json" % node
    parameters = ("?query=%s&start=%s&display=%s"
                  % (urllib.parse.quote(srcText), start, display))
    url = base + node + parameters
    #print('url = %s' % url)
    
    responseDecode = getRequestUrl(url) #[CODE 1]
    #print('responseDecode = %s' % responseDecode)
    
    if (responseDecode == None):
        return None
    else:
        return json.loads(responseDecode)
