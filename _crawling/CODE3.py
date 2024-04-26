#=======================
# CODE3 program
#=======================
#=======================
# CODE3 program
#=======================
from datetime import datetime

def getPostData(post, jsonResult, cnt):
    title = post['title']
    description = post['description']
    org_link = post['originallink']
    link = post['link']
    
    pDate = datetime.strptime(post['pubDate'], '%a, %d %b %Y %H:%M:%S +0900')
    pDate = pDate.strftime('%Y-%m-%d %H:%M:%S')
    
    jsonResult.append({
        'cnt': cnt,
        'title': title,
        'description': description,
        'org_link': org_link,
        'link': org_link,
        'pDate': pDate
    })
    return
