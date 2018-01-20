import sys
import time
import random
import urllib
import socket

def single_count(url, word, errors):
    
    import urllib
    import socket
    wordcount = 0

    req = urllib.request.Request(url)
    try:
        site = urllib.request.urlopen(req, timeout = 2)
    except urllib.error.HTTPError as e:
        errors[url] = e.code
    except urllib.request.URLError as e:
        errors[url] = e.reason
    except UnicodeEncodeError:
        errors[url] = 'UnicodeEncode'
        pass
    except socket.timeout:
        errors[url] = 'Timeout'
    except:
        errors[url] = 'Unexpected'
    try:
        site = site.readall().decode('utf8')
        site = site.lower()
        wordcount = site.count(word)
    except:
        errors[url] = 'Other'

    return(wordcount, errors)

def get_links(url, errors):
    
    import urllib
    import socket
    
    linklist = []
    req = urllib.request.Request(url)
    try:
        site = urllib.request.urlopen(req, timeout = 2)
    except urllib.error.HTTPError as e:
        errors[url] = e.code
    except urllib.request.URLError as e:
        errors[url] = e.reason
    except UnicodeEncodeError:
        pass
    except socket.timeout:
        errors[url] = 'Timeout'
    except:
        errors[url] = 'Unexpected'

    try:
        site = site.readall().decode('utf8')
        site = site.lower()
    except UnicodeDecodeError:
        errors[url] = 'UnicodeDecode'
    except:
        errors[url] = 'Other'
    
    
    loc, start, end = 0, 0, 0
    while start != -1:
        try:
            loc = site.find('href=', end)
            start = site.find('"',loc)
            end = site.find('"',start + 1)
            link = site[start+1:end]
            if link.count('http:') > 0:
                linklist.append(link)
            loc = end
        except:
            errors[url] = 'Other'
        
    return(linklist, errors)

def webcount(word, layers = 1, startsite = 'http://hs.fi'):
    
    t_start = time.time()
    t_limit = 600

    word = word.lower()
    links = [[] for n in range(layers+1)]
    counts = [[] for n in range(layers+1)]
    links[0] = [startsite]
    errors = {}
    if layers == 0:
        linklist = [startsite]
        counts, errors = singlecount(linklist, errors, word)
        return(linklist, counts, errors)

    for i in range(layers):
        if time.time() > (t_start + t_limit):
            print('Time limit exceeded')
            break
            
        counter = 0
        for item in links[i]:
            linklist, errors, wordcount = urlcount(item, errors, word, t_start, t_limit)
            if linklist == 'Fail':
                continue
            links[i+1] = linklist
            print(i,counter)
            counts[i].append(wordcount)
            counter += 1
        print(len(links[i+1]))
    
    lastcounts, lasterrors = singlecount(links[-1], errors, word, t_start, t_limit)
    counts[i+1] = lastcounts
    errors.update(lasterrors)
    return(links, counts, errors)
    
def urlcount(url, errors, word, t_start, t_limit):
    
    item = url
    linklist = []
    req = urllib.request.Request(item)
    try:
        site = urllib.request.urlopen(req, timeout = 2)
    except urllib.error.HTTPError as e:
        errors[item] = e.code
    except urllib.request.URLError as e:
        errors[item] = e.reason
    except UnicodeEncodeError:
        pass
    except socket.timeout:
        errors[item] = 'Timeout'
    except:
        errors[item] = 'Unexpected'
    else:
        linklist, wordcount = 'Fail', 0
        return(linklist, errors, wordcount)

    try:
        site = site.readall().decode('utf8')
    except UnicodeDecodeError:
        pass
    site = site.lower()
    wordcount = site.count(word)
    
    loc, start, end = 0, 0, 0
    while start != -1:
        loc = site.find('href=', end)
        start = site.find('"',loc)
        end = site.find('"',start + 1)
        link = site[start+1:end]
        if link.count('http') > 0:
            linklist.append(link)
        loc = end
        if time.time() > (t_start + t_limit):
            print('Time limit exceeded')
            break
    linklist, wordcount = 'Fail', 0
    return(linklist, errors, wordcount)

	
def singlecount(linklist, errors, word, t_start, t_limit):

    wordcount = []
    for item in linklist:
        if time.time() > (t_start + t_limit):
            print('Time limit exceeded')
            break
        req = urllib.request.Request(item)
        try:
            site = urllib.request.urlopen(req, timeout = 2)
        except urllib.error.HTTPError as e:
            errors[item] = e.code
            continue
        except urllib.request.URLError as e:
            errors[item] = e.reason
            continue
        except UnicodeEncodeError:
            continue
        except socket.timeout:
            errors[item] = 'Timeout'
            continue
        except:
            errors[item] = 'Unexpected'
        try:
            site = site.readall().decode('utf8')
        except UnicodeDecodeError:
            continue
        site = site.lower()
        
        wordcount.append(site.count(word))
    return(wordcount, errors)
	
	
if __name__ == '__main__':

	t_start = time.time()
	word = 'isis'
	iterations = 10
	errors = {}
	startsite = 'http://ampparit.com'
	
	startcount, errors = single_count(startsite, word, errors)
	results = {startsite: startcount}
	maxkey = max(results)
	maxvalue = max(results.values())

	for i in range(iterations):
		
	#     if maxvalue == 0:
	#         maxkey = random.choice(list(results.keys()))
			
	#     for key in list(results):
	#         wordcount, errors = single_count(key, word, errors)
	#         results.update({key: wordcount})
		
		linklist, errors = get_links(maxkey, errors)
		if linklist == []:
			maxkey = random.choice(list(results.keys()))
			continue
		print('Now searching ' + str(len(linklist)) + ' links at ' + maxkey + ' (wordcount ' + str(maxvalue) + ').')
		
		counter = 0
		for link in linklist:
	#         if link not in errors.keys() and link not in results.keys():
			wordcount, errors = single_count(link, word, errors)
			if wordcount >= maxvalue:
				results.update({link: wordcount})
			counter += 1
			if  len(linklist) > 10 and counter % round(len(linklist)/10) == 0:
				print(str(counter) + ' out of ' + str(len(linklist)) + ' done.')
		if maxkey == max(results):
			maxkey = random.choice(list(results.keys()))
		else:
			maxkey = max(results)
		maxvalue = max(results.values())
			
		print('Iteration ' + str(i+1) + ': max count ' + str(maxvalue) + ' at ' + maxkey)
		
	print('Elapsed ' + str(round(time.time()-t_start)) + ' s.')    
	
	# # Main, count from all links (slow)
	# 
	# word = 'monkey'
	# layers = 2
	# threshold = 0
	# errors = {}
	# startsite = 'http://reddit.com'
	# startcount, errors = single_count(startsite, word, errors)
	# results = {startsite: startcount}
	# print(results)
	# if startcount < threshold:
	#     print('Count on startsite below threshold')
	#     sys.exit()
	# 
	# for i in range(layers):
	#     
	#     for key in list(results):
	#         wordcount, errors = single_count(key, word, errors)
	#         if wordcount > threshold:
	#             results.update({key: wordcount})
	#     
	#     threshold = sum(results.values()) / len(results)
	#     
	#     counter = 0
	#     for key in list(results):
	#         linklist, errors = get_links(key, errors)
	#         for link in linklist:
	#             if link not in errors.keys() and link not in results.keys():
	#                 wordcount, errors = single_count(link, word, errors)
	#                 if wordcount > threshold:
	#                     results.update({link: wordcount})
	#         
	#         counter += 1
	# #         if counter % 10 == 0:
	#         print(counter)
	# 
	#     print('Layer ' + str(i) + ': ' + str(len(results)))