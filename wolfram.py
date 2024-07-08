import json
import requests
from hashlib import md5
from urllib.parse import urlsplit, urlencode, unquote_plus

headers = {"User-Agent": "Wolfram Android App"}
APPID = "3H4296-5YPAGQUJK7" # Mobile app AppId
SERVER = "api.wolframalpha.com"
SIG_SALT = "vFdeaRwBTVqdc5CL" # Mobile app salt

s = requests.Session()
s.headers.update(headers)

def calc_sig(query):
	"""
	Calculates WA sig value(md5(salt + concatenated_query)) with pre-known salt
	
	@query
	In format of "input=...&arg1=...&arg2=..."
	"""

	params = list(filter(lambda x: len(x) > 1, list(map(lambda x: x.split("="), query.split("&"))))) # split string by & and = and remove empty strings
	params.sort(key = lambda x: x[0]) # sort by the key

	s = SIG_SALT
	# Concatenate query together
	for key, val in params:
		s += key + val
	s = s.encode("utf-8")
	return md5(s).hexdigest().upper()

def craft_signed_url(url):
	"""
	Craft valid signed URL if parameters known
	
	@query
	In format of "https://server/path?input=...&arg1=...&arg2=..."
	"""

	(scheme, netloc, path, query, _) = urlsplit(url)
	_query = {"appid": APPID}

	_query.update(dict(list(filter(lambda x: len(x) > 1, list(map(lambda x: list(map(lambda y: unquote_plus(y), x.split("="))), query.split("&")))))))
	query = urlencode(_query)
	_query.update({"sig": calc_sig(query)}) # Calculate signature of all query before we set "sig" up.
	return f"{scheme}://{netloc}{path}?{urlencode(_query)}"

def basic_test(query_part):
	"""
	Simple PoC

	@query_part
	Example is "input=%url_encoded_string%&arg1=...&arg2=..."
	https://products.wolframalpha.com/api/documentation/#formatting-input
	"""
	r = s.get(craft_signed_url(f"https://{SERVER}/v2/query.jsp?{query_part}"))
	if r.status_code == 200:
		return r.text
	else:
		raise Exception(f"Error({r.status_code}) happened!\n{r.text}")

from urllib.parse import quote
def query_wolfram(query):
	json_response = basic_test(f"input={quote(query)}&format=plaintext&output=json&includepodid=Result")
	response = json.loads(json_response)
	text = "\n".join([ "\n".join([ j["plaintext"] for j in i["subpods"]]) for i in response["queryresult"]["pods"]])
	return text

if __name__ == "__main__":
	pass
	# print(basic_test("input=y%27+%3D+y%2F%28x%2By%5E3%29&podstate=Solution__Step-by-step+solution&format=plaintext&output=json"))
	# print(basic_test(f"input={quote('what is the weight of hydrogen?')}&format=plaintext&output=json&includepodid=Result"))