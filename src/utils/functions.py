
import re
import urllib.parse as urllp
def encodeURIComponent(s): return urllp.quote(s, safe='/', encoding='utf-8', errors=None)


def parse_log_line(entry):
    """
    Parse a log entry and extract its components.
    """
    log_pattern1 = re.compile(r'^(?P<timestamp>[\d-]+\s[\d:,]+)\s-\s(?P<logger>\w+)\s-\s(?P<level>\w+)\s-\s(?P<message>.+)$')
    log_pattern2 = re.compile(r'^(?P<timestamp>[\d-]+\s[\d:,]+)\s-\s(?P<level>\w+)\s-\s(?P<message>.+?)\s\[(?P<source_file>[\w.]+):(?P<line_number>\d+)\]$')
    log_pattern3 = re.compile(r'^(?P<timestamp>[\d-]+\s[\d:,]+)\s-\s(?P<level>\w+)\s-\s(?P<message>.+)$')

    match = log_pattern1.match(entry[0])
    if match:
        result = match.groupdict()
        if len(entry) > 1:
            result['traceback'] = "\n".join(entry[1:])
        return result

    match = log_pattern2.match(entry[0])
    if match:
        result = match.groupdict()
        if len(entry) > 1:
            result['traceback'] = "\n".join(entry[1:])
        return result
    
    match = log_pattern3.match(entry[0])
    if match:
        result = match.groupdict()
        if len(entry) > 1:
            result['traceback'] = "\n".join(entry[1:])
        return result
    
    return None

def parse_json_response(json_data):
    result_list = []
    for entry in json_data["results"]["bindings"]:
        result_dict={}
        for key in json_data["head"]["vars"]:
            if key in entry:
                result_dict[key]=entry[key]["value"]
            else:
                result_dict[key]=""
        if len(result_dict)>0:
            result_list.append(result_dict) 

    return result_list if len(result_list)>0 else None


def clean_json_string(json_string):
    # Remove invalid control characters
    json_string=json_string.strip('```').strip()
    if json_string.startswith("json") or json_string.startswith("html"):
        json_string = json_string[5:]
    json_string=json_string.replace("'","")
    json_string = re.sub(r'[\x00-\x1F\x7F]', '', json_string)
    return json_string


def parse_nlp_response(nlp_resp):
    try:
        if nlp_resp['data'] and len(eval(nlp_resp['data'])) > 0:
            return eval(nlp_resp['data'])
        else:
            return []
    except Exception as e:
        print("Error occured while parsing NLP response :" ,e) 



def log_separator(module_name: str) -> str:
    """Generate a formatted separator string for logs."""
    separator_length = 80
    title = f" {module_name.upper()} ".center(separator_length, "=")
    return f"\n{title}\n"