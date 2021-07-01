# API Methods
[toc]


## 'Dict_key' Error Detector 

Analyze the "dict_key" error for Python functions.

### Request
* Request URL: https://smart-scripts.org/py/smartModel
* Request Method: POST
* Request Payload: Source code to analyze.

Sample request:

```

curl  https://smart-scripts.org/py/smartModel \
-X POST \
-H "Referer: https://smart-scripts.org/py/model" \
-H "Content-Type: application/json" \
-d @request.json

```

request.json:

```
{"code":"\ndef _CheckGoogleSupportAnswerUrl(input_api, output_api):\n  pattern = input_api.re.compile('support\\.google\\.com\\/chrome.*/answer')\n  errors = []\n  for f in input_api.AffectedFiles():\n    for line_num, line in f.ChangedContents():\n      if pattern.search(line):\n        errors.append('    %s:%d %s' % (f.LocalPath(), line_num, line))\n\n  results = []\n  if errors:\n    results.append(output_api.PresubmitPromptWarning(\n      'Found Google support URL addressed by answer number. Please replace with '\n      'a p= identifier instead. See crbug.com/679462\\n', errors))\n  return results\n\n\ndef _CheckHardcodedGoogleHostsInLowerLayers(input_api, output_api):\n  def FilterFile(affected_file):\n    \"\"\"Filter function for use with input_api.AffectedSourceFiles,\n    below.  This filters out everything except non-test files from\n    top-level directories that generally speaking should not hard-code\n    service URLs (e.g. src/android_webview/, src/content/ and others).\n    \"\"\"\n    return input_api.FilterSourceFile(\n      affected_file,\n      white_list=(r'^(android_webview|base|content|net)[\\\\\\/].*', ),\n      black_list=(_EXCLUDED_PATHS +\n                  _TEST_CODE_EXCLUDED_PATHS +\n                  input_api.DEFAULT_BLACK_LIST))\n\n  base_pattern = ('\"[^\"]*(google|googleapis|googlezip|googledrive|appspot)'\n                  '\\.(com|net)[^\"]*\"')\n  comment_pattern = input_api.re.compile('//.*%s' % base_pattern)\n  pattern = input_api.re.compile(base_pattern)\n  problems = []  # items are (filename, line_number, line)\n  for f in input_api.AffectedSourceFiles(FilterFile):\n    for line_num, line in f.ChangedContents():\n      if not comment_pattern.search(line) and pattern.search(line):\n        problems.append((f.LocalPath(), line_num, line))\n\n  if problems:\n    return [output_api.PresubmitPromptOrNotify(\n        'Most layers below src/chrome/ should not hardcode service URLs.\\n'\n        'Are you sure this is correct?',\n        ['  %s:%d:  %s' % (\n            problem[0], problem[1], problem[2]) for problem in problems])]\n  else:\n    return []\n\n\ndef _CheckNoAbbreviationInPngFileName(input_api, output_api):\n  \"\"\"Makes sure there are no abbreviations in the name of PNG files.\n  The native_client_sdk directory is excluded because it has auto-generated PNG\n  files for documentation.\n  \"\"\"\n  errors = []\n  white_list = (r'.*_[a-z]_.*\\.png$|.*_[a-z]\\.png$',)\n  black_list = (r'^native_client_sdk[\\\\\\/]',)\n  file_filter = lambda f: input_api.FilterSourceFile(\n      f, white_list=white_list, black_list=black_list)\n  for f in input_api.AffectedFiles(include_deletes=False,\n                                   file_filter=file_filter):\n    errors.append('    %s' % f.LocalPath())\n\n  results = []\n  if errors:\n    results.append(output_api.PresubmitError(\n        'The name of PNG files should not have abbreviations. \\n'\n        'Use _hover.png, _center.png, instead of _h.png, _c.png.\\n'\n        'Contact oshima@chromium.org if you have questions.', errors))\n  return results\n\n\ndef _ExtractAddRulesFromParsedDeps(parsed_deps):\n  \"\"\"Extract the rules that add dependencies from a parsed DEPS file.\n\n  Args:\n    parsed_deps: the locals dictionary from evaluating the DEPS file.\"\"\"\n  add_rules = set()\n  add_rules.update([\n      rule[1:] for rule in parsed_deps.get('include_rules', [])\n      if rule.startswith('+') or rule.startswith('!')\n  ])\n  for specific_file, rules in parsed_deps.get('specific_include_rules',\n                                              {}).iteritems():\n    add_rules.update([\n        rule[1:] for rule in rules\n        if rule.startswith('+') or rule.startswith('!')\n    ])\n  return add_rules\n"}

```

### Response

In JSON format, results can be found at **issue** field in JSON response.

```
{"issues": "Total Number of Function: 5\nThe function:_CheckGoogleSupportAnswerUrl(input_api, output_api) in uploaded source code has a \"dict_key\" error.\nThe function:_CheckHardcodedGoogleHostsInLowerLayers(input_api, output_api) in uploaded source code has a \"dict_key\" error.\nThe function:FilterFile(affected_file) in uploaded source code has a \"dict_key\" error.\nTotal bug count: 3\n"}j

```


## Type Generator


### API Interface
* Request Method: POST
* Request URL: https://smart-scripts.org/py/TypeModel


Request Payload:
**code**: Source code to analyze.
**model**: integer value. 
* 0: 15 Base Types
* 1: 500 Types Model
* 2: PYI Model

**Response**:
A JSON String.
Fields:
* **result**: Source code appended with annotations as comments.
* **issue**: Model issues placed here. In Context Model, the inconsistency check results are included.


### CURL Sample
**Sample Requests**:

Normal Request:
```
curl  https://smart-scripts.org/py/TypeModel \
-X POST \
-H "Referer: https://smart-scripts.org/py/type" \
-H "Content-Type: application/json" \
-d '{"code":"category = \"AST\"\nsize = 3","model":0}'
```

Normal Response:

```

{"result": "category = \"AST\" #  category:str - 0.4278 \nsize = 3 #  size:int - 0.4433 ", "issue": "2 variables analyzed in 0.036 s, one variable at 17.762 ms"}
```

Request Contains Error:
```

curl  https://smart-scripts.org/py/TypeModel \
-X POST \
-H "Referer: https://smart-scripts.org/py/type" \
-H "Content-Type: application/json" \
-d '{"code":"category = \"AST\"\\nsize = 3","model":0}'
```

Error Response:
```
{"result": "category = \"AST\"\\nsize = 3", "issue": "AST ERROR: unexpected character after line continuation character (<unknown>, line 1)"}
```

Request not in Json Format:
```
curl  https://smart-scripts.org/py/TypeModel \
-X POST \
-H "Referer: https://smart-scripts.org/py/type" \
-H "Content-Type: application/json" \
-d 'code":"category = \"AST\"\\nsize = 3","model":0'
```

Response is a HTML file:
```
JSONDecodeError at /py/TypeModel
Expecting value: line 1 column 1 (char 0)
Request Method: POST
Request URL:    https://smart-scripts.org/py/TypeModel
Django Version: 2.2.3
Exception Type: JSONDecodeError
Exception Value:    
Expecting value: line 1 column 1 (char 0)
Exception Location: /usr/lib/python3.6/json/decoder.py in raw_decode, line 357
Python Executable:  /usr/local/bin/uwsgi
Python Version: 3.6.8
...
```

For larger Python Scripts, we can use:
```
>>> import json
>>> import os
>>> with open("test.py",'r') as f:
...   code = "\n".join(f.readlines())
...
>>> result = json.dumps({'code': code, 'model':0})
>>> with open('result.json','w') as f:
...   f.write(result)
...
>>>
```
Then, we can send request using:

Request:
```
curl  https://smart-scripts.org/py/TypeModel \
-X POST \
-H "Referer: https://smart-scripts.org/py/type" \
-H "Content-Type: application/json" \
-d @result.json
```


### Java Sample

We can send a POST request using Java API.

```
import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.URL;
import javax.net.ssl.HttpsURLConnection;
public class Main {
    public static void main(String[] args) throws Exception {
        Main http = new Main();
        http.sendPost();
    }
    // HTTP POST REQUEST
    private void sendPost() throws Exception {
        String url = "https://smart-scripts.org/py/TypeModel";
        URL obj = new URL(url);
        HttpsURLConnection con = (HttpsURLConnection) obj.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json");
        con.setRequestProperty("Referer", "https://smart-scripts.org/py/type");
        String urlParameters = "{\"code\":\"category = \\\"AST\\\"\\nsize = 3\",\"model\":0}";
        //Send POST Request
        con.setDoOutput(true);
        DataOutputStream wr = new DataOutputStream(con.getOutputStream());
        wr.writeBytes(urlParameters);
        wr.flush();
        wr.close();
        int responseCode = con.getResponseCode();
        System.out.println("\nSending 'POST' request to URL : " + url);
        System.out.println("Post parameters : " + urlParameters);
        System.out.println("Response Code : " + responseCode);
        BufferedReader in = new BufferedReader(
                new InputStreamReader(con.getInputStream()));
        String inputLine;
        StringBuffer response = new StringBuffer();
        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();
        System.out.println(response.toString());

    }

}

```

Sample Response:

```
Sending 'POST' request to URL : https://smart-scripts.org/py/TypeModel
Post parameters : {"code":"category = \"AST\"\nsize = 3","model":0}
Response Code : 200
{"result": "category = \"AST\" #  category:str - 0.4278 \nsize = 3 #  size:int - 0.4433 ", "issue": "2 variables analyzed in 0.036 s, one variable at 18.159 ms"}

Process finished with exit code 0
```


### Python Sample

Before we start, we have to install the **request** package for Python using the following command:

>pip install requests

Python Sample code:

```
# importing the requests library
import requests
import JSON

def main():
    # defining the api-endpoint
    API_ENDPOINT = "https://smart-scripts.org/py/TypeModel"

    # your source code here
    source_code = ''' 
category = "AST"
size = 3
    '''

    # data to be sent to api
    data = {'code': source_code,
            'model': 0}
    headers = {"Referer": "https://smart-scripts.org/py/type",
               "Content-Type": "application/json"}
    # sending post request and saving response as response object
    r = requests.post(url=API_ENDPOINT,headers = headers , data=json.dumps(data))

    # extracting response text
    resp = r.text
    print(resp)

if __name__ == "__main__":
    main()

```


Sample Response:

```
{"result": " \ncategory = \"AST\" #  category:str - 0.4278 \nsize = 3 #  size:int - 0.4433 \n    ", "issue": "2 variables analyzed in 0.036 s, one variable at 17.88 ms"}

```



## Type Checker

Infer type and check type inconsistencies.

### API Interface
* Request Method: POST
* Request URL: https://smart-scripts.org/py/TypeModel


* Request Payload:
**code**: Source code to analyze.
**model**: integer value. 
* 3: Context Model with type inconsistency detection. Time consumption high depends on the length of the code.
* 4: 15 Base Types with type inconsistency detection.

**Response**:
A JSON String.
Fields:
* **result**: Source code appended with annotations as comments.
* **issue**: The inconsistency check results and time consumption information are included in this field.





## SmartKube

* Request URL: https://smart-scripts.org/py/kubeModel
* Request Method: POST

* Request Payload:
The **code** field contains the Kubernetes code to check.
```

{"code":"apiVersion: v1\nkind: Deployment\nmetadata:\n  name: admin-user\n  namespace: kubernetes-dashboard\n\n---\napiVersion: rbac.authorization.k8s.io/v1\nkind: ClusterRoleBinding\nmetadata:\n  name: admin-user\nroleRef:\n  apiGroup: rbac.authorization.k8s.io\n  kind: ClusterRole\n  name: cluster-admin\nsubjects:\n- kind: ServiceAccount\n  name: admin-user\n  namespace: kubernetes-dashboard\n\n---\napiVersion: extensions/v1beta1\nkind: Deployment\nmetadata:\n  name: redis\nspec:\n  replicas: \"2\"\n  selector:\n    name: redis\n  template:\n    metadata:\n      labels:\n        name: redis\n    spec:\n      containers:\n      - name: redis\n        image: kubernetes/redis:v1\n        ports:\n        - containerPort: 6379\n        resources:\n          limits:\n            cpu: \"0.1\"\n        volumeMounts:\n        - mountPath: /redis-master-data\n          name: data\n      volumes:\n        - name: data\n          emptyDir: {}"}

```
* Sample Response:
Result is in json format. The **issue** field contains the result of checking kubernetes in HTML format.
```
{"issues": "<font color=\"yellow\">Warnings</font> for Document Starting from line 21 to 48:\nIncorrect Type <font color=\"yellow\">Warning</font>: expect one of { 'int',  'NoneType'}, but got  'str' for \"apiVersion(extensions/v1beta1)/kind(Deployment)/spec/replicas\"\nMissing Entry <font color=\"yellow\">Warning</font>: expect \"apiVersion(extensions/v1beta1)/kind(Deployment)/spec/selector/matchLabels\" when \"apiVersion(extensions/v1beta1)/kind(Deployment)/spec/selector\" presents\n<font color=\"#00bfff\">Finish!</font>\n"}

```