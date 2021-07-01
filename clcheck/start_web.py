from web.server.app import app
import os
import ssl
os.environ['ENV'] = 'development'
DEBUG=True

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('/etc/letsencrypt/live/smart-scripts.org/fullchain.pem', '/etc/letsencrypt/live/smart-scripts.org/privkey.pem')

app.run(debug=DEBUG, host="0.0.0.0", port=5000, ssl_context=context)
