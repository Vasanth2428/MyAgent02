content = open('static/app.js', 'r', encoding='utf-8').read()

# Check what's at line 963 area
idx = content.find('Catch session list to update message count and timestamp')
if idx > 0:
    snippet = content[idx-200:idx+300]
    print(repr(snippet))
