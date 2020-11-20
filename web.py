from flask import Flask

app=Flask(__name__)


#This url will generate audio with the given text
@app.route("/audio/<text>")
def generate_audio(text):
    return "Hello World"

#This url will give training status
@app.route("/train/status")
def generate_audio():
    return "Hello World"

if __name__=="__main__":
    app.run()
