from flask import Flask, render_template, request, url_for
from model import clip_class
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/")
def index():
    global g_gt, g_cut_rfile, g_rfile
    gt, rfile = clip_class.get_ramdom_path()
    g_rfile = rfile
    rfile_cut = rfile[9:]
    g_gt = gt
    g_cut_rfile = rfile_cut
    return render_template("index.html",  image_file=g_cut_rfile)


@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./static/image/uploads/' + secure_filename(f.filename))
        upload = 'image/uploads/'+secure_filename(f.filename)
        submit_path = "./static/" + upload

        given_answer, _ = clip_class.clip_predict(submit_path)

        if g_gt == given_answer:
            print(upload)
            return render_template('correct.html', value = g_gt, image_file=g_cut_rfile, predict = upload, predict_class = given_answer)
        else:
            print(upload)
            return render_template('wrong.html', value = g_gt, image_file=g_cut_rfile, predict = upload, predict_class = given_answer)


if __name__ == '__main__':
    app.run(debug=True)
