from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
# Library import
# 深層学習でよく使うテンソル型の配列を使うときに用いる
import torch
# 画像処理を行うときに使う
from PIL import Image
# 画像をリサイズしたりするのに使う
from torchvision import transforms
# Vision transformer の事前学習モデルを呼び出すために使う
from pytorch_pretrained_vit import ViT
# json形式のものを使うときにいる
import json
import os

app = Flask(__name__)
app.config['MODEL'] = ViT('B_16_imagenet1k', pretrained=True)
app.config['UPLOAD_PATH'] = './static'

@app.route('/')    # http://xxx 以降のURLパスを '/' と指定
def index():
    return render_template('index.html')   #defalutではtemplatesの直下のindex.htmlを見に行くことになっている

@app.route('/upload', methods = ['post'])
def upload():
    # アップロードしたファイルをimg_fileとしてapp.pyに読み込む
    img_file = request.files['img_file']
    # filenameの安全性を高くする
    filename = secure_filename(img_file.filename)
    ul_path = f"{app.config['UPLOAD_PATH']}/{filename}"
    # pythonでファイルやディレクトリが存在の有無を確認している
    if os.path.exists(app.config['UPLOAD_PATH']) != True:
        # 無かったら作成する
        os.mkdir(app.config['UPLOAD_PATH'])
    # ファイルを場所を指定してサーバー側に保存する
    img_file.save(ul_path)
    # img_urlをhtml側に返している
    return render_template("index.html", img_url=ul_path)

@app.route('/recognition', methods=['post'])
def recognition():
    img_path = request.form['img_path']
    # パスから画像へ
    img = Image.open(img_path)
    # データをロードした後に行う下処理の関数を構成
    tfms = transforms.Compose([
        # 画像をリサイズ
        transforms.Resize(app.config['MODEL'].image_size),
        # pytorchのテンソル型に変換
        transforms.ToTensor(),
        # 標準化(平均、標準偏差)
        transforms.Normalize(0.5, 0.5),
    ])
    # テンソル型
    img = tfms(img)
    # モデルで予測できる形に変える
    img = img.unsqueeze(0)
    # 勾配降下法
    with torch.no_grad():
        outputs = app.config['MODEL'](img)
    # 予測結果（番号）をpredに代入する
    pred = torch.argmax(outputs)
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[key] for key in labels_map]
    pred_label = labels_map[pred]
    return render_template('index.html', img_url=img_path, pred_label=pred_label)

if __name__ == "__main__":
    app.run(debug = True)