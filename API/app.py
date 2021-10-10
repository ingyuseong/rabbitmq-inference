from flask import Flask, jsonify, request
from starGAN import starGAN_inference

app = Flask(__name__)

def check_user(user_id):
    import boto3
    import config

    s3_client = boto3.client(
        's3',
        aws_access_key_id = config.S3_CONFIG['aws_access_key_id'],
        aws_secret_access_key = config.S3_CONFIG['aws_secret_access_key'],
        region_name = config.S3_CONFIG['region_name']
    )
    obj = s3_client.list_objects(
        Bucket=config.S3_CONFIG['bucket'],
        Prefix='image/user/{}/result'.format(user_id),
    )
    files = [file['Key'] for file in obj['Contents']]

    if len(files) == 0:
        return False
    return True

@app.route('/ai/stargan', methods=['POST'])
def stargan():
    if request.method == 'POST':
        file = request.files['file']
        user_id = 112345
        length = request.form['length']
        gender = request.form['gender']
        img_bytes = file.read()

        # try:
        img_list = starGAN_inference(user_id, gender, length, img_bytes)
        print(img_list)
        return jsonify({'status': "success", 'data': img_list})
        # except:
        #     return jsonify({'status': "error"})
        

if __name__ == '__main__':
    app.run(debug=True)