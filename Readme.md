# Captcha Breaker
Build with Tensorflow (ConvNets) and  Node.js :muscle::muscle::muscle:

# Installation
#### Python packages
```
$ pip install -r requirements.txt
```

#### Node.js packages (Node.js user only)
```
$ npm i
```

# Usage
## 1. Create train data
#### Prepare your training dataset
* Copy captcha images to `data/captcha` folder
```
|_data
      |_captcha
          |_ captcha_1.jpg
          |_ captcha_2.jpg
```
* Create mapping file `data/captcha.json` to map your train image with corresponding label
```json
{
    "captcha_1.jpg": "HEYMEN",
    "captcha_2.jpg": "XINCHA"
}
```

#### Build train data for model
Run `src/create_train_data.py` will save your train data as `data/captcha.npz` compressed file.
```
$ python src/create_train_data.py
```

The compressed `data/captcha.npz` includes:
* Train Data ( `x_train`, `y_train` ): `80%`
* Test Data ( `x_test`, `y_test` ): `20%`

## 2. Train
Run `src/train.py` to train the model with your own dataset.
```
$ python src/train.py
```

Take :coffee: or :tea: while waiting!

## 3. Attack
Now, enjoy your war :fire::fire::fire: :stuck_out_tongue_winking_eye::stuck_out_tongue_winking_eye::stuck_out_tongue_winking_eye:

#### Python
```
$ python src/predict --fname YOUR_IMAGE_PATH_or_URL
```

Sample output:
```
loading image: data/captcha/captcha_2.jpg
load captcha classifier
predict for 1 char: `X` with probability: 99.956%
predict for 2 char: `I` with probability: 99.909%
predict for 3 char: `N` with probability: 99.556%
predict for 4 char: `C` with probability: 99.853%
predict for 5 char: `H` with probability: 99.949%
predict for 6 char: `A` with probability: 98.889%
Captcha: `XINCHA` with confident: `99.686%`
XINCHA
```

#### Node.js
```js
const captchaPredict = require('src/predict')

captchaPredict(YOUR_IMAGE_PATH_or_URL)
  .then(console.log)
  .catch(console.error)
```
Sample output:
```
[
  "loading image: data/captcha/captcha_2.jpg",
  "load captcha classifier",
  "predict for 1 char: `X` with probability: 99.956%",
  "predict for 2 char: `I` with probability: 99.909%",
  "predict for 3 char: `N` with probability: 99.556%",
  "predict for 4 char: `C` with probability: 99.853%",
  "predict for 5 char: `H` with probability: 99.949%",
  "predict for 6 char: `A` with probability: 98.889%",
  "Captcha: `XINCHA` with confident: `99.686%`",
  "XINCHA"
]
```
