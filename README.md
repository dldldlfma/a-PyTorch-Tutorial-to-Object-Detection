This is a **[PyTorch](https://pytorch.org) Tutorial to Object Detection**.

This is the third in [a series of tutorials](https://github.com/sgrvinod/Deep-Tutorials-for-PyTorch) I'm writing about _implementing_ cool models on your own with the amazing PyTorch library.

Pytorch와 CNN에 대한 기본적인 이해가 필요합니다. 

Pytorch를 해본적이 없다면 다음 두가지를 참고하세요.
[Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).

Questions, suggestions, or corrections can be posted as issues.

I'm using `PyTorch 0.4` in `Python 3.6`.

# Contents

[***Objective***](https://github.com/dldldlfma/pytorch_tutorial_ssd#objective)

[***Concepts***](https://github.com/dldldlfma/pytorch_tutorial_ssd#concepts)

[***Overview***](https://github.com/dldldlfma/pytorch_tutorial_ssd#overview)

[***Implementation***](https://github.com/dldldlfma/pytorch_tutorial_ssd#implementation)

[***Training***](https://github.com/dldldlfma/pytorch_tutorial_ssd#training)

[***Evaluation***](https://github.com/dldldlfma/pytorch_tutorial_ssd#evaluation)

[***Inference***](https://github.com/dldldlfma/pytorch_tutorial_ssd#inference)

[***Frequently Asked Questions***](https://github.com/dldldlfma/pytorch_tutorial_ssd#faqs)

# Objective


**이미지 속 물체가 무엇이고 어디에 있는지 검출하기 위한 모델을 빌드하기위해서**

<p align="center">
<img src="./img/baseball.gif">
</p>

우린 Object Detection분야에서 인기있고, 강력하며 특히 빠른 [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325)를 수행해 볼겁니다. 저자가 작성한 실행 가능한 원본은 다음 주소에서 찾을수 있습니다. [here](https://github.com/weiliu89/caffe/tree/ssd).


다음은 트레이닝에 사용하지 않았던 그림을 통한 몇가지 예시 입니다. 

---

<p align="center">
<img src="./img/000001.jpg">
</p>

---

<p align="center">
<img src="./img/000022.jpg">
</p>

---

<p align="center">
<img src="./img/000069.jpg">
</p>

---

<p align="center">
<img src="./img/000082.jpg">
</p>

---

<p align="center">
<img src="./img/000144.jpg">
</p>

---

<p align="center">
<img src="./img/000139.jpg">
</p>

---

<p align="center">
<img src="./img/000116.jpg">
</p>

---

<p align="center">
<img src="./img/000098.jpg">
</p>

---

이곳에 몇가지 예시가 더 있습니다. [end of the tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#some-more-examples).

---

# Concepts

* **Object Detection**. duh.

* **Single-Shot Detection**. 이 전의 객체 검출 구조는 두 개의 구분된 단계를 가지고 있었습니다. 물체의 위치를 제안하는 위치 제안 네트워크(region proposal network)와 제안된 위치에서 물체의 종류를 결정하는 구분자(classifier) 두가지 였습니다. 계산적으로, 이런 구조는 매우 많은 연산을 필요로 하며 실제 세상에서 실시간으로 사용하기 어렵습니다. Single-shot models 은 localization과 detection 작업을 단일 네트워크의 단일 forward 연산을 통해서 해결합니다. 결과적으로 상당히 빠르게 수행할 수 있으며 연산능력이 적은 Hardware에서도 수행가능해 집니다..


* **Multiscale Feature Maps**. 이미지 분류 작업에서, 우리는 CNN의 마지막 출력(final convolutional feature map)을 사용합니다. 이는 가장 작지만 원래 이미지의 가장 깊은 표현입니다. 객체 검출에서, 네트워크의 중간에 있는 Convolutional layer의 feature maps은 원래 이미지 대비 다양한 크기에서의 특징을 담고 있어서 객체 검출에서 쓸만합니다. 그러므로, 크기가 다른 다양한 feature maps에서 고정된 크기의 filter의 동작은 다양한 크기의 객체 검출을 가능하게 할것입니다.


* **Priors**. 이것들은 명확한 feature maps위, 명확한 위치에 명확한 크기 및 비율로 정의되어 있는 pre-computed boxes입니다. pre-computed box들은 데이터셋안에 있는 객체표시상자(object's bounding box)의 특성과 match하기 위해서 조심스럽게 선택됩니다.  (Priors : 사전 확률 or 사전 예측)


* **Multibox**. 이건 회기문제(regression problem)로 객체의 검출상자(object's bounding box) 예측값을 계산하는  [기술](https://arxiv.org/abs/1312.2249) 이다. 검출된 객체의 좌표는 정답에 해당하는 ground truth's의 좌표로 회귀된다. 게다가, 각 predicted box에 대해서, 다양한 객체 종류에 대한 점수가 생성된다. 사전확률은 정답(ground truth)를 모델로 하기 때문에 예측을 위한 시작점 역할을 한다. 그러므로, 많은 예측 상자가 존재할것이고 그것들은 대부분 객체를 담고 있지 않을 것 입니다.

* **Hard Negative Mining**. 본 예제에서 이 것(Hard Negative Mining)은 직접적으로 모델로 부터 나온 매우 어려운 false positive 예측을 보내고 이것을 학습하도록 강요합니다. 다른 말로 하면, 우리는 모델이 식별하기 매우 어려운 negative만들 모아서 학습한다는 말입니다. 객체 검출 과정에서, 대부분의 predicted box들은 객체가 포함되어 있지 않기에 이 것은 negative-positive 사이의 불균형을 바로잡는 역할을 합니다.


* **Non-Maximum Suppression**. 주어진 어떤 위치에서도, 다중 예측은 상당히 겹칠 것이다. 그러므로, 사전예측(priors)들은 사실 같은 객체의 사본이 될수 있다. 비최대억제 (Non-Maximum Suppression = NMS) 는 가장 높은 점수를 받은 한개를 제외하고 다른 것들을 억제함으로써 겹치는 예측을 제거하는 것을 말한다. 

# Overview


여기서는, 이 모델을 설명할 것이다. 만약 너가 이미 이것에 대해 잘알고 있다면, 여기를 생략하고 [Implementation](https://github.com/dldldlfma/pytorch_tutorial_ssd#implementation) section이나 the commented code로 넘어가도 좋다.

이 과정에서, 당신은 SSD가 매우 구체적인 구조와 공식을 가지고 있을수 있도록 하는 정당한 공학적 과정이 있음을 알 수 있을 것이다.어떤 부분이 처음에는 무의미하거나 인위적으로 보일수도 있지만,이 것은 이 분야에서 몇년동안(종종 경험적으로) 긴 연구를 통해 만들어진 것임을 기억하자.

### 몇가지 정의(Some definitions)

A box 는 상자다. A 경계상자(_bounding_ box) 는 객체 주변을 감싸고 있는 상자다. 즉, 그 경계를 나타낸다.

이 tutorial에서, 우리는 두가지 종류의 box를 마주치게 될텐데 하나는 그냥 '상자'고 다른 하나는 경계상자(bounding box)다. 그러나 모든 상자가 이미지 위에서 표현되고 우리는 위치, 모양, 크기 또 다른 속성등을 측정할 필요가 있다.

#### 경계 좌표(Boundary coordinates)

상자를 나타내기 위해 가장 명확한 방법은 그것의 경계에 있는 pixel의 'x', 'y'좌표 줄에 의해서 구성됩니다. 

![](./img/bc1.PNG)

경계좌표는 간단하게 오른쪽과 같이 표현됩니다.  **`(x_min, y_min, x_max, y_max)`**.

그런데 만약 우리가 이미지의 크기가 달라져서 실제 크기를 모른다면 픽셀 값은 쓸모 없어진다. 좋은 방법은 모든 좌표를 0~1사이의 숫자로 표현해 놓는 것이다.

![](./img/bc2.PNG)

이제 좌표는 크기 불변임과 동시에 모든 상자가 모든 이미지에서 같은 scale로 측정가능하다. 

#### 중심-크기 좌표(Center-Size coordinates)

이건 상자의 위치와 크기를 나타내는 좀더 명확한 방법입니다. 

![](./img/cs.PNG)

상자의 중심-크기 좌표는 오른쪽과 같습니다. **`(c_x, c_y, w, h)`**.

코드안에서, 당신은 우리가 반복적으로 이 작업에 적합한 두 가지 좌표시스템을 사용하는 것을 볼 수 있을 것이다. 그리고 항상 이 것은 0~1 사이의 값으로 표현되어 있을 것이다. (always in their fractional forms)

#### Intersection-over-Union(IoU) or Jaccard Index 

Jaccard Index 또는 Jaccard Overlap 또는 Intersection-over-Union (IoU)으로 불리는 이것은 **두 상자의 중첩되는 범위 혹은 정도를** 측정한다.

![](./img/jaccard.jpg)

IoU가 1인것은 같은 상자임을 의미한다. 반면 0일 경우 겹치는 공간이 하나도 없는 서로 관련 없은 상자임을 말한다.

이건 간단한 metric이지만 우리 application의 다양한 곳에서 발견할수 있는 지표입니다.

### Multibox

Multibox는 객체를 검출하는 기술로 이것의 예측은 두가지로 구성요소를 가지고 있습니다.

- **객체를 포함할수도 포함하지 않을수도 있는 상자의 좌표**. 이것은 회귀(_regression_) 문제 입니다.

- **이 상자의 다양한 객체 종류에 대한 점수**, 상자속에 객체가 없음을 의미하는 배경 class를 포함해서. 이건 분류(_classification_) 문제입니다.

### Single Shot Detector (SSD)

SSD는 3단계로 구성된 순수한 Convolution Neural Network(CNN)입니다.

- __Base(기본) convolutions__ : 낮은 수준의 feature map들을 제공하기 위한 기존에 존재하던 이미지 분류 구조로부터 가져온 Convolution (= ImageNet Chellenge의 Classification을 미리 학습해둔 것들을 말한다.)

- __Auxiliary(보조) convolutions__ : Base network의 상단에 붙이는 것으로 높은 수준의 feature maps을 제공한다.


- __Prediction(예측) convolutions__ : feature maps에서 객체의 위치를 찾고 식별합니다.

논문에서는 SSD300 과 SSD512라고 불리는 모델을 증명했습니다. 접미사(사진 위의 숫자)는 입력이미지의 크기를 말합니다. 비록 두 네트워크의 구성이 약간 다르지만 원리는 같습니다.그저 SSD512의 네트워크가 조금 더 크고 성능이 약간 좋을 뿐입니다.


편의를 위해서 우리는 SSD300을 다뤄볼겁니다.

### Base Convolutions – part 1

첫째로, 왜 기존에 존재하던 network architecture의 Convolution을 가져다가 사용하는 걸까?

기존의 학습된 network architecture가 image classification에서 이미지의 본질을 잘 잡아내는 것이 이미 증명되어 있기 때문입니다.

비록 object detection은 이미지 전체보다는 물체가 들어있는 특정영역에 더 관심이 있지만 
이미 학습된 CNN 모델의 Convolutional 특징은 object detection에 유용합니다.

또 다른 장점으로, classification dataset을 통해서 안정적으로 학습되어 있는 layer를 사용한다는 것이 있습니다. 당신은 이미 알겠지만 이걸 **Transfer Learning**이라고 합니다. 

비록 detection과 다른 classification으로 학습된 네트워크를 가져온것이지만 이건 여전히 깊은 관계가 있습니다. 이러한 과정을 통해서 시작도하기전에 큰 진전을 이뤄냈습니다.

이 논문의 작가는 **VGG-16 architecture**를 base network로 차용하였습니다. 이것은 original form보다는 간단하게 되어 있습니다.

![](./img/vgg16.PNG)

저자는 ImageNet Large Scale Visual Recognition Competition(ILSVRC)의 classification 부분에 대해서 미리 학습된 모델을 사용하는 것을 추천합니다. 운이 좋게도 이미 학습된 유명한 구조의 모델들을 Pytorch에서 사용할 수 있습니다. 당신이 원한다면 더 큰사이즈의 모델인 ResNet을 사용할수 있습니다. 그럴땐 많은 계산이 필요할수 도 있음을 염두해 두세요.

논문에 따라서 **우린 이미 학습된 네트워크를 약간 변형하여** 우리의 object detection에 채택할 것 입니다. 이미 학습된 네트워크를 적용함에 있어 어떤 부분은 논리적으로 필요하지만, 어떤 부분은 편의나 선호에 의한 것일 수 있습니다.


- **입력 이미지 크기** 는 (300,300) 으로 시작합니다. 

- 크기를 반으로 만드는 **3번째 pooling layer**는, 출력 크기를 결정하기 위해서 `ceiling`이라고 부르는 함수를 `floor` 라고 불리는 기본 함수대신 사용합니다. 이것은 이전 feature map의 차원이 짝수가 아닌 홀수인 경우에 중요합니다. 위에서 image를 봄으로써, 당신은 `300, 300`이라는 input image 사이즈를 계산할 수 있습니다. the `conv3_3`의 feature map은 cross-section의 `75, 75`이 될것이고, 이것의 절반은 `37, 37`대신 `38, 38`이 될겁니다.

- 우리는 **5th pooling layer**를 수정합니다. 기존에 `2, 2` kernel 사이즈에 stride `2` 로 진행 했던 것을 `3, 3` kernel size에 stride `1` 수정합니다. 수정으로 인해 더 이상 Convolution layer를 진행하면서 feature map의 크기가 절반으로 줄어 들지 않습니다.

- 우리는 분류를 하는 것이 아니기 때문에 더 이상 fully-connected layer가 필요하지 않습니다. 그래서 우리는 `fc8` layer는 완전히 버립니다. 그런데 선택적으로 **`fc6` layer와  `fc7` layer는 convolution layer로 변환합니다. 그리고 그건 앞으로 `conv6` and `conv7` 라고 부를겁니다.**.

위 3가지 수정은 간단하지만 마지막에 위치한 FC-> Convolutional layer는 설명이 필요해 보입니다.

### FC → Convolutional Layer

우린 어떻게 Fully Connected layer를 Convolution layer로 대체 할수 있을까요?

아래 시나리오(Scenario)를 따라가봅시다.

일반적인 image classification setting에서 첫번째 Fully Connected layer는 featrue map이나 image를 바로 받아서 동작할수 없고 **flatten**이라는 과정을 거쳐서 1D structure로 변환한 뒤에 적용할수 있게 됩니다.

![](./img/fcconv1.jpg)

이 예제에서, `2x2x3`에 해당하는 이미지를 사용합니다. **flattened**된 이미지는 1D vector로 크기가 `12`가 됩니다. 출력의 크기가 `2`인 Fully connected layer는 flattened image인 길이 `12`의 이미지에 대해 2번의 dot-products 를 진행합니다. **두번의 dot product시에 사용되는 1D vector는, 회색으로, 이것의 parameter들은 fully-connected layer의 parameter들 입니다.**

이제 Convolution layer를 이용해서 같은 수의  output을 만들어내는 시나리오(Senario2) 를 고려해 봅시다.

![](./img/fcconv2.jpg)

여기 차원이 `2x2x3`인 이미지가 있습니다. 이제는 확실히 flattened이 필요하지 않습니다. 여기서 Convolutional layer는 image와 동일한 모양의 `12`의 element를 가지고 있는 2개의 layer를 통해서 진행됩니다. **이 두개의 filter는 회색으로 표시되어 있습니다. 이건 Convolution layer의 parameter들 입니다.**

이 두가지에서 꼭 확인해야 하는 것은 **두개의 시나리오 속 출력 `Y_0`와 `Y_1`는 같다는 것 입니다.**

![](./img/fcconv3.jpg)

두개의 시나리오(Scenario1 and Senario2)는 동일 합니다.

이건 무엇을 말할까요?

**`H, W` with `I`의 크기를 가지는 input image와 output의 크기가 `N`인 fully-connected layer는 convolution layer의 크기가 input image의 크기인 `H, W`와 같고 출력 채널이 `N` 으로 같은 경우와 동일 합니다.** 주어진 fully connected network의 parameter인 `N, H * W * I`에 해당 하는 값은 컨벌루션의 `N, H, W, I`에 해당하는 값과 같습니다.

![](./img/fcconv4.jpg)

그러므로 fully connected layer는 parameter reshape를 통해 간단하게 동등한 convolution layer로 전환 될 수 있습니다.

### Base Convolutions – part 2

우린 이제 기존의 VGG-16 구조에 있는 `fc6`와 `fc7`이 `conv6`와 `conv7`으로 어떻게 바뀔수 있는지를 알고 있습니다. 

입력 이미지의 크기가 `224,224,3`인 ImageNet VGG-16 [shown previously](https://github.com/dldldlfma/pytorch_tutorial_ssd#base-convolutions--part-1) 에서 우리는 `conv5_3`의 출력이 `7, 7, 512`이 되는 것을 확인 할수 있습니다.
그러므로 -

- 펼쳐진 입력 크기가 `7 * 7 * 512`인 `fc6` 그리고 `4096`의 출력크기는 `4096, 7 * 7 * 512` 라는 차원을 가집니다. **동등한 Convolutional Layer `conv6`는 `7, 7`의 kernel size와 `4096`의 출력채널을 가지는데 해당 kernel의 parameter 값의 차원은 기존 `fc6`의 parameter를 reshape한 형태인 `4096, 7, 7, 512` 입니다.**

- 입력 크기가 `4096`(i.e. `fc6`의 출력 크기)이고 출력 크기가 `4096`인 `fc7`의 parameter는 `4096, 4096`의 차원을 가집니다. 입력은 `1, 1`크기에 `4096`의 입력 채널을 갖는 이미지로 고려될 수 있습니다. **동등한 Convolutional layer `conv7`은 `1, 1`의 kernel size와 `4096` 의 output channel을 가지는데 이는 기존의 `fc7`의 parameter를 reshape한 형태인 `4096, 1, 1, 4096`입니다.**

우리는 `conv6`가 `7, 7, 512`크기의 `4096`개의 필터를 가지고 있는 것을 볼수 있습니다. 그리고 `conv7`은 `1, 1, 4096`크기의 `4096`개 필터들을 가지고 있는 것도 볼수 있습니다.

이 필터들은 엄청나게 크고 – 계산할 양이 너무 많습니다.

이 문제를 해결 하기 위해서 , 저자는 전환된 Convolutional layer들에서 **subsampling parameters를 이용해서 각 필터에 맞게 그들의 숫자와 크기를 줄이는 방법**을 선택했습니다.

- `conv6`는 `1024`개의 filter들을 `3, 3, 512`에 해당하는 각 차원에 대해 사용할 것입니다. 그러므로, parameter들은 `4096, 7, 7, 512`에서 `1024, 3, 3, 512`로 subsample 됩니다.

- `conv7` will use `1024` filters, each with dimensions `1, 1, 1024`. Therefore, the parameters are subsampled from `4096, 1, 1, 4096` to `1024, 1, 1, 1024`.

- `conv7`는 `1024`개의 필터들을, `1, 1, 1024`의 각 차원에 대해 사용합니다. 그러므로 parameter들은 `4096, 1, 1, 4096`에서 `1024, 1, 1, 1024`로 subsample 됩니다.

논문의 참조에 따르면, 우리는 [_decimation_](https://en.wikipedia.org/wiki/Downsampling_(signal_processing))라고 알려진 **특별한 차원에 속한 `m`번째 파라미터마다 골라 내는 방식으로 subsample을 수행할 것입니다.** => (`m`번째 파라미터만 남긴다).

Since the kernel of `conv6` is decimated from `7, 7` to `3,  3` by keeping only every 3rd value, there are now _holes_ in the kernel. Therefore, we would need to **make the kernel dilated or _atrous_**.

3번째 해당하는 값마다 남기는 방식으로 `conv6`에 decimate를 수행해서 `7, 7`크기의 필터가 `3,  3`필터로 변경된 이래로, 커널에는 _빈공간_ 이 발생한다. 그래서 우리는 **kernel dilated 또는 _atrous_**로 불리는 방식을 적용한 Convolution을 사용할 필요가 있다. 아래 GIF는 3x3 filter로 5x5 convolution을 수행하는 Dilated_conv의 연산 방식이다. 

![](./explain_data/Dilated_conv.GIF)

이것은 `3`만큼의 확장을 의미합니다. (decimation factor `m = 3`를 적용한 것과 동일). 그러나, 저자는 실제론 dilation of `6`를 적용합니다. 아마 5번째 pooling layer가 더 이상 feature map의 크기를 절반으로 줄이지 않기 때문일겁니다.

우리는 이제 base network인 **수정된 VGG-16**의 형태를 제시할 수 있습니다.

![](./img/modifiedvgg.PNG)

위 그림에서 특별히 `conv4_3`과 `conv_7`의 출력을 집중해서 봐두시기 바랍니다. 곧 왜 이런 말을 했는지 알게 될겁니다.

### Auxiliary Convolutions(보조 Convolution)

우린 이제 **기본 네트워크 위에 몇개의 convolution layer를 추가할 겁니다**. 이 convolution layer들은 추가적인 feature map들을 제공하고, 그 feature map 크기는 앞선 크기보다 점진적으로 작아질겁니다.

![](./img/auxconv.jpg)

We introduce four convolutional blocks, each with two layers. While size reduction happened through pooling in the base network, here it is facilitated by a stride of `2` in every second layer.

우린 여기서 각각 2개의 layer와 함께 있는 4개의 convolutional block를 소개합니다. base network에서는 pooling layer를 통해 사이즈 감소가 일어난 반면, 추가된 4개의 convolution block에서는 block당 두번째 layer의 convolution에서 stride `2`를 이용해 사이즈 감소를 만들어 냅니다.

다시 주의 깊게 `conv8_2`, `conv9_2`, `conv10_2`, 그리고 `conv11_2`의 출력을 확인해보세요.

### A detour

Before we move on to the prediction convolutions, we must first understand what it is we are predicting. Sure, it's objects and their positions, _but in what form?_

It is here that we must learn about _priors_ and the crucial role they play in the SSD.


prediction convolutions로 넘어가기전에, 우리가 뭘 예측하는 것인지를 반드시 이해해야 합니다. 당연히 그건 객체와 객체의 위치일건데, _이게 어떤 형태로 존재하는 걸까요?_

여기서 우리는 반드시 _priors_에 대해서 배우고 SSD를 수행하면서 그것이 가지고 있는 가장 중점적인 역할을 알게 될겁니다.



#### Priors

Object predictions can be quite diverse, and I don't just mean their type. They can occur at any position, with any size and shape. Mind you, we shouldn't go as far as to say there are _infinite_ possibilities for where and how an object can occur. While this may be true mathematically, many options are simply improbable or uninteresting. Furthermore, we needn't insist that boxes are pixel-perfect.

Object predictions은 엄청 넓게 정의 될수 있습니다. 근데 이건 단순하게 type을 정의하는 문제가 아닙니다. 이건 어떤 위치에서든, 다양한 크기와 형태로 나타날수 있죠. 그렇다고 해서 우리가 예측할 대상이 정말 아무 위치에서나 등장할 수 있다고 보는 건 무리입니다. 수학적으로 그럴수는 있겠지만, 대부분의 결과는 간단하게 의미가 없거나 별 상관없는 것들입니다. 추가적으로 , 우리는 픽셀단위까지 완벽한 object detection을 구현할 필요가 없습니다.

In effect, we can discretize the mathematical space of potential predictions into just _thousands_ of possibilities.

문제 해결을 위해 우리는 몇 천개에 달하는 예측공간을 수학적으로 분리해둘 수 있습니다.

**Priors are precalculated, fixed boxes which collectively represent this universe of probable and approximate box predictions**.

**Priors는 미리 계산된 fixed box들 입니다. 이것들은 상자의 대략적 예측과 전체 공간에서의 가능성을 적절하게 표현합니다**.

Priors are manually but carefully chosen based on the shapes and sizes of ground truth objects in our dataset. By placing these priors at every possible location in a feature map, we also account for variety in position.

Priors는 수작업으로 하지만 조심스럽게 선택 됩니다. 선택의 기준은 우리가 사용할 데이터 셋의 객체들의 모양과 크기를 입니다. feature map에서 모든 가능성있는 위치에 대한 priors를 설정함으로써, 우리는 위치의 다양성 문제를 다룹니다. 

In defining the priors, the authors specify that –

priors를 정의하면서, 작가는 아래의 것들을 명확히 했습니다 –

- **they will be applied to various low-level and high-level feature maps**, viz. those from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2`, and `conv11_2`. These are the same feature maps indicated on the figures before.

- **priors는 low-level부터 high-level까지 다양한 feature map들에서 적용될수 있어야 한다**, 즉. `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2`, 그리고 `conv11_2`에 이르는 레이어에서도 모두 적절한 형태여야 한다는 것입니다. These are the same feature maps indicated on the figures before.


- **if a prior has a scale `s`, then its area is equal to that of a square with side `s`**. The largest feature map, `conv4_3`, will have priors with a scale of `0.1`, i.e. `10%` of image's dimensions, while the rest have priors with scales linearly increasing from `0.2` to `0.9`. As you can see, larger feature maps have priors with smaller scales and are therefore ideal for detecting smaller objects.


- **if a prior has a scale `s`, then its area is equal to that of a square with side `s`**. 가장 큰 featrue map인 `conv4_3`는 `0.1`의 scale을 가지는 priors를 가질 것 입니다. 이 것은 이미지 차원의 `10%`를 의미하는 것이고, 반면에 나머지 layer들은 priors 그 scales이 `0.2`에서 `0.9`까지 선형적으로 증가하는 priors를 가질겁니다. 당신이 보는 것 처럼, 더 큰 feature maps은 더 작은 scale의 priors를 가지는데 이는 작은 객체를 찾아내기에 이상적 입니다.


- **At _each_ position on a feature map, there will be priors of various aspect ratios**. All feature maps will have priors with ratios `1:1, 2:1, 1:2`. The intermediate feature maps of `conv7`, `conv8_2`, and `conv9_2` will _also_ have priors with ratios `3:1, 1:3`. Moreover, all feature maps will have *one extra prior* with an aspect ratio of `1:1` and at a scale that is the geometric mean of the scales of the current and subsequent feature map.


- **feature map의 _각_ 위치에서, 다양한 aspect ratios를 가지는 priors를 가지게 됩니다.**. 모든 feature maps은 `1:1, 2:1, 1:2`비율의 priors를 가질겁니다. 중간에 위치한 `conv7`, `conv8_2`, 그리고 `conv9_2`의 feature maps은 추가적으로 `3:1, 1:3`비율의 priors를 가집니다. 더하여, 모든 feature maps은 `1:1` 비율인 *한개의 추가적인 prior*을 가집니다. and at a scale that is the geometric mean of the scales of the current and subsequent feature map.


| Feature Map From | Feature Map Dimensions | Prior Scale | Aspect Ratios | Number of Priors per Position | Total Number of Priors on this Feature Map |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| `conv4_3`      | 38, 38       | 0.1 | 1:1, 2:1, 1:2 + an extra prior | 4 | 5776 |
| `conv7`      | 19, 19       | 0.2 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 2166 |
| `conv8_2`      | 10, 10       | 0.375 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 600 |
| `conv9_2`      | 5, 5       | 0.55 | 1:1, 2:1, 1:2, 3:1, 1:3 + an extra prior | 6 | 150 |
| `conv10_2`      | 3,  3       | 0.725 | 1:1, 2:1, 1:2 + an extra prior | 4 | 36 |
| `conv11_2`      | 1, 1       | 0.9 | 1:1, 2:1, 1:2 + an extra prior | 4 | 4 |
| **Grand Total**      |    –    | – | – | – | **8732 priors** |

There are a total of 8732 priors defined for the SSD300!

SSD300은 8732개의 priors를 가지고 있습니다.

#### Visualizing Priors (Priors 시각화)

We defined the priors in terms of their _scales_ and _aspect ratios_.

우리는 priors를 _scales_과 _aspect ratios_관점에서 정의 합니다.

![](./img/wh1.jpg)

Solving these equations yields a prior's dimensions `w` and `h`.

이 방정식을 풀면 prior의 `w`와 `h`가 산출 됩니다. 

![](./img/wh2.jpg)

We're now in a position to draw them on their respective feature maps.

이제 각각의 feature map에서 그릴수 있습니다. 

For example, let's try to visualize what the priors will look like at the central tile of the feature map from `conv9_2`.

예를들어 `conv9_2`에서 feature map의 중간 타일 위치에 prior의 모습을 그려봅시다.


![](./img/priors1.jpg)

The same priors also exist for each of the other tiles.

각 tile 마다 동일한 priors가 존재합니다. 

![](./img/priors2.jpg)

#### Predictions vis-à-vis Priors

[Earlier](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#multibox), we said we would use regression to find the coordinates of an object's bounding box. But then, surely, the priors can't represent our final predicted boxes?

[이전에](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#multibox), 우리가 확인한 것중에 객체 bounding box의 좌표를 찾기 위해서 Regression을 사용한다는 것이 있었습니다. 그러면 여기서, 확실하게 priors가 최종 예측 박스를 대표할 수 없을까?

They don't.

They don't.


Again, I would like to reiterate that the priors represent, _approximately_, the possibilities for prediction.

다시, 나는 priors의 대략적인 예측 가능성을 나타내고 싶다.

This means that **we use each prior as an approximate starting point and then find out how much it needs to be adjusted to obtain a more exact prediction for a bounding box**.


이 뜻은 **우리는 각 prior를 대략적인 시작점으로 그리고 예측된 bounding box를 얼마나 더 수정할 필요가 있는지를 확인하는 용도로 사용하는것을 의미한다.**.


So if each predicted bounding box is a slight deviation from a prior, and our goal is to calculate this deviation, we need a way to measure or quantify it.


그래서 만약에 예측된 bounding box 각각이 prior과 약간의 편차가 있다면, 그리고 우리의 목표가 이 편차를 계산하는 거라면 우리는 이것을 정량적으로 나타내기 위한 방법이 필요하다. 

Consider a cat, its predicted bounding box, and the prior with which the prediction was made.  

아래 그림의 고양이를 고려해보자, 고양이 예측 상자와 예측을 나타내는 prior가 있다.

![](./img/ecs1.PNG)

Assume they are represented in center-size coordinates, which we are familiar with.

이것은 우리에게 익숙한 중심 크기 좌표로 표현되어 있다고 가정합니다. 


Then –

그때

![](./img/ecs2.PNG)

This answers the question we posed at the [beginning of this section](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#a-detour). Considering that each prior is adjusted to obtain a more precise prediction, **these four offsets `(g_c_x, g_c_y, g_w, g_h)` are the form in which we will regress bounding boxes' coordinates**.

이건 [이 섹션의 시작의 질문](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#a-detour)에 대한 답변입니다. 
(질문은 예측이 최종박스를 대표하도록 만들수 있을까?)

각 prior은 좀더 정확한 예측으로 수정된다는 것을 고려해보자.
**이것의 4가지 offsets인 `(g_c_x, g_c_y, g_w, g_h)`는 bonding box의 좌표를 회귀로 계산 시킬수 있도록 표현한 것입니다.**

As you can see, each offset is normalized by the corresponding dimension of the prior. This makes sense because a certain offset would be less significant for a larger prior than it would be for a smaller prior.

당신이 보는것 처럼, 각 offset은 prior의 해당 치수로 정규화 됩니다. 
이것은 특정 offsets이 prior보다 작을때 보다 더 클수록 중요하지 않기 떄문입니다.

### Prediction convolutions

Earlier, we earmarked and defined priors for six feature maps of various scales and granularity, viz. those from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2`, and `conv11_2`.

Then, **for _each_ prior at _each_ location on _each_ feature map**, we want to predict –

- the **offsets `(g_c_x, g_c_y, g_w, g_h)`** for a bounding box.

- a set of **`n_classes` scores** for the bounding box, where `n_classes` represents the total number of object types (including a _background_ class).

To do this in the simplest manner possible, **we need two convolutional layers for each feature map** –

- a **_localization_ prediction** convolutional layer with a `3,  3` kernel evaluating at each location (i.e. with padding and stride of `1`) with `4` filters for _each_ prior present at the location.

  The `4` filters for a prior calculate the four encoded offsets `(g_c_x, g_c_y, g_w, g_h)` for the bounding box predicted from that prior.

- a **_class_ prediction** convolutional layer with a `3,  3` kernel evaluating at each location (i.e. with padding and stride of `1`) with `n_classes` filters for _each_ prior present at the location.

  The `n_classes` filters for a prior calculate a set of `n_classes` scores for that prior.

![](./img/predconv1.jpg)

All our filters are applied with a kernel size of `3, 3`.

We don't really need kernels (or filters) in the same shapes as the priors because the different filters will _learn_ to make predictions with respect to the different prior shapes.

Let's take a look at the **outputs of these convolutions**. Consider again the feature map from `conv9_2`.

![](./img/predconv2.jpg)

The outputs of the localization and class prediction layers are shown in blue and yellow respectively. You can see that the cross-section (`5, 5`) remains unchanged.

What we're really interested in is the _third_ dimension, i.e. the channels. These contain the actual predictions.

If you **choose a tile, _any_ tile, in the localization predictions and expand it**, what will you see?

![](./img/predconv3.jpg)

Voilà! The channel values at each position of the localization predictions represent the encoded offsets with respect to the priors at that position.

Now, **do the same with the class predictions.** Assume `n_classes = 3`.

![](./img/predconv4.jpg)

Similar to before, these channels represent the class scores for the priors at that position.

Now that we understand what the predictions for the feature map from `conv9_2` look like, we can **reshape them into a more amenable form.**

![](./img/reshaping1.jpg)

We have arranged the `150` predictions serially. To the human mind, this should appear more intuitive.

But let's not stop here. We could do the same for the predictions for _all_ layers and stack them together.

We calculated earlier that there are a total of 8732 priors defined for our model. Therefore, there will be **8732 predicted boxes in encoded-offset form, and 8732 sets of class scores**.

![](./img/reshaping2.jpg)

**This is the final output of the prediction stage.** A stack of boxes, if you will, and estimates for what's in them.

It's all coming together, isn't it? If this is your first rodeo in object detection, I should think there's now a faint light at the end of the tunnel.

### Multibox loss

Based on the nature of our predictions, it's easy to see why we might need a unique loss function. Many of us have calculated losses in regression or classification settings before, but rarely, if ever, _together_.

Obviously, our total loss must be an **aggregate of losses from both types of predictions** – bounding box localizations and class scores.

Then, there are a few questions to be answered –

>_What loss function will be used for the regressed bounding boxes?_

>_Will we use multiclass cross-entropy for the class scores?_

>_In what ratio will we combine them?_

>_How do we match predicted boxes to their ground truths?_

>_We have 8732 predictions! Won't most of these contain no object? Do we even consider them?_

Phew. Let's get to work.

#### Matching predictions to ground truths

Remember, the nub of any supervised learning algorithm is that **we need to be able to match predictions to their ground truths**. This is tricky since object detection is more open-ended than the average learning task.

For the model to learn _anything_, we'd need to structure the problem in a way that allows for comparisions between our predictions and the objects actually present in the image.

Priors enable us to do exactly this!

- **Find the Jaccard overlaps** between the 8732 priors and `N` ground truth objects. This will be a tensor of size `8732, N`.

- **Match** each of the 8732 priors to the object with which it has the greatest overlap.

- If a prior is matched with an object with a **Jaccard overlap of less than `0.5`**, then it cannot be said to "contain" the object, and is therefore a **_negative_ match**. Considering we have thousands of priors, most priors will test negative for an object.

- On the other hand, a handful of priors will actually **overlap significantly (greater than `0.5`)** with an object, and can be said to "contain" that object. These are **_positive_ matches**.

- Now that we have **matched each of the 8732 priors to a ground truth**, we have, in effect, also **matched the corresponding 8732 predictions to a ground truth**.  

Let's reproduce this logic with an example.

![](./img/matching1.PNG)

For convenience, we will assume there are just seven priors, shown in red. The ground truths are in yellow – there are three actual objects in this image.

Following the steps outlined earlier will yield the following matches –

![](./img/matching2.jpg)

Now, **each prior has a match**, positive or negative. By extension, **each prediction has a match**, positive or negative.

Predictions that are positively matched with an object now have ground truth coordinates that will serve as **targets for localization**, i.e. in the _regression_ task. Naturally, there will be no target coordinates for negative matches.

All predictions have a ground truth label, which is either the type of object if it is a positive match or a _background_ class if it is a negative match. These are used as **targets for class prediction**, i.e. the _classification_ task.

#### Localization loss

We have **no ground truth coordinates for the negative matches**. This makes perfect sense. Why train the model to draw boxes around empty space?

Therefore, the localization loss is computed only on how accurately we regress positively matched predicted boxes to the corresponding ground truth coordinates.

Since we predicted localization boxes in the form of offsets `(g_c_x, g_c_y, g_w, g_h)`, we would also need to encode the ground truth coordinates accordingly before we calculate the loss.

The localization loss is the averaged **Smooth L1** loss between the encoded offsets of positively matched localization boxes and their ground truths.

![](./img/locloss.jpg)

#### Confidence loss

Every prediction, no matter positive or negative, has a ground truth label associated with it. It is important that the model recognizes both objects and a lack of them.

However, considering that there are usually only a handful of objects in an image, **the vast majority of the thousands of predictions we made do not actually contain an object**. As Walter White would say, _tread lightly_. If the negative matches overwhelm the positive ones, we will end up with a model that is less likely to detect objects because, more often than not, it is taught to detect the _background_ class.

The solution may be obvious – limit the number of negative matches that will be evaluated in the loss function. But how do we choose?

Well, why not use the ones that the model was most _wrong_ about? In other words, only use those predictions where the model found it hardest to recognize that there are no objects. This is called **Hard Negative Mining**.

The number of hard negatives we will use, say `N_hn`, is usually a fixed multiple of the number of positive matches for this image. In this particular case, the authors have decided to use three times as many hard negatives, i.e. `N_hn = 3 * N_p`. The hardest negatives are discovered by finding the Cross Entropy loss for each negatively matched prediction and choosing those with top `N_hn` losses.

Then, the confidence loss is simply the sum of the **Cross Entropy** losses among the positive and hard negative matches.

![](./img/confloss.jpg)

You will notice that it is averaged by the number of positive matches.

#### Total loss

The **Multibox loss is the aggregate of the two losses**, combined in a ratio `α`.

![](./img/totalloss.jpg)

In general, we needn't decide on a value for `α`. It could be a learnable parameter.

For the SSD, however, the authors simply use `α = 1`, i.e. add the two losses. We'll take it!

### Processing predictions

After the model is trained, we can apply it to images. However, the predictions are still in their raw form – two tensors containing the offsets and class scores for 8732 priors. These would need to be processed to **obtain final, human-interpretable bounding boxes with labels.**

This entails the following –

- We have 8732 predicted boxes represented as offsets `(g_c_x, g_c_y, g_w, g_h)` from their respective priors. Decode them to boundary coordinates, which are actually directly interpretable.

- Then, for each _non-background_ class,

  - Extract the scores for this class for each of the 8732 boxes.

  - Eliminate boxes that do not meet a certain threshold for this score.

  - The remaining (uneliminated) boxes are candidates for this particular class of object.

At this point, if you were to draw these candidate boxes on the original image, you'd see **many highly overlapping boxes that are obviously redundant**. This is because it's extremely likely that, from the thousands of priors at our disposal, more than one prediction corresponds to the same object.

For instance, consider the image below.

![](./img/nms1.PNG)

There's clearly only three objects in it – two dogs and a cat. But according to the model, there are _three_ dogs and _two_ cats.

Mind you, this is just a mild example. It could really be much, much worse.

Now, to you, it may be obvious which boxes are referring to the same object. This is because your mind can process that certain boxes coincide significantly with each other and a specific object.

In practice, how would this be done?

First, **line up the candidates for each class in terms of how _likely_ they are**.

![](./img/nms2.PNG)

We've sorted them by their scores.

The next step is to find which candidates are redundant. We already have a tool at our disposal to judge how much two boxes have in common with each other – the Jaccard overlap.

So, if we were to **draw up the Jaccard similarities between all the candidates in a given class**, we could evaluate each pair and **if found to overlap significantly, keep only the _more likely_ candidate**.

![](./img/nms3.jpg)

Thus, we've eliminated the rogue candidates – one of each animal.

This process is called __Non-Maximum Suppression (NMS)__ because when multiple candidates are found to overlap significantly with each other such that they could be referencing the same object, **we suppress all but the one with the maximum score**.

Algorithmically, it is carried out as follows –

- Upon selecting candidades for each _non-background_ class,

  - Arrange candidates for this class in order of decreasing likelihood.

  - Consider the candidate with the highest score. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than, say, `0.5` with this candidate.

  - Consider the next highest-scoring candidate still remaining in the pool. Eliminate all candidates with lesser scores that have a Jaccard overlap of more than `0.5` with this candidate.

  - Repeat until you run through the entire sequence of candidates.

The end result is that you will have just a single box – the very best one – for each object in the image.

![](./img/nms4.PNG)

Non-Maximum Suppression is quite crucial for obtaining quality detections.

Happily, it's also the final step.

# Implementation

The sections below briefly describe the implementation.

They are meant to provide some context, but **details are best understood directly from the code**, which is quite heavily commented.

### Dataset

We will use Pascal Visual Object Classes (VOC) data from the years 2007 and 2012.

#### Description

This data contains images with twenty different types of objects.

```python
{'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
```

Each image can contain one or more ground truth objects.

Each object is represented by –

- a bounding box in absolute boundary coordinates

- a label (one of the object types mentioned above)

-  a perceived detection difficulty (either `0`, meaning _not difficult_, or `1`, meaning _difficult_)

#### Download

Specfically, you will need to download the following VOC datasets –

- [2007 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (460MB)

- [2012 _trainval_](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (2GB)

- [2007 _test_](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (451MB)

Consistent with the paper, the two _trainval_ datasets are to be used for training, while the VOC 2007 _test_ will serve as our validation and testing data.  

Make sure you extract both the VOC 2007 _trainval_ and 2007 _test_ data to the same location, i.e. merge them.

### Inputs to model

We will need three inputs.

#### Images

Since we're using the SSD300 variant, the images would need to be sized at `300, 300` pixels and in the RGB format.

Remember, we're using a VGG-16 base pretrained on ImageNet that is already available in PyTorch's `torchvision` module. [This page](https://pytorch.org/docs/master/torchvision/models.html) details the preprocessing or transformation we would need to perform in order to use this model – pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.

Therefore, **images fed to the model must be a `Float` tensor of dimensions `N, 3, 300, 300`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.

#### Objects' Bounding Boxes

We would need to supply, for each image, the bounding boxes of the ground truth objects present in it in fractional boundary coordinates `(x_min, y_min, x_max, y_max)`.

Since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the bounding boxes for the entire batch of `N` images.

Therefore, **ground truth bounding boxes fed to the model must be a list of length `N`, where each element of the list is a `Float` tensor of dimensions `N_o, 4`**, where `N_o` is the number of objects present in that particular image.

#### Objects' Labels

We would need to supply, for each image, the labels of the ground truth objects present in it.

Each label would need to be encoded as an integer from `1` to `20` representing the twenty different object types. In addition, we will add a _background_ class with index `0`, which indicates the absence of an object in a bounding box. (But naturally, this label will not actually be used for any of the ground truth objects in the dataset.)

Again, since the number of objects in any given image can vary, we can't use a fixed size tensor for storing the labels for the entire batch of `N` images.

Therefore, **ground truth labels fed to the model must be a list of length `N`, where each element of the list is a `Long` tensor of dimensions `N_o`**, where `N_o` is the number of objects present in that particular image.

### Data pipeline

As you know, our data is divided into _training_ and _test_ splits.

#### Parse raw data

See `create_data_lists()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This parses the data downloaded and saves the following files –

- A **JSON file for each split with a list of the absolute filepaths of `I` images**, where `I` is the total number of images in the split.

- A **JSON file for each split with a list of `I` dictionaries containing ground truth objects, i.e. bounding boxes in absolute boundary coordinates, their encoded labels, and perceived detection difficulties**. The `i`th dictionary in this list will contain the objects present in the `i`th image in the previous JSON file.

- A **JSON file which contains the `label_map`**, the label-to-index dictionary with which the labels are encoded in the previous JSON file. This dictionary is also available in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) and directly importable.

#### PyTorch Dataset

See `PascalVOCDataset` in [`datasets.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py).

This is a subclass of PyTorch [`Dataset`](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset), used to **define our training and test datasets.** It needs a `__len__` method defined, which returns the size of the dataset, and a `__getitem__` method which returns the `i`th image, bounding boxes of the objects in this image, and labels for the objects in this image, using the JSON files we saved earlier.

You will notice that it also returns the perceived detection difficulties of each of these objects, but these are not actually used in training the model. They are required only in the [Evaluation](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#evaluation) stage for computing the Mean Average Precision (mAP) metric. We also have the option of filtering out _difficult_ objects entirely from our data to speed up training at the cost of some accuracy.

Additionally, inside this class, **each image and the objects in them are subject to a slew of transformations** as described in the paper and outlined below.

#### Data Transforms

See `transform()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This function applies the following transformations to the images and the objects in them –

- Randomly **adjust brightness, contrast, saturation, and hue**, each with a 50% chance and in random order.

- With a 50% chance, **perform a _zoom out_ operation** on the image. This helps with learning to detect small objects. The zoomed out image must be between `1` and `4` times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.

- Randomly crop image, i.e. **perform a _zoom in_ operation.** This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between `0.3` and `1` times the original dimensions. The aspect ratio is to be between `0.5` and `2`. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either `0`, `0.1`, `0.3`, `0.5`, `0.7`, or `0.9`, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.

- With a 50% chance, **horizontally flip** the image.

- **Resize** the image to `300, 300` pixels. This is a requirement of the SSD300.

- Convert all boxes from **absolute to fractional boundary coordinates.** At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.

- **Normalize** the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.

As mentioned in the paper, these transformations play a crucial role in obtaining the stated results.

#### PyTorch DataLoader

The `Dataset` described above, `PascalVOCDataset`, will be used by a PyTorch [`DataLoader`](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) in `train.py` to **create and feed batches of data to the model** for training or validation.

Since the number of objects vary across different images, their bounding boxes, labels, and difficulties cannot simply be stacked together in the batch. There would be no way of knowing which objects belong to which image.

Instead, we need to **pass a collating function to the `collate_fn` argument**, which instructs the `DataLoader` about how it should combine these varying size tensors. The simplest option would be to use Python lists.

### Base Convolutions

See `VGGBase` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply base convolutions.**

The layers are initialized with parameters from a pretrained VGG-16 with the `load_pretrained_layers()` method.

We're especially interested in the lower-level feature maps that result from `conv4_3` and `conv7`, which we return for use in subsequent stages.

### Auxiliary Convolutions

See `AuxiliaryConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply auxiliary convolutions.**

Use a [uniform Xavier initialization](https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_uniform_) for the parameters of these layers.

We're especially interested in the higher-level feature maps that result from `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, which we return for use in subsequent stages.

### Prediction Convolutions

See `PredictionConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply localization and class prediction convolutions** to the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`.

These layers are initialized in a manner similar to the auxiliary convolutions.

We also **reshape the resulting prediction maps and stack them** as discussed. Note that reshaping in PyTorch is only possible if the original tensor is stored in a [contiguous](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous) chunk of memory.

As expected, the stacked localization and class predictions will be of dimensions `8732, 4` and `8732, 21` respectively.

### Putting it all together

See `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, the **base, auxiliary, and prediction convolutions are combined** to form the SSD.

There is a small detail here – the lowest level features, i.e. those from `conv4_3`, are expected to be on a significantly different numerical scale compared to its higher-level counterparts. Therefore, the authors recommend L2-normalizing and then rescaling _each_ of its channels by a learnable value.

### Priors

See `create_prior_boxes()` under `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

This function **creates the priors in center-size coordinates** as defined for the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, _in that order_. Furthermore, for each feature map, we create the priors at each tile by traversing it row-wise.

This ordering of the 8732 priors thus obtained is very important because it needs to match the order of the stacked predictions.

### Multibox Loss

See `MultiBoxLoss` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Two empty tensors are created to store localization and class prediction targets, i.e. _ground truths_, for the 8732 predicted boxes in each image.

We **find the ground truth object with the maximum Jaccard overlap for each prior**, which is stored in `object_for_each_prior`.

We want to avoid the rare situation where not all of the ground truth objects have been matched. Therefore, we also **find the prior with the maximum overlap for each ground truth object**, stored in `prior_for_each_object`. We explicitly add these matches to `object_for_each_prior` and artificially set their overlaps to a value above the threshold so they are not eliminated.

Based on the matches in `object_for_each prior`, we set the corresponding labels, i.e. **targets for class prediction**, to each of the 8732 priors. For those priors that don't overlap significantly with their matched objects, the label is set to _background_.

Also, we encode the coordinates of the 8732 matched objects in `object_for_each prior` in offset form `(g_c_x, g_c_y, g_w, g_h)` with respect to these priors, to form the **targets for localization**. Not all of these 8732 localization targets are meaningful. As we discussed earlier, only the predictions arising from the non-background priors will be regressed to their targets.

The **localization loss** is the [Smooth L1 loss](https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss) over the positive matches.

Perform Hard Negative Mining – rank class predictions matched to _background_, i.e. negative matches, by their individual [Cross Entropy losses](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss). The **confidence loss** is the Cross Entropy loss over the positive matches and the hardest negative matches. Nevertheless, it is averaged only by the number of positive matches.

The **Multibox Loss is the aggregate of these two losses**, combined in the ratio `α`. In our case, they are simply being added because `α = 1`.

# Training

Before you begin, make sure to save the required data files for training and validation. To do this, run the contents of [`create_data_lists.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/create_data_lists.py) after pointing it to the `VOC2007` and `VOC2012` folders in your [downloaded data](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#download).

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py).

The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you need to.

To **train your model from scratch**, run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.

Note that we perform validation at the end of every training epoch.

### Remarks

In the paper, they recommend using **Stochastic Gradient Descent** in batches of `32` images, with an initial learning rate of `1e−3`, momentum of `0.9`, and `5e-4` weight decay.

I ended up using a batch size of `8` images for increased stability. If you find that your gradients are exploding, you could reduce the batch size, like I did, or clip gradients.

The authors also doubled the learning rate for bias parameters. As you can see in the code, this is easy do in PyTorch, by passing [separate groups of parameters](https://pytorch.org/docs/stable/optim.html#per-parameter-options) to the `params` argument of its [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD).

The paper recommends training for 80000 iterations at the initial learning rate. Then, it is decayed by 90% for an additional 20000 iterations, _twice_. With the paper's batch size of `32`, this means that the learning rate is decayed by 90% once at the 155th epoch and once more at the 194th epoch, and training is stopped at 232 epochs.

In practice, I just decayed the learning rate by 90% when the validation loss stopped improving for long periods. I resumed training at this reduced learning rate from the best checkpoint obtained thus far, not the most recent.

On a TitanX (Pascal), each epoch of training required about 6 minutes. My best checkpoint was from epoch 186, with a validation loss of `2.515`.

### Model checkpoint

You can download this pretrained model [here](https://drive.google.com/file/d/1YZp2PUR1NYKPlBIVoVRO0Tg1ECDmrnC3/view?usp=sharing).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load) for evaluation or inference – see below.

# Evaluation

See [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py).

The data-loading and checkpoint parameters for evaluating the model are at the beginning of the file, so you can easily check or modify them should you wish to.

To begin evaluation, simply run the `evaluate()` function with the data-loader and model checkpoint. **Raw predictions for each image in the test set are obtained and parsed** with the checkpoint's `detect_objects()` method, which implements [this process](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#processing-predictions). Evaluation has to be done at a `min_score` of `0.01`, an NMS `max_overlap` of `0.45`, and `top_k` of `200` to allow fair comparision of results with the paper and other implementations.

**Parsed predictions are evaluated against the ground truth objects.** The evaluation metric is the _Mean Average Precision (mAP)_. If you're not familiar with this metric, [here's a great explanation](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

We will use `calculate_mAP()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) for this purpose. As is the norm, we will ignore _difficult_ detections in the mAP calculation. But nevertheless, it is important to include them from the evaluation dataset because if the model does detect an object that is considered to be _difficult_, it must not be counted as a false positive.

The model scores **77.1 mAP**, against the 77.2 mAP reported in the paper.

Class-wise average precisions are listed below.

| Class | Average Precision |
| :-----: | :------: |
| aeroplane |  78.9|
|  bicycle | 83.7|
|  bird |  76.9|
|  boat |  72.0|
|  bottle |  46.0|
|  bus |  86.7|
|  car |  86.9|
|  cat |  89.2|
|  chair |  59.6|
|  cow |  82.7|
|  diningtable |  75.2|
|  dog |  85.6|
|  horse |  87.4|
|  motorbike |  82.9|
|  person |  78.8|
|  pottedplant |  50.3|
|  sheep |  78.7|
|  sofa |  80.5|
|  train |  85.7|
|  tvmonitor |  75.0|

You can see that some objects, like bottles and potted plants, are considerably harder to detect than others.

# Inference

See [`detect.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py).

Point to the model you want to use for inference with the `checkpoint` parameter at the beginning of the code.

Then, you can use the `detect()` function to identify and visualize objects in an RGB image.

```python
img_path = '/path/to/ima.ge'
original_image = PIL.Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
```

This function first **preprocesses the image by resizing and normalizing its RGB channels** as required by the model. It then **obtains raw predictions from the model, which are parsed** by the `detect_objects()` method in the model. The parsed results are converted from fractional to absolute boundary coordinates, their labels are decoded with the `label_map`, and they are **visualized on the image**.

There are no one-size-fits-all values for `min_score`, `max_overlap`, and `top_k`. You may need to experiment a little to find what works best for your target data.

### Some more examples

---

<p align="center">
<img src="./img/000029.jpg">
</p>

---

<p align="center">
<img src="./img/000045.jpg">
</p>

---

<p align="center">
<img src="./img/000062.jpg">
</p>

---

<p align="center">
<img src="./img/000075.jpg">
</p>

---

<p align="center">
<img src="./img/000085.jpg">
</p>

---

<p align="center">
<img src="./img/000092.jpg">
</p>

---

<p align="center">
<img src="./img/000100.jpg">
</p>

---

<p align="center">
<img src="./img/000124.jpg">
</p>

---

<p align="center">
<img src="./img/000127.jpg">
</p>

---

<p align="center">
<img src="./img/000128.jpg">
</p>

---

<p align="center">
<img src="./img/000145.jpg">
</p>

---

# FAQs

__I noticed that priors often overshoot the `3, 3` kernel employed in the prediction convolutions. How can the kernel detect a bound (of an object) outside it?__

Don't confuse the kernel and its _receptive field_, which is the area of the original image that is represented in the kernel's field-of-view.

For example, on the `38, 38` feature map from `conv4_3`, a `3, 3` kernel covers an area of `0.08, 0.08` in fractional coordinates. The priors are `0.1, 0.1`, `0.14, 0.07`, `0.07, 0.14`, and `0.14, 0.14`.

But its receptive field, which [you can calculate](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807), is a whopping `0.36, 0.36`! Therefore, all priors (and objects contained therein) are present well inside it.

Keep in mind that the receptive field grows with every successive convolution. For `conv_7` and the higher-level feature maps, a `3, 3` kernel's receptive field will cover the _entire_ `300, 300` image. But, as always, the pixels in the original image that are closer to the center of the kernel have greater representation, so it is still _local_ in a sense.

---

__While training, why can't we match predicted boxes directly to their ground truths?__

We cannot directly check for overlap or coincidence between predicted boxes and ground truth objects to match them because predicted boxes are not to be considered reliable, _especially_ during the training process. This is the very reason we are trying to evaluate them in the first place!

And this is why priors are especially useful. We can match a predicted box to a ground truth box by means of the prior it is supposed to be approximating. It no longer matters how correct or wildly wrong the prediction is.

---

__Why do we even have a _background_ class if we're only checking which _non-background_ classes meet the threshold?__

When there is no object in the approximate field of the prior, a high score for _background_ will dilute the scores of the other classes such that they will not meet the detection threshold.

---

__Why not simply choose the class with the highest score instead of using a threshold?__

I think that's a valid strategy. After all, we implicitly conditioned the model to choose _one_ class when we trained it with the Cross Entropy loss. But you will find that you won't achieve the same performance as you would with a threshold.

I suspect this is because object detection is open-ended enough that there's room for doubt in the trained model as to what's really in the field of the prior. For example, the score for _background_ may be high if there is an appreciable amount of backdrop visible in an object's bounding box. There may even be multiple objects present in the same approximate region. A simple threshold will yield all possibilities for our consideration, and it just works better.

Redundant detections aren't really a problem since we're NMS-ing the hell out of 'em.


---

__Sorry, but I gotta ask... _[what's in the boooox?!](https://cnet4.cbsistatic.com/img/cLD5YVGT9pFqx61TuMtcSBtDPyY=/570x0/2017/01/14/6d8103f7-a52d-46de-98d0-56d0e9d79804/se7en.png)___

Ha.
