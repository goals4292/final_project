# final_project
오픈소스SW 파이널 프로젝트

### 프로젝트 소개
이 프로젝트의 목적은 brain tumor MRI 사진 데이터를 입력으로 받아
이 사진 데이터를 'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor' 
이렇게 4개의 클래스 중 하나로 분류하는 classification을 만드는 것이다.
여기서 조건은 scikit-learn package만 사용해야 한다는 것이다. (scipy, skimage, keras, pytorch등 사용 금지)

### dataset 설명
일단 tumor_dataset 파일에는 총 2870개의 MRI 사진들이 들어있는데
이 dataset은 매우 다양한 형태로 구성되어 있다는 것을 알 수 있다.
(ex. 정면, 위, 오른쪽, 왼쪽 등 여러 방향에서 찍은 사진들로 구성, 또한 object의 크기도 제각각)

한편, 이 각각의 사진들은 제공된 코드 안에서 데이터 전처리 과정을 거쳐
64x64 크기의 회색조(rgb 없이 오직 하나의 depth로 구성된 것) image로 변환된다.

"""
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('./tumor_dataset/Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)
"""


즉, dataset은 feature가 4096개인 2870개의 데이터로 구성되어 있다.
( X.shape >>> (2870, 4096) )

이제 이 dataset을 training data와 test data로 7 : 3 비율로 나눠서
학습을 진행한다.


이 MRI dataset은 정면, 위, 오른쪽, 왼쪽 등 여러 방향에서 찍은 사진들로 구성되어있다.
즉, 이 MRI dataset은 MNIST같은 dataset과는 다르게 매우 다양한 형태로 구성되어있다는 의미다.
