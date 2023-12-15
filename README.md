# final_project
오픈소스SW 파이널 프로젝트

-----------------------------------------------------------

### Project 설명

이 프로젝트의 목적은 brain tumor MRI 사진 데이터를 입력으로 받아

이 사진 데이터를 'glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor' 

이렇게 4개의 클래스 중 하나로 분류하는 classification을 만드는 것이다.

여기서 조건은 scikit-learn package만 사용해야 한다는 것이다. (scipy, skimage, keras, pytorch등 사용 금지)

-----------------------------------------------------------

### Dataset 설명

일단 tumor_dataset 파일에는 총 2870개의 MRI 사진들이 들어있는데

이 dataset은 매우 다양한 형태로 구성되어 있다는 것을 알 수 있다.

(ex. 정면, 위, 오른쪽, 왼쪽 등 여러 방향에서 찍은 사진들로 구성, 또한 object의 크기도 제각각)

한편, 이 각각의 사진들은 제공된 코드 안에서 데이터 전처리 과정을 거쳐

64x64 크기의 회색조(rgb 없이 오직 하나의 depth로 구성된 것) image로 변환된다.

```
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
```


즉, dataset X는 feature가 4096개인 2870개의 데이터로 구성되어 있다.
( X.shape >>> (2870, 4096) )

이제 이 dataset을 training data와 test data로(7 : 3 비율) 나눠서
학습을 진행한다.

```
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
```

-----------------------------------------------------------

### Algorithm 설명


앞서 말했듯이 이 MRI dataset은 MNIST같은 dataset과는 다르게 매우 다양한(정형화되어있지 않은) 형태로 구성되어있다.

(이런 경우 사실 CNN을 이용해 공간적 구조를 고려하는 알고리즘을 선택하는 것이 적합하다. 그러나 scikit-learn package만을 이용해야하는 이 프로젝트의 조건으로 인해 CNN 방법은 사용할 수 없다.)

먼저, 데이터가 정형화 되어있지 않으므로, 또한 분류에 있어서 중요하지 않은 feature(가령 검은색 배경)을 덜 고려하게끔 하기 위해
feature engineering 방법으로 PCA를 사용했다.

(PCA는 고차원 데이터의 주요 특징을 유지하면서 차원을 줄이는 것을 목표로 하는 차원 축소 기법이다.)

그리고 PCA를 진행한 후에 SVM을 사용해 분할을 진행하였다.

(SVM이란 클래스 사이의 마진(가장 가까운 데이터 포인트 사이의 거리)을 최대화하는
최적의 경계선(decision boundary)을 찾아 이를 통해 데이터를 분류하는 알고리즘이다.)

이 PCA와 SVM을 pipeline으로 묶어 하나의 classification 알고리즘을 만들었다.

-----------------------------------------------------------

### Hyper-parameter  설명

SVM의 주요 하이퍼파라미터로는 C, gamma, Kernel이 있는데

각각에 대해서 설명을 하자면

C는 모델의 복잡성을 제어하는 하이퍼파라미터로 오류를 얼마나 허용할지를 결정한다.

C가 크면 오류를 최대한 허용하지 않기 위해 decision boundary는 더욱 복잡해지고(overfitting의 위험성 증가)

C가 작으면 decision boundary는 직선에 가깝게 된다.

gamma는 rbf같은 비선형 커널의 반경을 결정하는 하이퍼파라미터로

C와 마찬가지로 gamma가 크면 각 샘플의 영향 범위가 좁아져 decision boundary가 더욱 복잡해지고(overfitting의 위험성 증가)

gamma가 작으면 decision boundary는 직선에 가깝게 된다.

Kernel은 데이터를 더 높은 차원의 공간으로 매핑하여, 비선형적으로 분리 가능하게 만드는 하이퍼파라미터이다.

이 3개의 하이퍼파라미터에 대해

나는 Kernel을 'Linear'가 아닌 'RBF'로 둬서 비선형 분리를 가능하게끔 설정하였고

하이퍼파라미터 C와 gamma에 대해서는 grid search를 통해 최적의 하이퍼 파라미터를 탐색하는 방법을 사용하였다.

'''
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('clf', SVC(kernel='rbf', probability=True))
])

# 하이퍼파라미터 그리드 설정
param_grid = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001],
}

# 그리드 서치로 최적의 하이퍼파라미터 탐색
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 최적의 파라미터 조합과 그때의 성능 출력
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validated accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

# 테스트 데이터셋에 대해 예측
y_pred = grid_search.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}%".format(accuracy * 100))
'''

(이 grid search를 해 본 결과 C = 10, gamma = 'scale'일 때가 최적의 결과로 나왔다.)







