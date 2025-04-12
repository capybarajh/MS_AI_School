import argparse

#### 인자값 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description="사용법 테스트")

#### 입력받을 인자값 등록 
parser.add_argument('--target', type=str, default="mode01", help="target is ????")
parser.add_argument('--epochs', type=int, default=100, help='total training epochs')

##### -> --nosave -> 값이 true
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')

### 입력받을 인자값을 args 에 저장
# args = parser.parse_args()
args = parser.parse_args(args=[])

###### 
epochs = args.epochs
target = args.target
print(epochs, target)